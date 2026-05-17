"""LLM-based clause extraction using Google Gemini (configurable model).

Uses the new google-genai SDK with:
- Text chunking to stay within model token limits
- Safety filter handling
- Retry logic for transient failures
"""

from __future__ import annotations

import json
import logging
import re
import time

from google import genai
from google.genai import types

from charter_parser.models import Clause, ExtractionResult

logger = logging.getLogger(__name__)

MAX_CHARS_PER_CHUNK = 200_000  # ~50K tokens per chunk (safe margin)
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds between retries

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
You are an expert maritime legal document parser.

Your task is to extract ALL numbered clauses from a Voyage Charter Party document.

DOCUMENT STRUCTURE:
This document contains MULTIPLE SECTIONS, each with its own numbering restarting from 1:

1. "Part II" — The standard printed clauses of the base charter form (e.g. Shellvoy 5).
   These are typically the FIRST set of numbered clauses, often in a standard/printed font.
   Usually clauses 1–44.

2. "Additional Clauses" — Typed additions that supplement or amend Part II clauses.
   Introduced by a heading such as "ADDITIONAL CLAUSES TO SHELLVOY 5" or similar.
   Numbering restarts from 1.

3. "Rider Clauses" — Further contractual additions, often introduced by headings like
   "ESSAR PROVISIONS" or "RIDER CLAUSES".
   Numbering restarts from 1 again.  Clause titles are often in ALL CAPITALS.

You MUST identify which section each clause belongs to and tag it in the "section" field.

CORE EXTRACTION RULES:

1. Extract every clause that starts with:
   <number>. <title>
   A clause is defined by its number + title and all following content until the next
   clause or section heading begins.

2. For each clause return:
   - "id":      clause number as a string
   - "title":   clause heading / title
   - "text":    FULL clause content (verbatim)
   - "section": one of "Part II", "Additional Clauses", or "Rider Clauses"

3. Clause content must include:
   - All paragraphs
   - Sub-clauses (a), (b), (c)
   - Roman numerals (i), (ii), (iii)
   - All continuation text until the next clause starts
   - Both original printed text AND handwritten/typed amendments that are NOT struck through

4. SECTION RULES:
   - The SAME clause number may appear in DIFFERENT sections — these are DIFFERENT clauses.
     Always include ALL of them with the correct section tag.
   - NEVER merge clauses from different sections.

5. WITHIN EACH SECTION, each clause ID should appear ONLY ONCE.
   - The struck-through text has already been removed from the input.
   - If you still see what looks like an original clause and an amendment with the
     same id in the same section, output ONLY the amendment (the one with
     handwritten / typed additions, often containing ALL CAPS text).
   - If two different sub-sections genuinely share an id but have completely
     unrelated titles and content, include both.

6. DO NOT:
   - Summarize or rewrite text
   - Merge clauses from different sections
   - Invent missing content
   - Change numbering
   - Alter wording
   - Include any text that was crossed out / struck through

7. CLEAN ONLY:
   Ignore page numbers, headers/footers, watermarks, OCR noise.

8. PRESERVE:
   - Original order of appearance within each section
   - Exact wording and formatting
   - Full legal structure

9. Clause boundary rule:
   A clause ends ONLY when a new line starting with <number>. <title> appears,
   a new section heading appears, or the document ends.

10. OUTPUT FORMAT — return ONLY valid JSON:

{
  "clauses": [
    {
      "id": "1",
      "title": "Condition Of vessel",
      "text": "Full clause text",
      "section": "Part II"
    }
  ]
}
"""

USER_PROMPT_TEMPLATE = """Extract all numbered legal clauses from the following charter party
document text.  The document contains multiple sections (Part II, Additional Clauses,
Rider Clauses) with numbering restarting in each section.  Tag every clause with its
correct section.  Return them as a JSON object with a "clauses" array.

Document text:
{text}"""

CHUNK_PROMPT_TEMPLATE = """Extract all numbered legal clauses from the following CHUNK
({chunk_num} of {total_chunks}) of a charter party document.
The document contains multiple sections (Part II, Additional Clauses, Rider Clauses)
with numbering restarting in each section.  Tag every clause with its correct section.
Return them as a JSON object with a "clauses" array.
If a clause started in a previous chunk and continues here, include the continuation
with the same clause id and the same section.
Document text (chunk {chunk_num}/{total_chunks}):
{text}"""


def _split_into_chunks(text: str, max_chars: int = MAX_CHARS_PER_CHUNK) -> list[str]:
    """Split text into chunks at page boundaries (--- PAGE N ---).

    Tries to split at page markers first; falls back to splitting at
    double-newlines if no page markers exist.
    """
    if len(text) <= max_chars:
        return [text]

    # Split at page boundaries
    pages = re.split(r"(?=--- PAGE \d+ ---)", text)
    pages = [p for p in pages if p.strip()]

    chunks: list[str] = []
    current_chunk: list[str] = []
    current_len = 0

    for page in pages:
        # If a single page is larger than the max, split the page into parts
        # so we never leave very large chunks to be sent to the LLM.
        if len(page) > max_chars:
            start = 0
            while start < len(page):
                part = page[start : start + max_chars]
                # Flush current chunk if adding this part would exceed limit
                if current_len + len(part) > max_chars and current_chunk:
                    chunks.append("".join(current_chunk))
                    current_chunk = []
                    current_len = 0
                current_chunk.append(part)
                current_len += len(part)

                if current_len >= max_chars:
                    chunks.append("".join(current_chunk))
                    current_chunk = []
                    current_len = 0

                start += max_chars
            # continue to next page
            continue

        # Regular page: if adding it would overflow, flush current chunk first
        if current_len + len(page) > max_chars and current_chunk:
            chunks.append("".join(current_chunk))
            current_chunk = []
            current_len = 0
        current_chunk.append(page)
        current_len += len(page)

    if current_chunk:
        chunks.append("".join(current_chunk))

    logger.info("Split %d chars into %d chunks", len(text), len(chunks))
    return chunks


def _call_gemini(
    client: genai.Client,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """Call Gemini with retry logic for transient failures."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    response_mime_type="application/json",
                ),
            )

            # Check for safety blocks
            if not response.candidates:
                block_reason = "unknown"
                if hasattr(response, "prompt_feedback") and response.prompt_feedback:
                    block_reason = getattr(
                        response.prompt_feedback, "block_reason", "unknown"
                    )
                raise ValueError(
                    f"Gemini blocked the request. Reason: {block_reason}"
                )

            candidate = response.candidates[0]
            if (
                candidate.finish_reason
                and hasattr(candidate.finish_reason, "name")
                and candidate.finish_reason.name == "SAFETY"
            ):
                raise ValueError(
                    f"Response blocked by safety filters: {candidate.safety_ratings}"
                )

            raw = response.text
            if not raw:
                raise ValueError("LLM returned an empty response")

            logger.info("Received %d chars from LLM (attempt %d)", len(raw), attempt)
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                logger.debug("Token usage: %s", response.usage_metadata)

            return raw

        except Exception as e:
            logger.warning("Attempt %d/%d failed: %s", attempt, MAX_RETRIES, e)
            if attempt == MAX_RETRIES:
                raise
            time.sleep(RETRY_DELAY * attempt)

    raise RuntimeError("All retries exhausted")  # should not reach here


def _parse_clauses(raw_content: str) -> list[Clause]:
    """Parse raw JSON response into a list of Clause objects."""
    try:
        result = ExtractionResult.model_validate_json(raw_content)
        return result.clauses
    except Exception:
        pass

    # Fallback: manual JSON parsing
    try:
        data = json.loads(raw_content)
        if "clauses" in data:
            return [Clause.model_validate(c) for c in data["clauses"]]
    except Exception:
        pass

    # Last resort: try to find JSON in the response
    try:
        start = raw_content.index("{")
        end = raw_content.rindex("}") + 1
        data = json.loads(raw_content[start:end])
        if "clauses" in data:
            return [Clause.model_validate(c) for c in data["clauses"]]
    except (ValueError, json.JSONDecodeError):
        pass

    raise ValueError(f"Could not parse LLM response as clause JSON: {raw_content[:200]}...")


def _deduplicate_clauses(clauses: list[Clause]) -> list[Clause]:
    """Merge / deduplicate clauses across chunks.

    Dedup key: ``(id, section, normalised_title)``.
    This correctly keeps clauses with the same number from *different*
    sections separate while merging continuations from chunk boundaries.

    When two clauses share an ID + section + title but have different text:
      - If one text is a subset of the other → keep the longer one.
      - If the second clause looks like a **replacement** (its opening words
        overlap the first clause's opening), keep only the later/longer one.
      - Otherwise concatenate (chunk-boundary continuation).
    """
    from difflib import SequenceMatcher

    def _norm(s: str) -> str:
        return re.sub(r"\s+", " ", s.strip().lower())

    def _is_replacement(existing_text: str, new_text: str) -> bool:
        """Detect if *new_text* is a replacement for *existing_text*
        rather than a continuation.

        Heuristics:
          1. Both start with the same sub-clause marker, e.g. "(1)", "(a)", etc.
          2. Their opening 80 chars share > 40 % similarity.
        """
        e_start = _norm(existing_text[:120])
        n_start = _norm(new_text[:120])
        ratio = SequenceMatcher(None, e_start, n_start).ratio()
        if ratio > 0.40:
            return True
        # Both begin with a numbered sub-clause marker
        marker = re.compile(r'^\(?\d+\)?[\s.):]')
        if marker.match(e_start) and marker.match(n_start):
            return True
        return False

    seen_by_key: dict[tuple[str, str, str], Clause] = {}
    result: list[Clause] = []

    for clause in clauses:
        key = (clause.id or "", clause.section or "", _norm(clause.title or ""))

        if key in seen_by_key:
            existing = seen_by_key[key]
            norm_new = _norm(clause.text)
            norm_existing = _norm(existing.text)

            # Exact duplicate — skip
            if norm_new == norm_existing:
                continue

            # New text already contained in existing — skip
            if norm_new in norm_existing:
                continue

            # Existing text is a subset of new — keep the longer one
            if norm_existing in norm_new:
                seen_by_key[key] = clause
                for i, e in enumerate(result):
                    if e is existing:
                        result[i] = clause
                        break
                continue

            # Replacement detection: if the new text restates the clause
            # (opening words overlap), prefer the one with more content.
            if _is_replacement(existing.text, clause.text):
                winner = clause if len(clause.text) >= len(existing.text) else existing
                if winner is not existing:
                    seen_by_key[key] = clause
                    for i, e in enumerate(result):
                        if e is existing:
                            result[i] = clause
                            break
                logger.info(
                    "Dedup: clause %s '%s' — kept replacement (%d chars over %d)",
                    clause.id, clause.title[:30],
                    len(winner.text), len((clause if winner is existing else existing).text),
                )
                continue

            # Otherwise concatenate (likely a chunk-boundary continuation)
            merged_text = existing.text.rstrip() + "\n" + clause.text.lstrip()
            merged = Clause(
                id=existing.id,
                title=existing.title,
                text=merged_text,
                section=existing.section,
            )
            seen_by_key[key] = merged
            for i, e in enumerate(result):
                if e is existing:
                    result[i] = merged
                    break
        else:
            seen_by_key[key] = clause
            result.append(clause)

    return result


# ---------------------------------------------------------------------------
# Post-extraction validation
# ---------------------------------------------------------------------------

def _validate_extraction(clauses: list[Clause]) -> list[Clause]:
    """Log warnings for suspicious patterns and clean up edge cases.

    Checks performed:
      - Duplicate (id, section) combinations with **different titles** — suggests
        a struck-through clause was not fully filtered (e.g. 41 TOVALOP → 41 ITOPF).
        The *later* clause (amendment) is kept; the earlier is dropped.
      - Empty or trivially short clause text (< 20 chars).
    """
    from collections import Counter

    # --- Detect same-id + same-section with different titles ------------------
    id_section_entries: dict[tuple[str, str], list[int]] = {}
    for idx, c in enumerate(clauses):
        key = (c.id or "", c.section or "")
        id_section_entries.setdefault(key, []).append(idx)

    indices_to_drop: set[int] = set()
    for (cid, sec), positions in id_section_entries.items():
        if len(positions) <= 1:
            continue
        titles = [clauses[i].title for i in positions]
        # All same title → already handled by _deduplicate_clauses
        unique_titles = set(t.strip().lower() for t in titles)
        if len(unique_titles) <= 1:
            continue
        # Different titles for same id in same section → keep the LAST one
        # (amendments/replacements come after the original in the document)
        logger.warning(
            "Clause id=%s section='%s' has %d entries with differing titles: %s — "
            "keeping last entry (amendment)",
            cid, sec, len(positions), titles,
        )
        for pos in positions[:-1]:
            indices_to_drop.add(pos)

    # --- Drop empty / trivially short clauses ---------------------------------
    for idx, c in enumerate(clauses):
        if len((c.text or "").strip()) < 20:
            logger.warning(
                "Dropping clause id=%s '%s' — text too short (%d chars)",
                c.id, c.title[:30], len(c.text or ""),
            )
            indices_to_drop.add(idx)

    if indices_to_drop:
        clauses = [c for i, c in enumerate(clauses) if i not in indices_to_drop]
        logger.info("Validation removed %d suspicious entries", len(indices_to_drop))

    return clauses


# ---------------------------------------------------------------------------
# Main Extraction Entry Point
# ---------------------------------------------------------------------------

def extract_clauses(
    text: str,
    *,
    api_key: str,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 65_536, 
) -> ExtractionResult:
    """Send extracted PDF text to Gemini and return structured clauses.

    Automatically chunks text if it exceeds model limits, processes each
    chunk separately, and merges/deduplicates results.

    Args:
        text: The raw text extracted from the PDF.
        api_key: Google Gemini API key.
        model: Model identifier.
        temperature: Sampling temperature (0 = deterministic).
        max_tokens: Max output tokens (default: 65,536 for Gemini 2.5 Flash).

    Returns:
        ExtractionResult with all parsed clauses.
    """
    client = genai.Client(api_key=api_key)
    chunks = _split_into_chunks(text)

    all_clauses: list[Clause] = []

    if len(chunks) == 1:
        # Single chunk — simple case
        prompt = USER_PROMPT_TEMPLATE.format(text=text)
        logger.info(
            "Sending %d chars to %s (approx %d tokens)",
            len(prompt), model, len(prompt) // 4,
        )
        raw = _call_gemini(client, model, prompt, temperature, max_tokens)
        all_clauses = _parse_clauses(raw)
        # Deduplicate even for a single chunk — the LLM may still emit
        # the same clause twice (e.g. original + amendment with same id/title).
        all_clauses = _deduplicate_clauses(all_clauses)
    else:
        # Multiple chunks — process each and merge
        for i, chunk in enumerate(chunks, 1):
            prompt = CHUNK_PROMPT_TEMPLATE.format(
                chunk_num=i, total_chunks=len(chunks), text=chunk
            )
            logger.info(
                "Sending chunk %d/%d (%d chars, approx %d tokens) to %s",
                i, len(chunks), len(prompt), len(prompt) // 4, model,
            )
            raw = _call_gemini(client, model, prompt, temperature, max_tokens)
            chunk_clauses = _parse_clauses(raw)
            logger.info("Chunk %d/%d: extracted %d clauses", i, len(chunks), len(chunk_clauses))
            all_clauses.extend(chunk_clauses)

        # Deduplicate clauses that may span chunk boundaries
        all_clauses = _deduplicate_clauses(all_clauses)

    result = ExtractionResult(clauses=all_clauses)
    logger.info("Total after dedup: %d clauses", result.count)

    # Post-extraction validation — catch remaining edge cases
    validated = _validate_extraction(result.clauses)
    result = ExtractionResult(clauses=validated)
    logger.info("Total after validation: %d clauses", result.count)
    return result
