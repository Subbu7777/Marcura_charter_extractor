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

Your task is to extract ALL numbered clauses from PART II of a Voyage Charter Party agreement.

CORE EXTRACTION RULES:

1. Extract every clause that starts with:
   <number>. <title>
   Example:
   1. Condition Of vessel
   2. Cleanliness Of tanks

2. A clause is defined by its number + title and all following content
   until the next clause begins.

3. For each clause return:
   - "id": clause number as string
   - "title": clause heading/title
   - "text": FULL clause content (verbatim)

4. Clause content must include:
   - paragraphs
   - sub-clauses (a), (b), (c)
   - roman numerals (i), (ii), (iii)
   - all continuation text until next clause starts

5. IMPORTANT DEDUPLICATION RULE:
   - If the SAME clause (same id + same title + same text) appears multiple times,
     include it ONLY ONCE in the output.
   - If same id appears but text differs, treat them as separate clauses and include both.

6. DO NOT:
   - summarize or rewrite text
   - merge different clauses
   - remove valid repeated clauses with different content
   - invent missing content
   - change numbering
   - alter wording

7. CLEAN ONLY:
   Ignore:
   - page numbers
   - headers/footers
   - watermarks
   - OCR noise
   - crossed-out text

8. PRESERVE:
   - original order of first appearance
   - exact wording and formatting
   - full legal structure

9. Clause boundary rule:
   A clause ends ONLY when a new line starting with:
   <number>. <title>
   appears OR PART II ends.

10. OUTPUT FORMAT:
Return ONLY valid JSON:

{
  "clauses": [
    {
      "id": "1",
      "title": "Condition Of vessel",
      "text": "Full clause text"
    }
  ]
}
"""

USER_PROMPT_TEMPLATE = """Extract all numbered legal clauses from the following charter party
document text. Return them as a JSON object with a "clauses" array.

Document text:
{text}"""

CHUNK_PROMPT_TEMPLATE = """Extract all numbered legal clauses from the following CHUNK
({chunk_num} of {total_chunks}) of a charter party document.
Return them as a JSON object with a "clauses" array.
If a clause started in a previous chunk and continues here, include the continuation
with the same clause id.
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
    """Merge duplicate clauses from chunked processing.

    If multiple chunks produce the same clause ID, merge them by
    concatenating text (for clauses split across chunks).
    """
    # Merge only when both id and title match. Preserve separate entries
    # for the same id with different titles.
    seen_by_key: dict[tuple[str, str], Clause] = {}
    result: list[Clause] = []

    for clause in clauses:
        key = (clause.id or "", clause.title or "")
        if key in seen_by_key:
            existing = seen_by_key[key]
            # If text differs and is not already contained, append continuation
            if clause.text and clause.text not in existing.text:
                merged_text = existing.text.rstrip() + "\n" + clause.text.lstrip()
                merged = Clause(id=existing.id, title=existing.title, text=merged_text)
                seen_by_key[key] = merged
                # update the entry in result (preserve position)
                for i, e in enumerate(result):
                    if e.id == existing.id and e.title == existing.title:
                        result[i] = merged
                        break
        else:
            seen_by_key[key] = clause
            result.append(clause)

    return result


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
    logger.info("Total extracted: %d clauses", result.count)
    return result
