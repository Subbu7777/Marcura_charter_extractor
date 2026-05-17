"""PDF download, text extraction, and strikethrough filtering using PyMuPDF."""

from __future__ import annotations
import logging
import re
from collections import Counter
from dataclasses import dataclass
import fitz  # PyMuPDF
import requests

logger = logging.getLogger(__name__)


# Configuration

@dataclass(frozen=True)
class PageRange:
    """Inclusive 1-based page range (matches human-readable page numbers)."""

    start: int  
    end: int   

    def __post_init__(self) -> None:
        if self.start < 1:
            raise ValueError(f"start page must be >= 1, got {self.start}")
        if self.end < self.start:
            raise ValueError(f"end page ({self.end}) must be >= start page ({self.start})")

    @property
    def start_0(self) -> int: 
        return self.start - 1

    @property
    def end_0(self) -> int:
        return self.end  # exclusive upper bound

# Default range for Part II of the voyage charter party
PART_II_PAGES = PageRange(start=6, end=39)


# PDF Download

def download_pdf(url: str, *, timeout: int = 60) -> bytes:
    """Download a PDF from *url* and return the raw bytes.

    Raises:
        requests.HTTPError: If the download fails.
    """
    logger.info("Downloading PDF from %s", url)
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    content_type = response.headers.get("Content-Type", "")
    if "pdf" not in content_type and not url.endswith(".pdf"):
        logger.warning("Response Content-Type is '%s'; expected PDF", content_type)
    logger.info("Downloaded %d bytes", len(response.content))
    return response.content


# Strikethrough Detection


def _collect_strikeout_rects(page: fitz.Page) -> list[fitz.Rect]:
    """Collect rectangles of strikethrough annotations on a page.

    Handles three strikethrough mechanisms common in charter-party PDFs:

      1. Standard PDF *StrikeOut* annotations  (annotation subtype 12).
      2. Thin horizontal **lines** drawn through text  (vector-graphic ``l`` items).
      3. Thin filled / stroked **rectangles** drawn over text  (vector-graphic
         ``re`` items) — the **dominant** mechanism in many annotated PDFs and
         previously unhandled.
    """
    rects: list[fitz.Rect] = []
    n_annot = n_line = n_rect = 0

    # --- Method 1: PDF StrikeOut annotations (type code 12) -------------------
    for annot in page.annots() or []:
        if annot.type[0] == 12:  # StrikeOut
            # Prefer QuadPoints for precise per-line coverage
            try:
                vertices = annot.vertices
                if vertices and len(vertices) >= 4:
                    # QuadPoints: groups of 4 points, each defining a quadrilateral
                    for qi in range(0, len(vertices) - 3, 4):
                        xs = [vertices[qi + j].x for j in range(4)]
                        ys = [vertices[qi + j].y for j in range(4)]
                        qr = fitz.Rect(min(xs), min(ys), max(xs), max(ys))
                        if not qr.is_empty:
                            rects.append(qr)
                            n_annot += 1
                    continue
            except Exception:
                pass
            # Fallback to annotation bounding box
            rects.append(annot.rect)
            n_annot += 1

    # --- Methods 2 & 3: vector-graphic lines and thin rectangles --------------
    for drawing in page.get_drawings():
        fill     = drawing.get("fill")          # fill colour (tuple) or None
        color    = drawing.get("color")         # stroke colour (tuple) or None
        stroke_w = drawing.get("width", 0)      # stroke width in pt

        for item in drawing.get("items", []):
            kind = item[0]

            # Method 2 — horizontal lines
            if kind == "l":
                p1, p2 = fitz.Point(item[1]), fitz.Point(item[2])
                if abs(p1.y - p2.y) < 2 and abs(p1.x - p2.x) > 10:
                    # Pad vertically by half the stroke width (min 1.5 pt)
                    pad = max(stroke_w / 2, 1.5)
                    mid_y = (p1.y + p2.y) / 2
                    rect = fitz.Rect(
                        min(p1.x, p2.x), mid_y - pad - 1,
                        max(p1.x, p2.x), mid_y + pad + 1,
                    )
                    rects.append(rect)
                    n_line += 1

            # Method 3 — thin filled / stroked rectangles
            elif kind == "re":
                rect = fitz.Rect(item[1])
                # A strikethrough bar is thin (< 8 pt) and wide (> 10 pt).
                # It must be filled or stroked to be visible.
                if (
                    rect.width > 10
                    and rect.height < 8
                    and (fill is not None or color is not None or stroke_w > 0)
                ):
                    rects.append(rect)
                    n_rect += 1

    if rects:
        logger.debug(
            "Page strike candidates: %d total (annot=%d, line=%d, rect=%d)",
            len(rects), n_annot, n_line, n_rect,
        )
    return rects


def _is_struck_through(span_rect: fitz.Rect, strike_rects: list[fitz.Rect]) -> bool:
    """Return True if *span_rect* is overlapped by a strikethrough indicator.

    Two conditions must both be met:

      1. **Horizontal coverage** — the strike rect covers >= 50 % of the span width.
      2. **Vertical position** — the vertical centre of the strike rect falls
         within the middle band (15 %–85 %) of the span height.  This filters
         out underlines (very bottom) and overlines (very top) that could
         otherwise satisfy condition 1.
    """
    span_width  = span_rect.width  or 1
    span_height = span_rect.height or 1

    for sr in strike_rects:
        intersection = span_rect & sr  # intersection
        if intersection.is_empty:
            continue

        # 1) horizontal coverage
        if intersection.width / span_width < 0.50:
            continue

        # 2) vertical position — 0.0 = top (ascent), 1.0 = bottom (descent)
        sr_v_centre = (sr.y0 + sr.y1) / 2
        v_ratio = (sr_v_centre - span_rect.y0) / span_height
        if 0.15 <= v_ratio <= 0.85:
            return True

    return False



# Text Extraction

# Standalone page number on its own line (e.g. "- 15 -" or just "7")
_PAGE_NUM_RE = re.compile(r'^[\s\-\–\—]*\d{1,3}[\s\-\–\—]*$')


def _normalize_line(line: str) -> str:
    """Collapse multiple spaces and strip isolated bullet characters."""
    line = re.sub(r'[\u2022\u25CF\u25A0\u25A1\u25E6\u25AA\u25BA\u25B8\u2023\u2043]\s*', '', line)
    line = re.sub(r' {2,}', ' ', line)
    return line.strip()


def _extract_page_text(page: fitz.Page) -> str:
    """Extract text from a single page, filtering struck-through spans.

    Improvements over a naive extraction:
      - Text blocks are sorted into visual reading order (top-to-bottom, left-to-right).
      - A blank line is emitted between blocks to preserve paragraph structure.
      - Each line is normalised (collapsed whitespace, bullet artefacts removed).
      - Standalone page-number lines are dropped.
    """
    strike_rects = _collect_strikeout_rects(page)
    raw_blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]

    # Keep only text blocks and sort into visual reading order
    text_blocks = sorted(
        [b for b in raw_blocks if b.get("type") == 0],
        key=lambda b: (round(b["bbox"][1], 1), b["bbox"][0]),
    )

    lines_out: list[str] = []

    for block in text_blocks:
        # Paragraph separator between blocks (avoid double-blanks)
        if lines_out and lines_out[-1] != "":
            lines_out.append("")

        for line in block["lines"]:
            spans_text: list[str] = []
            for span in line["spans"]:
                span_rect = fitz.Rect(span["bbox"])
                if strike_rects and _is_struck_through(span_rect, strike_rects):
                    logger.debug("Filtered strikethrough text: %s", span["text"][:50])
                    continue
                spans_text.append(span["text"])
            joined = "".join(spans_text).rstrip()
            normalised = _normalize_line(joined)
            if not normalised:
                continue
            # Drop standalone page numbers
            if _PAGE_NUM_RE.match(normalised):
                continue
            lines_out.append(normalised)

    return "\n".join(lines_out)


def _remove_headers_footers(
    pages: list[tuple[int, str]],
    *,
    look_lines: int = 2,
    min_page_ratio: float = 0.5,
    max_line_len: int = 80,
) -> list[tuple[int, str]]:
    """Remove short lines that repeat across many pages (likely headers/footers).

    Only lines shorter than *max_line_len* characters that appear in at least
    *min_page_ratio* of pages (at the top or bottom *look_lines*) are removed.
    """
    if len(pages) < 4:
        return pages

    threshold = int(len(pages) * min_page_ratio)
    top_counts: Counter[str] = Counter()
    bot_counts: Counter[str] = Counter()

    for _, text in pages:
        non_empty = [l.strip() for l in text.split("\n") if l.strip()]
        for line in non_empty[:look_lines]:
            norm = re.sub(r'\s+', ' ', line.lower())
            if 2 < len(norm) <= max_line_len:
                top_counts[norm] += 1
        for line in non_empty[-look_lines:]:
            norm = re.sub(r'\s+', ' ', line.lower())
            if 2 < len(norm) <= max_line_len:
                bot_counts[norm] += 1

    to_remove = {l for l, c in top_counts.items() if c >= threshold}
    to_remove |= {l for l, c in bot_counts.items() if c >= threshold}

    if to_remove:
        logger.info("Removing %d detected header/footer patterns", len(to_remove))

    cleaned: list[tuple[int, str]] = []
    for page_num, text in pages:
        out_lines = [
            line for line in text.split("\n")
            if re.sub(r'\s+', ' ', line.strip().lower()) not in to_remove
        ]
        cleaned.append((page_num, "\n".join(out_lines)))
    return cleaned


def _final_normalize(text: str) -> str:
    """Collapse excessive blank lines and trailing whitespace."""
    text = re.sub(r'\n{3,}', '\n\n', text)   # max 2 consecutive newlines
    text = re.sub(r'[ \t]+\n', '\n', text)    # strip line-trailing whitespace
    return text.strip()


def extract_text(
    pdf_bytes: bytes,
    page_range: PageRange = PART_II_PAGES,
) -> str:
    """Extract clean text from the given page range, excluding strikethrough text.

    Pipeline:
      1. Per-page extraction with strike filtering and line normalisation.
      2. Cross-page header / footer removal.
      3. Final whitespace normalisation.

    Args:
        pdf_bytes: Raw PDF file bytes.
        page_range: 1-indexed inclusive page range to extract.

    Returns:
        Concatenated text of all pages in the range.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    total_pages = doc.page_count
    logger.info("PDF has %d pages; extracting pages %d–%d", total_pages, page_range.start, page_range.end)

    if page_range.end_0 > total_pages:
        logger.warning(
            "Requested end page %d exceeds document length %d; clamping",
            page_range.end, total_pages,
        )

    # Step 1 — per-page extraction
    raw_pages: list[tuple[int, str]] = []
    for page_num in range(page_range.start_0, min(page_range.end_0, total_pages)):
        page = doc[page_num]
        text = _extract_page_text(page)
        if text.strip():
            raw_pages.append((page_num, text))
    doc.close()

    # Step 2 — header / footer removal
    cleaned_pages = _remove_headers_footers(raw_pages)

    # Step 3 — assemble with page markers
    pages_text: list[str] = []
    for page_num, text in cleaned_pages:
        pages_text.append(f"--- PAGE {page_num + 1} ---\n{text}")

    full_text = "\n\n".join(pages_text)

    # Step 4 — final normalisation
    full_text = _final_normalize(full_text)

    logger.info("Extracted %d characters from %d pages", len(full_text), len(cleaned_pages))
    return full_text
