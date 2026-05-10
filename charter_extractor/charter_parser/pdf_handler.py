"""PDF download, text extraction, and strikethrough filtering using PyMuPDF."""

from __future__ import annotations
import logging
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
    """Collect rectangles of strikethrough annotations on a page."""
    rects: list[fitz.Rect] = []

    # Method 1: PDF StrikeOut annotations (type code 12 in PyMuPDF)
    for annot in page.annots() or []:
        if annot.type[0] == 12:  # StrikeOut
            rects.append(annot.rect)

    # Method 2: Horizontal lines drawn through text (vector graphics)
    for drawing in page.get_drawings():
        for item in drawing.get("items", []):
            if item[0] != "l":  # not a line
                continue
            p1, p2 = fitz.Point(item[1]), fitz.Point(item[2])
            # Horizontal line: negligible vertical delta, meaningful width
            if abs(p1.y - p2.y) < 2 and abs(p1.x - p2.x) > 10:
                rect = fitz.Rect(
                    min(p1.x, p2.x), p1.y - 4,
                    max(p1.x, p2.x), p1.y + 4,
                )
                rects.append(rect)

    return rects


def _is_struck_through(span_rect: fitz.Rect, strike_rects: list[fitz.Rect]) -> bool:
    """Return True if *span_rect* overlaps significantly with any strikethrough rect."""
    for sr in strike_rects:
        intersection = span_rect & sr  # intersection
        if intersection.is_empty:
            continue
        # If the strikethrough line covers most of the span's width
        span_width = span_rect.width or 1
        overlap_ratio = intersection.width / span_width
        if overlap_ratio > 0.65:
            return True
    return False



# Text Extraction


def _extract_page_text(page: fitz.Page) -> str:
    """Extract text from a single page, filtering out struck-through spans."""
    strike_rects = _collect_strikeout_rects(page)
    blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
    lines_out: list[str] = []

    for block in blocks:
        if block.get("type") != 0:  # type 0 = text block; skip non-text (images, etc.)
            continue
        for line in block["lines"]:
            spans_text: list[str] = []
            for span in line["spans"]:
                span_rect = fitz.Rect(span["bbox"])
                if strike_rects and _is_struck_through(span_rect, strike_rects):
                    logger.debug("Filtered strikethrough text: %s", span["text"][:50])
                    continue
                spans_text.append(span["text"])
            joined = "".join(spans_text).rstrip()
            if joined:
                lines_out.append(joined)

    return "\n".join(lines_out)


def extract_text(
    pdf_bytes: bytes,
    page_range: PageRange = PART_II_PAGES,
) -> str:
    """Extract clean text from the given page range, excluding strikethrough text.

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

    pages_text: list[str] = []
    for page_num in range(page_range.start_0, min(page_range.end_0, total_pages)):
        page = doc[page_num]
        text = _extract_page_text(page)
        if text.strip():
            pages_text.append(f"--- PAGE {page_num + 1} ---\n{text}")

    doc.close()
    full_text = "\n\n".join(pages_text)
    logger.info("Extracted %d characters from %d pages", len(full_text), len(pages_text))
    return full_text
