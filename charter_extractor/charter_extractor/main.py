#!/usr/bin/env python3
"""CLI entry point for Charter Party Clause Extractor.
Requires Python 3.10+ (uses union type syntax: str | None)
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from charter_parser.models import ExtractionResult
from charter_parser.pdf_handler import PageRange, download_pdf, extract_text
from charter_parser.llm_extractor import extract_clauses

logger = logging.getLogger(__name__)


DEFAULT_PDF_URL = (
    "https://shippingforum.wordpress.com/wp-content/uploads/2012/09/"
    "voyage-charter-example.pdf"
)
DEFAULT_OUTPUT = "extracted_clauses.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract legal clauses from a maritime charter party PDF.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_PDF_URL,
        help="URL of the charter party PDF (default: voyage charter example)",
    )
    parser.add_argument(
        "--pdf",
        default=None,
        help="Path to a local PDF file (overrides --url if provided)",
    )
    parser.add_argument(
        "--pages",
        nargs=2,
        type=int,
        default=[6, 39],
        metavar=("START", "END"),
        help="1-indexed inclusive page range for Part II (default: 6 39)",
    )
    parser.add_argument(
        "--output", "-o",
        default=DEFAULT_OUTPUT,
        help=f"Output JSON file path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose/debug logging",
    )
    return parser


def run_pipeline(
    url: str,
    page_range: PageRange,
    output_path: Path,
    local_pdf: str | None = None,
) -> ExtractionResult:
    """Execute the full extraction pipeline: download → extract text → LLM → JSON."""

    model = os.getenv("GEMINI_MODEL")
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        logger.error(
            "ERROR: GEMINI_API_KEY not set. Add it to .env or export it."
        )
        sys.exit(1)
    if not model:
        logger.error(
            "ERROR: GEMINI_MODEL not set. Add it to .env (e.g. GEMINI_MODEL=your_gemini_model_name)",
        )
        sys.exit(1)

    # Step 1: Get PDF bytes
    if local_pdf:
        pdf_path = Path(local_pdf)
        if not pdf_path.exists():
            logger.error(f"ERROR: Local PDF not found: {pdf_path}")
            sys.exit(1)
        logger.info(f"[1/3] Reading local PDF: {pdf_path.name}...")
        pdf_bytes = pdf_path.read_bytes()
    else:
        logger.info(f"[1/3] Downloading PDF from {url[:80]}...")
        pdf_bytes = download_pdf(url)

    # Step 2: Extract text (with strikethrough filtering)
    logger.info(f"[2/3] Extracting text from pages {page_range.start}–{page_range.end}...")
    text = extract_text(pdf_bytes, page_range)

    if not text.strip():
        logger.error("ERROR: No text extracted from the specified pages.")
        sys.exit(1)

    char_count = len(text)
    approx_tokens = char_count // 4
    logger.info(f"Extracted {char_count:,} chars (~{approx_tokens:,} tokens)")

    # Step 3: LLM clause extraction
    logger.info(f"[3/3] Extracting clauses via {model}...")
    try:
        result = extract_clauses(
            text,
            api_key=api_key,
            model=model,
        )
    except Exception as e:
        logger.error(f"\nERROR during LLM extraction: {e}")
        sys.exit(1)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(result.to_json(indent=2), encoding="utf-8")
    logger.info(f"\n  Extracted {result.count} clauses → {output_path}")

    return result


def main() -> None:
    load_dotenv()

    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # Validate page range before creating PageRange (cleaner error message)
    if args.pages[0] < 1 or args.pages[1] < args.pages[0]:
        parser.error(f"Invalid page range: {args.pages[0]}-{args.pages[1]}")

    page_range = PageRange(start=args.pages[0], end=args.pages[1])
    output_path = Path(args.output)

    result = run_pipeline(
        url=args.url,
        page_range=page_range,
        output_path=output_path,
        local_pdf=args.pdf,
    )

    # Log summary
    logger.info("\n" + "=" * 60)
    logger.info("EXTRACTION SUMMARY")
    logger.info("=" * 60)
    for clause in result.clauses:
        preview = clause.text[:80].replace("\n", " ")
        logger.info(f"Clause {clause.id:>4s} | {clause.title[:40]:<40s} | {preview}...")
    logger.info("=" * 60)
    logger.info(f"Total: {result.count} clauses")


if __name__ == "__main__":
    main()
