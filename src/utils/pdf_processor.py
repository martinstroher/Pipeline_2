"""
PDF → Markdown converter using pymupdf4llm.

Scans a folder for PDFs and converts each to Markdown,
preserving headers and structure for downstream RAG indexing.
"""

import os
import pathlib

import pymupdf4llm

from src.utils import log


def process_folder(folder: str):
    """Convert all PDFs in *folder* to Markdown (.md) files.

    Skips files that already have a corresponding .md file.
    """
    if not folder.endswith("/"):
        folder += "/"

    if not os.path.exists(folder):
        log.error(f"Folder {folder} does not exist.")
        return

    pdfs = sorted(f for f in os.listdir(folder) if f.endswith(".pdf"))
    log.info(f"Found {len(pdfs)} PDF files in {folder}")

    converted = 0
    for pdf_name in pdfs:
        md_path = os.path.join(folder, pdf_name.replace(".pdf", ".md"))
        if os.path.exists(md_path):
            log.detail(f"Skipping (already converted): {pdf_name}")
            continue

        pdf_path = os.path.join(folder, pdf_name)
        try:
            md_text = pymupdf4llm.to_markdown(pdf_path)
            pathlib.Path(md_path).write_bytes(md_text.encode("utf-8"))
            converted += 1
            log.detail(f"Converted: {pdf_name}")
        except Exception as e:
            log.error(f"Failed to convert {pdf_name}: {e}")

    log.success(f"PDF conversion complete: {converted} new, {len(pdfs) - converted} skipped.")
