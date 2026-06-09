"""Document loading & text extraction (ported from the prototype).

Faithful port of ``brief/P10_DSML/utils/data_loader.py``: every extractor is
kept (PDF / DOCX / TXT / CSV / Excel / ZIP). Two minimal, output-neutral changes:

1. **OCR is lazy-loaded.** The prototype initialised EasyOCR *at import time*
   (pulling ~GB of models and slowing every startup / test). Here the reader is
   built on first actual OCR use, and absent OCR deps degrade gracefully.
2. **PyPDF2 -> pypdf.** PyPDF2 is archived/unmaintained; ``pypdf`` is its
   maintained successor with an identical ``PdfReader`` API — same extraction
   method, just the living package.

Heavy/optional libraries (pypdf, pandas, python-docx, easyocr, requests) are
imported *inside* their functions so a missing one only disables that one path.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# --- OCR (lazy) ---------------------------------------------------------

_ocr_reader: Any | None = None
_ocr_unavailable = False


def _get_ocr_reader() -> Any | None:
    """Build the EasyOCR reader on first use; cache it. ``None`` if unavailable.

    This is the key fix vs the prototype: nothing OCR-related runs at import.
    """
    global _ocr_reader, _ocr_unavailable
    if _ocr_reader is not None or _ocr_unavailable:
        return _ocr_reader
    try:
        import easyocr  # heavy: pulls torch

        logger.info("Initialising EasyOCR reader (first OCR use)...")
        _ocr_reader = easyocr.Reader(["en", "fr"])
        logger.info("EasyOCR reader ready.")
    except Exception as exc:  # noqa: BLE001 - any failure -> OCR simply off
        logger.warning("OCR unavailable (%s). PDFs will use text extraction only.", exc)
        _ocr_unavailable = True
    return _ocr_reader


# --- Text extractors ----------------------------------------------------

def extract_text_from_pdf_with_ocr(file_path: str) -> str | None:
    """Extract text from a PDF via OCR (EasyOCR). ``None`` if OCR is unavailable."""
    reader = _get_ocr_reader()
    if reader is None:
        return None
    try:
        import fitz  # PyMuPDF
        import numpy as np
        from PIL import Image
    except Exception as exc:  # noqa: BLE001
        logger.warning("OCR imaging deps missing (%s); skipping OCR.", exc)
        return None

    text_content: list[str] = []
    try:
        doc = fitz.open(file_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x for OCR quality
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            try:
                results = reader.readtext(np.array(img))
                text_content.append("\n".join(res[1] for res in results))
            except Exception as ocr_exc:  # noqa: BLE001
                logger.error("OCR failed on page %d of %s: %s", page_num + 1, file_path, ocr_exc)
                continue
        doc.close()
        full_text = "\n".join(text_content).strip()
        if full_text:
            logger.info("OCR extracted %d chars from %s", len(full_text), file_path)
            return full_text
        logger.warning("OCR produced no significant text for %s", file_path)
        return None
    except Exception as exc:  # noqa: BLE001
        logger.error("OCR processing failed for %s: %s", file_path, exc)
        return None


def extract_text_from_pdf(file_path: str) -> str | None:
    """Extract text from a PDF, falling back to OCR when little text is found."""
    try:
        from pypdf import PdfReader

        reader = PdfReader(file_path)
        text = "".join(
            page.extract_text() + "\n" for page in reader.pages if page.extract_text()
        )
        if len(text.strip()) < 100:  # likely a scanned PDF -> try OCR
            logger.info("Little text in %s (%d chars); trying OCR...", file_path, len(text.strip()))
            ocr_text = extract_text_from_pdf_with_ocr(file_path)
            return ocr_text if ocr_text else text
        logger.info("Extracted %d chars from PDF %s", len(text), file_path)
        return text
    except Exception as exc:  # noqa: BLE001
        logger.error("PDF extraction failed for %s: %s; trying OCR...", file_path, exc)
        return extract_text_from_pdf_with_ocr(file_path)


def extract_text_from_docx(file_path: str) -> str | None:
    """Extract text from a Word DOCX file."""
    try:
        import docx

        doc = docx.Document(file_path)
        text = "\n".join(para.text for para in doc.paragraphs if para.text)
        logger.info("Extracted %d chars from DOCX %s", len(text), file_path)
        return text
    except Exception as exc:  # noqa: BLE001
        logger.error("DOCX extraction failed for %s: %s", file_path, exc)
        return None


def extract_text_from_txt(file_path: str) -> str | None:
    """Extract text from a plain-text file."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        logger.info("Extracted %d chars from TXT %s", len(text), file_path)
        return text
    except Exception as exc:  # noqa: BLE001
        logger.error("TXT extraction failed for %s: %s", file_path, exc)
        return None


def extract_text_from_csv(file_path: str) -> str | None:
    """Extract text from a CSV (flattened to a string via pandas)."""
    try:
        import pandas as pd

        try:
            df = pd.read_csv(file_path)
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding="latin1")
        text = df.to_string()
        logger.info("Extracted %d chars from CSV %s", len(text), file_path)
        return text
    except Exception as exc:  # noqa: BLE001
        logger.error("CSV extraction failed for %s: %s", file_path, exc)
        return None


def extract_text_from_excel(file_path: str) -> str | dict[str, str] | None:
    """Extract text from each sheet of an Excel file (flattened via pandas).

    NOTE: flattening numeric tables to text is exactly the prototype behaviour
    that yields poor answers to numeric questions. We keep it on purpose so the
    RAGAS *baseline* exhibits that failure; the same Excel is later loaded into
    SQL (Phase 3) for precise numeric retrieval.
    """
    try:
        import pandas as pd

        excel_file = pd.ExcelFile(file_path)
        sheets = {name: excel_file.parse(name).to_string() for name in excel_file.sheet_names}
        logger.info("Extracted %d sheet(s) from Excel %s", len(sheets), file_path)
        if len(sheets) == 1:
            return next(iter(sheets.values()))
        return sheets
    except Exception as exc:  # noqa: BLE001
        logger.error("Excel extraction failed for %s: %s", file_path, exc)
        return None


# --- Loaders ------------------------------------------------------------

def download_and_extract_zip(url: str, output_dir: str) -> bool:
    """Download a ZIP from ``url`` and extract it into ``output_dir``."""
    if not url:
        logger.warning("No URL provided for download.")
        return False
    try:
        import io
        import zipfile

        import requests

        logger.info("Downloading data from %s...", url)
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            z.extractall(output_dir)
        logger.info("Download and extraction complete.")
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error("Download/extraction failed: %s", exc)
        return False


# Maps a file suffix to its extractor.
_EXTRACTORS = {
    ".pdf": extract_text_from_pdf,
    ".docx": extract_text_from_docx,
    ".txt": extract_text_from_txt,
    ".csv": extract_text_from_csv,
    ".xlsx": extract_text_from_excel,
    ".xls": extract_text_from_excel,
}


def load_and_parse_files(input_dir: str | Path) -> list[dict[str, Any]]:
    """Recursively load & parse files under ``input_dir``.

    Returns a list of documents ``{"page_content": str, "metadata": {...}}``,
    the format expected by the splitter. An Excel with several sheets yields one
    document per sheet.
    """
    documents: list[dict[str, Any]] = []
    input_path = Path(input_dir)
    if not input_path.is_dir():
        logger.error("Input directory '%s' does not exist.", input_dir)
        return []

    logger.info("Scanning source directory: %s", input_dir)
    for file_path in input_path.rglob("*.*"):
        if not file_path.is_file():
            continue
        relative_path = file_path.relative_to(input_path)
        source_folder = relative_path.parts[0] if len(relative_path.parts) > 1 else "root"
        extractor = _EXTRACTORS.get(file_path.suffix.lower())
        if extractor is None:
            logger.warning("Unsupported file type ignored: %s", relative_path)
            continue

        content = extractor(str(file_path))
        if not content:
            logger.warning("No content extracted from %s", relative_path)
            continue

        base_meta = {
            "filename": file_path.name,
            "category": source_folder,
            "full_path": str(file_path.resolve()),
        }
        if isinstance(content, dict):  # multi-sheet Excel -> one doc per sheet
            for sheet_name, text in content.items():
                documents.append({
                    "page_content": text,
                    "metadata": {
                        **base_meta,
                        "source": f"{relative_path} (Sheet: {sheet_name})",
                        "sheet": sheet_name,
                    },
                })
        else:
            documents.append({
                "page_content": content,
                "metadata": {**base_meta, "source": str(relative_path)},
            })

    logger.info("%d document(s) loaded and parsed.", len(documents))
    return documents
