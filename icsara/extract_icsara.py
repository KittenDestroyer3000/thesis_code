"""
extract_icsara.py
-----------------
Extracts clean Spanish text from the ICSARA PDF.

Consistent with addenda_txt.py:
  - pdfplumber as primary extractor
  - EasyOCR (GPU, Spanish) as fallback for image-heavy pages
  - Same MIN_CHARS threshold for deciding when to OCR

Differences from addenda_txt.py (intentional):
  - Single file in, single file out — no threading or batch logic needed
  - No page markers in output — we want continuous text for parse_icsara.py
  - Boilerplate stripped at extraction time (SEA signature validation strings)
  - Output is a single clean round_3.txt

Output: icsara/data/round_3.txt  (same folder as input PDF by default)

Usage:
    python extract_icsara.py                            # uses DEFAULT_INPUT below
    python extract_icsara.py path\\to\\icsara.pdf       # or supply path as argument
"""

import os
import re
import sys
import io
import tempfile
import pytesseract
import numpy as np
import easyocr
import pdfplumber
from pdf2image import convert_from_path
from pathlib import Path
from tqdm.auto import tqdm

# ── Configuration — adjust paths to match your setup ──────────────────────────

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
POPPLER_PATH = r"C:\Program Files\Poppler\Library\bin"

DEFAULT_INPUT  = r"C:\Users\olesc\PycharmProjects\thesis\icsara\data\icsara_es.pdf"
OUTPUT_FILENAME = "round_3.txt"

MIN_CHARS = 50    # minimum characters before falling back to OCR (same as addenda_txt.py)
OCR_DPI   = 150   # same as addenda_txt.py

# ── Boilerplate patterns to strip ─────────────────────────────────────────────
# The SEA (Chilean environmental regulator) embeds a digital-signature validation
# string mid-document. It appears in two forms; both are stripped.

NOISE_PATTERNS = [
    r"Para validar las firmas de este documento usted debe ingresar a la siguiente url\s*https?://\S*",
    r"Para validar las firmas[^\n]*",
    r"https?://validador\.sea\.gob\.cl/\S*",
]

# ── OCR reader — initialised once ─────────────────────────────────────────────

print("Loading EasyOCR (Spanish, GPU)...")
reader = easyocr.Reader(["es"], gpu=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def strip_noise(text: str) -> str:
    """Remove SEA boilerplate strings from extracted text."""
    for pattern in NOISE_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
    # Collapse runs of blank lines to a single blank line
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_page(pdf_bytes: bytes, page_index: int, plumber_page) -> tuple[str, str]:
    """
    Extract text from a single page.
    Returns (text, method) where method is 'text' or 'OCR-GPU'.
    Consistent with addenda_txt.py logic.
    """
    text = plumber_page.extract_text() or ""
    if len(text.strip()) >= MIN_CHARS:
        return text, "text"

    # Fall back to EasyOCR — write bytes to temp file (pdf2image needs a path)
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        images = convert_from_path(
            tmp_path, dpi=OCR_DPI,
            first_page=page_index + 1,
            last_page=page_index + 1,
            poppler_path=POPPLER_PATH
        )
        img_array = np.array(images[0])
        result = reader.readtext(img_array, detail=0, paragraph=True, batch_size=8)
        return "\n".join(result), "OCR-GPU"
    finally:
        os.unlink(tmp_path)


# ── Main ──────────────────────────────────────────────────────────────────────

def extract_icsara(pdf_path: str) -> str:
    pdf_path = Path(pdf_path)
    print(f"\nReading PDF: {pdf_path.name}")

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    pdf_stream = io.BytesIO(pdf_bytes)
    pages_text = []
    ocr_count = 0

    with pdfplumber.open(pdf_stream) as pdf:
        total = len(pdf.pages)
        print(f"Pages: {total}")

        with tqdm(range(total), desc="Extracting pages") as pbar:
            for i in pbar:
                page = pdf.pages[i]
                page_text, method = extract_page(pdf_bytes, i, page)
                pages_text.append(page_text)
                if method == "OCR-GPU":
                    ocr_count += 1
                pbar.set_postfix(method=method)

    print(f"\nExtraction complete — {ocr_count}/{total} pages used OCR fallback")

    # Join all pages, strip boilerplate, return
    full_text = "\n\n".join(pages_text)
    clean_text = strip_noise(full_text)
    return clean_text


def main():
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT

    if not os.path.exists(pdf_path):
        print(f"ERROR: File not found: {pdf_path}")
        print("Usage: python extract_icsara.py [path\\to\\icsara_es.pdf]")
        sys.exit(1)

    clean_text = extract_icsara(pdf_path)

    output_path = Path(pdf_path).parent / OUTPUT_FILENAME
    output_path.write_text(clean_text, encoding="utf-8")

    print(f"\nOutput written : {output_path}")
    print(f"Characters     : {len(clean_text):,}")
    print(f"Lines          : {clean_text.count(chr(10)):,}")
    print("\nFirst 500 characters preview:")
    print("-" * 60)
    print(clean_text[:500])
    print("-" * 60)
    print("\nNext step: python parse_icsara.py")


if __name__ == "__main__":
    main()