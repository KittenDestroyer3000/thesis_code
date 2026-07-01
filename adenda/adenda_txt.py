import os

import pdfplumber
from pdf2image import convert_from_path
import pytesseract
from pathlib import Path
from tqdm.auto import tqdm
import json
import easyocr
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
POPPLER_PATH = r"C:\Program Files\Poppler\Library\bin"

pdf_dir = Path(r"/ADENDA_round_2")
out_dir = Path(r"C:\Users\olesc\PycharmProjects\thesis\output_addenda")
progress_dir = Path(r"/progress")
out_dir.mkdir(exist_ok=True)
progress_dir.mkdir(exist_ok=True)

MIN_CHARS = 50
DPI = 200
OCR_DPI = 150  # separate from your general DPI

reader = easyocr.Reader(["es"], gpu=True)  # "es" for Spanish


def extract_page_text(pdf_path, page_index, plumber_page):
    text = plumber_page.extract_text() or ""
    if len(text.strip()) >= MIN_CHARS:
        return text, "text"
    images = convert_from_path(
        pdf_path, dpi=OCR_DPI,
        first_page=page_index + 1,
        last_page=page_index + 1,
        poppler_path=POPPLER_PATH
    )
    img_array = np.array(images[0])
    result = reader.readtext(img_array, detail=0, paragraph=True, batch_size=8)
    return "\n".join(result), "OCR-GPU"


_position_lock = threading.Lock()
_positions = set()


def get_position():
    with _position_lock:
        pos = next(i for i in range(3, 20) if i not in _positions)
        _positions.add(pos)
        return pos


def release_position(pos):
    with _position_lock:
        _positions.discard(pos)


import io


def process_pdf(pdf_path):
    relative = pdf_path.relative_to(pdf_dir)
    out_file = (out_dir / relative).with_suffix(".txt")
    progress_file = (progress_dir / relative).with_suffix(".json")

    out_file.parent.mkdir(parents=True, exist_ok=True)
    progress_file.parent.mkdir(parents=True, exist_ok=True)

    if out_file.exists():
        return

    if progress_file.exists():
        try:
            with open(progress_file, encoding='utf-8') as f:
                progress = json.load(f)
            pages_text = progress["pages_text"]
            start_page = progress["next_page"]
        except (json.JSONDecodeError, KeyError):
            progress_file.unlink()
            pages_text = []
            start_page = 0
    else:
        pages_text = []
        start_page = 0

    pos = get_position()
    try:
        # Read PDF as bytes, wrap in BytesIO for pdfplumber
        with open(pdf_path, 'rb') as f:
            pdf_bytes = f.read()

        pdf_stream = io.BytesIO(pdf_bytes)

        with pdfplumber.open(pdf_stream) as pdf:
            total = len(pdf.pages)
            with tqdm(range(start_page, total), desc=relative.name[:40], position=pos,
                      leave=False, initial=start_page, total=total) as pbar:
                for i in pbar:
                    page = pdf.pages[i]
                    page_text, method = extract_page_text(pdf_bytes, i, page)
                    separator = "=" * 80
                    page_marker = "\n" + separator + "\nPAGE " + str(i + 1) + "\n" + separator + "\n"
                    pages_text.append(page_marker + page_text)
                    pbar.set_postfix(method=method)

                    if i % 5 == 0:
                        temp_file = progress_file.with_suffix(".json.tmp")
                        with open(temp_file, "w", encoding='utf-8') as f:
                            json.dump({"pages_text": pages_text, "next_page": i + 1}, f)
                        os.replace(temp_file, progress_file)

        out_file.write_text("".join(pages_text), encoding="utf-8")
        progress_file.unlink(missing_ok=True)
        return

    except Exception as e:
        return f"✗ Failed: {relative}: {str(e)}"

    finally:
        release_position(pos)


def extract_page_text(pdf_bytes, page_index, plumber_page):
    text = plumber_page.extract_text() or ""
    if len(text.strip()) >= MIN_CHARS:
        return text, "text"

    # For OCR, write bytes to temp file (pdf2image requires path)
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
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


if __name__ == "__main__":
    all_pdfs = list(pdf_dir.rglob("*.pdf"))
    WORKERS = 6

    outer = tqdm(total=len(all_pdfs), desc="Total PDFs", position=0)

with ThreadPoolExecutor(max_workers=WORKERS) as executor:
    futures = {executor.submit(process_pdf, p): p for p in all_pdfs}
    completed = 0
    for future in as_completed(futures):
        result = future.result()
        if result:
            tqdm.write(result)
        completed += 1
        outer.update(1)
        outer.set_description(f"Total PDFs ({completed}/{len(all_pdfs)})")

    outer.close()