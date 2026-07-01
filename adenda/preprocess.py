"""
preprocess.py
-------------
Converts raw EIA text files (with === PAGE X === separators) into
clean chunks with metadata, saved as chunks.csv.

Usage:
    python preprocess.py --input_dir ./docs --output chunks.csv

Dependencies:
    pip install pandas
"""

import os
import re
import argparse
import pandas as pd


# ---------------------------------------------------------------------------
# Boilerplate patterns to strip
# These repeat across pages/files and carry no topical signal
# ---------------------------------------------------------------------------
BOILERPLATE_PATTERNS = [
    # Running header: document title repeated on every page
    r"Adenda Ciudadana\s*\nAdenda - Proyecto de Desarrollo Minero.*?Tierras Raras",
    r"Anexo \d+\s*\n.*?Tierras Raras",
    # Consultant signature block
    r"Elaborado por[:\s]*\nGesti[oó]n Ambiental Consultores.*?www\.gac\.cl",
    r"Elaborado para[:\s]*\nGesti[oó]n Ambiental Consultores.*?www\.gac\.cl",
    # Copyright line
    r"©Gestión Ambiental Consultores.*?reservados\.",
    # Table of contents entries  (e.g. "Tabla 12. Medidas propuestas .... 53")
    r"^Tabla \d+[\.\-].*?\.{3,}\s*\d+\s*$",
    # Index entries (section number + dots + page number)
    r"^\d+(\.\d+)*\s+[A-ZÁÉÍÓÚ].*?\.{3,}\s*\d+\s*$",
    # Lone page numbers / roman numerals on their own line
    r"^\s*(i{1,4}|v|vi{1,3}|ix|x{1,3}|\d{1,3})\s*$",
    # Cartographic metadata blocks
    r"DATOS CARTOGRÁFICOS.*?ELABORADO PARA:",
    r"Coordenadas UTM.*?Huso \d+[SN]",
    r"Datum WGS \d+.*",
    r"Source: Esri.*",
    r"Esri, TomTom.*",
    # Scale / lámina lines
    r"ESCALA\s*\nLÁMINA.*",
    r"Tamaño Hoja:.*",
    r"LÁMINA[:\s]*\d+.*",
    # "Fuente: GAC" attribution lines
    r"^Fuente:\s*(GAC|Elaboración Propia).*$",
    # Revision table rows in technical annexes (e.g. "A 23-11-2022 DA / JQ RM RM ...")
    r"^[A-Z]\s+\d{2}-\d{2}-\d{4}\s+.*$",
    # "Rev.0" / revision markers
    r"^Rev\.\s*\d+\s*$",
]

# Compile once for speed
BOILERPLATE_RE = [
    re.compile(p, re.MULTILINE | re.DOTALL | re.IGNORECASE)
    for p in BOILERPLATE_PATTERNS
]

# Lines that are purely structural noise (after stripping)
NOISE_LINES = {
    "indice general", "indice de tablas", "indice de figuras",
    "índice general", "índice de tablas", "índice de figuras",
    "índice", "indice",
}


def strip_boilerplate(text: str) -> str:
    for pattern in BOILERPLATE_RE:
        text = pattern.sub(" ", text)
    return text


def clean_text(text: str) -> str:
    """Remove boilerplate, normalise whitespace, drop noise lines."""
    text = strip_boilerplate(text)
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lower() in NOISE_LINES:
            continue
        lines.append(stripped)
    return " ".join(lines)


def split_into_pages(raw: str) -> list[dict]:
    """
    Split on the === PAGE N === separator produced by the PDF-to-text pipeline.
    Returns a list of dicts: {page_num: int, text: str}
    """
    pattern = re.compile(
        r"={10,}\s*\nPAGE\s+(\d+)\s*\n={10,}",
        re.IGNORECASE
    )
    parts = pattern.split(raw)
    # parts = [pre_text, page_num, page_text, page_num, page_text, ...]
    pages = []
    # parts[0] is any text before the first page marker — skip it
    for i in range(1, len(parts) - 1, 2):
        page_num = int(parts[i])
        page_text = parts[i + 1].strip()
        if page_text:
            pages.append({"page_num": page_num, "text": page_text})
    return pages


def pages_to_chunks(
    pages: list[dict],
    min_tokens: int = 40,
    max_tokens: int = 300,
) -> list[dict]:
    """
    Merge consecutive pages into chunks that stay within [min_tokens, max_tokens].
    A rough token estimate is used (words / 0.75).

    Returns list of dicts: {start_page, end_page, text}
    """
    chunks = []
    buffer_pages = []
    buffer_words = 0

    def flush():
        if not buffer_pages:
            return
        combined = " ".join(p["clean"] for p in buffer_pages)
        chunks.append({
            "start_page": buffer_pages[0]["page_num"],
            "end_page":   buffer_pages[-1]["page_num"],
            "text":       combined,
        })

    for page in pages:
        cleaned = clean_text(page["text"])
        word_count = len(cleaned.split())

        # Skip pages that are too thin after cleaning (mostly boilerplate)
        if word_count < 10:
            continue

        page["clean"] = cleaned

        # If adding this page would exceed max_tokens, flush first
        if buffer_words + word_count > max_tokens and buffer_pages:
            flush()
            buffer_pages = []
            buffer_words = 0

        buffer_pages.append(page)
        buffer_words += word_count

        # If we've accumulated enough, flush
        if buffer_words >= min_tokens:
            flush()
            buffer_pages = []
            buffer_words = 0

    # Flush remainder
    flush()

    return chunks


def process_file(filepath: str) -> list[dict]:
    filename = os.path.basename(filepath)
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        raw = f.read()

    pages = split_into_pages(raw)
    chunks = pages_to_chunks(pages)

    records = []
    for i, chunk in enumerate(chunks):
        records.append({
            "doc_id":     filename,
            "chunk_id":   f"{filename}__chunk_{i:04d}",
            "start_page": chunk["start_page"],
            "end_page":   chunk["end_page"],
            "text":       chunk["text"],
            "word_count": len(chunk["text"].split()),
        })
    return records


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess EIA .txt files into chunks for topic modelling."
    )
    parser.add_argument(
        "--input_dir", default=".",
        help="Directory containing .txt files (default: current directory)"
    )
    parser.add_argument(
        "--output", default="chunks.csv",
        help="Output CSV path (default: chunks.csv)"
    )
    parser.add_argument(
        "--min_tokens", type=int, default=40,
        help="Minimum words per chunk before flushing (default: 40)"
    )
    parser.add_argument(
        "--max_tokens", type=int, default=300,
        help="Maximum words per chunk (default: 300)"
    )
    args = parser.parse_args()

    from pathlib import Path as _Path
    txt_files = sorted(_Path(args.input_dir).rglob("*.txt"))

    if not txt_files:
        print(f"No .txt files found in '{args.input_dir}' (searched recursively)")
        return

    print(f"Found {len(txt_files)} file(s) across all subfolders. Processing...")

    all_records = []
    for fp in txt_files:
        records = process_file(str(fp))
        print(f"  {fp.name:60s} -> {len(records):4d} chunks")
        all_records.extend(records)

    df = pd.DataFrame(all_records)
    df.to_csv(args.output, index=False, encoding="utf-8")
    print(f"\nSaved {len(df)} chunks to '{args.output}'")
    print(f"Word count stats:\n{df['word_count'].describe().round(1).to_string()}")


if __name__ == "__main__":
    main()