"""
parse_icsara.py
---------------
Parses the Spanish ICSARA txt file into a structured CSV.

Each numbered regulatory item (1), 2), 3)...) becomes one row, with:
  - item_number   : integer
  - section       : Roman numeral section heading
  - text_es       : cleaned Spanish text of the item
  - word_count    : word count of the item text

Strips:
  - Digital signature boilerplate (SEA validador URLs)
  - Placeholder tags like <NUM_ICSARA>, <CIUDAD_FECHA_INFORME>, <FIRMA_DIREC>

Usage:
    python parse_icsara.py                        # uses DEFAULT_INPUT below
    python parse_icsara.py path/to/round_2.txt    # or supply path as argument
"""

import re
import csv
import sys
import os
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────────

DEFAULT_INPUT   = r"C:\Users\olesc\PycharmProjects\thesis\icsara\data\round_2.txt"

# ── Noise patterns ─────────────────────────────────────────────────────────────

NOISE_PATTERNS = [
    r"Para validar las firmas de este documento usted debe ingresar a la siguiente url\s*https?://\S*",
    r"Para validar las firmas[^\n]*",
    r"https?://validador\.sea\.gob\.cl/\S*",
    r"<NUM_ICSARA>",
    r"<CIUDAD_FECHA_INFORME>",
    r"<FIRMA_DIREC>",
    r"<[A-Z_]+>",
]

# ── Section header pattern ─────────────────────────────────────────────────────
# Matches Roman numeral headers at start of line:
# e.g. "I. DESCRIPCIÓN DEL PROYECTO O ACTIVIDAD"

SECTION_PATTERN = re.compile(
    r"^((?:X{0,3})(?:IX|IV|V?I{0,3})\.\s+[A-ZÁÉÍÓÚÑÜ][^\n]{3,})",
    re.MULTILINE
)

# ── Item pattern ───────────────────────────────────────────────────────────────
# CRITICAL FIX: match ONLY digits followed by ) at the START of a line
# (with optional leading whitespace). This excludes:
#   - sub-items using letters: a), b), c)
#   - sub-items using Roman numerals: i), ii), iii)
#   - numbered sub-points using periods: 1. 2. 3.
#   - numbers mid-sentence like "pregunta n°9"

ITEM_PATTERN = re.compile(
    r"^[ \t]*(\d{1,3})\)[ \t]+",
    re.MULTILINE
)

# ── Helpers ────────────────────────────────────────────────────────────────────

def strip_noise(text: str) -> str:
    for pattern in NOISE_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def extract_sections(text: str) -> list:
    matches = list(SECTION_PATTERN.finditer(text))
    if not matches:
        return [("PREAMBLE", text)]

    sections = []
    if matches[0].start() > 0:
        sections.append(("PREAMBLE", text[:matches[0].start()]))

    for i, match in enumerate(matches):
        label = match.group(1).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append((label, text[start:end]))

    return sections


def extract_items(section_label: str, section_text: str) -> list:
    matches = list(ITEM_PATTERN.finditer(section_text))
    if not matches:
        cleaned = strip_noise(section_text)
        if cleaned and len(cleaned.split()) > 10:
            return [{
                "item_number": None,
                "section": section_label,
                "text_es": cleaned,
                "word_count": len(cleaned.split()),
            }]
        return []

    rows = []
    for i, match in enumerate(matches):
        item_num = int(match.group(1))
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(section_text)
        raw_text = section_text[start:end]
        cleaned = strip_noise(raw_text)
        if cleaned and len(cleaned.split()) > 5:
            rows.append({
                "item_number": item_num,
                "section": section_label,
                "text_es": cleaned,
                "word_count": len(cleaned.split()),
            })
    return rows


def parse_icsara(input_path: str) -> list:
    input_path = Path(input_path)
    print(f"Reading: {input_path}")

    with open(input_path, encoding="utf-8") as f:
        raw = f.read()

    print(f"  Raw file: {len(raw):,} characters")

    sections = extract_sections(raw)
    print(f"  Sections found: {len(sections)}")

    all_rows = []
    for section_label, section_text in sections:
        rows = extract_items(section_label, section_text)
        all_rows.extend(rows)
        print(f"  [{section_label[:60]}] → {len(rows)} items")

    # Sort: preamble first, then by item number
    all_rows.sort(key=lambda r: (r["item_number"] is None, r["item_number"] or 0))

    print(f"\n  Total rows: {len(all_rows)}")
    return all_rows


def write_csv(rows: list, input_path: str) -> str:
    # Derive output filename from input: round_3.txt -> icsara_items_round_1.csv
    stem = Path(input_path).stem  # e.g. "round_1"
    output_filename = f"icsara_items_{stem}.csv"
    output_path = Path(input_path).parent / output_filename
    fieldnames = ["item_number", "section", "text_es", "word_count"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV written: {output_path}")
    return str(output_path)


def main():
    input_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT

    if not os.path.exists(input_path):
        print(f"ERROR: File not found: {input_path}")
        sys.exit(1)

    rows = parse_icsara(input_path)

    if not rows:
        print("WARNING: No items extracted.")
        sys.exit(1)

    # Deduplicate: where item_number appears more than once, keep the longest text
    seen = {}
    duplicates = []
    for row in rows:
        n = row["item_number"]
        if n is None:
            continue
        if n in seen:
            duplicates.append(n)
            if row["word_count"] > seen[n]["word_count"]:
                seen[n] = row  # replace with longer version
        else:
            seen[n] = row

    if duplicates:
        print(f"\n  Duplicates found and resolved (kept longer): item(s) {sorted(set(duplicates))}")

    # Rebuild rows: deduplicated numbered items + any unnumbered rows
    unnumbered = [r for r in rows if r["item_number"] is None]
    rows = unnumbered + sorted(seen.values(), key=lambda r: r["item_number"])

    write_csv(rows, input_path)

    numbered = [r for r in rows if r["item_number"] is not None]
    if numbered:
        max_item  = max(r["item_number"] for r in numbered)
        min_item  = min(r["item_number"] for r in numbered)
        avg_words = sum(r["word_count"] for r in numbered) / len(numbered)
        item_nums = sorted(set(r["item_number"] for r in numbered))
        expected  = set(range(min_item, max_item + 1))
        missing   = sorted(expected - set(item_nums))

        print(f"\nDiagnostics:")
        print(f"  Item range        : {min_item} – {max_item}")
        print(f"  Numbered items    : {len(numbered)}")
        print(f"  Avg words/item    : {avg_words:.0f}")
        if missing:
            print(f"  Missing numbers   : {missing[:20]}{'...' if len(missing)>20 else ''}")
        else:
            print(f"  Missing numbers   : none — sequence is complete")
        sections = list(dict.fromkeys(r["section"] for r in rows))
        print(f"  Sections ({len(sections)}):")
        for s in sections:
            print(f"    {s}")


if __name__ == "__main__":
    main()