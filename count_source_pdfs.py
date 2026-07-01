"""
count_source_pdfs.py
---------------------
Scans source_data/ for zip archives (including nested zips) within
the adenda round folders, unzips them in place, counts all file
types found per round and per document type, and produces a stacked
bar chart visualising the file type breakdown per round.

Expected folder structure under source_data/:
    source_data/
    ├── adenda_round_1/   (may contain zip archives, possibly nested)
    ├── adenda_round_2/
    ├── adenda_round_3/
    ├── icsara_round_1/   (PDFs directly, no zips expected)
    ├── icsara_round_2/
    └── icsara_round_3/

Output:
    source_data_filetypes.png  — stacked bar chart, one bar per round/type
                                  combination, saved in source_data/

Usage:
    python count_source_pdfs.py

Dependencies:
    pip install matplotlib
"""

import zipfile
from pathlib import Path
from collections import Counter

BASE = Path(r"C:\Users\olesc\PycharmProjects\thesis")
SOURCE_DATA_DIR = BASE / "source_data"

ADENDA_FOLDERS = ["adenda_round_1", "adenda_round_2", "adenda_round_3"]
ICSARA_FOLDERS = ["icsara_round_1", "icsara_round_2", "icsara_round_3"]
ALL_FOLDERS    = ADENDA_FOLDERS + ICSARA_FOLDERS


def unzip_recursive(folder: Path, max_depth: int = 10):
    depth = 0
    found_any = True

    while found_any and depth < max_depth:
        found_any = False
        zip_files = list(folder.rglob("*.zip"))

        for zf_path in zip_files:
            extract_dir = zf_path.with_suffix("")
            if extract_dir.exists():
                continue

            try:
                with zipfile.ZipFile(zf_path, "r") as zf:
                    zf.extractall(extract_dir)
                print(f"    Unzipped: {zf_path.relative_to(folder)} -> {extract_dir.name}/")
                found_any = True
            except zipfile.BadZipFile:
                print(f"    Skipped (not a valid zip): {zf_path.relative_to(folder)}")

        depth += 1

    if depth >= max_depth:
        print(f"    Warning: reached max unzip depth ({max_depth}) — "
              f"there may still be nested zips remaining.")


def count_file_types(folder: Path) -> Counter:
    """Count all files by extension (lowercase, without the leading dot)."""
    counter = Counter()
    for f in folder.rglob("*"):
        if f.is_file():
            ext = f.suffix.lower().lstrip(".") or "(no extension)"
            counter[ext] += 1
    return counter


def print_file_type_breakdown(counter: Counter, indent: str = "    "):
    for ext, n in sorted(counter.items(), key=lambda x: -x[1]):
        print(f"{indent}.{ext}: {n}")


def main():
    if not SOURCE_DATA_DIR.exists():
        print(f"source_data/ not found at {SOURCE_DATA_DIR}")
        print(f"Create it and organise raw files into: {', '.join(ALL_FOLDERS)}")
        return

    filetype_counts = {}

    for sub in ADENDA_FOLDERS:
        folder = SOURCE_DATA_DIR / sub
        if not folder.exists():
            print(f"\n{sub}: folder not found — skipping")
            filetype_counts[sub] = Counter()
            continue

        print(f"\n{sub}:")
        print("  Checking for zip archives...")
        unzip_recursive(folder)

        types = count_file_types(folder)
        filetype_counts[sub] = types
        print(f"  File type breakdown:")
        print_file_type_breakdown(types)

    for sub in ICSARA_FOLDERS:
        folder = SOURCE_DATA_DIR / sub
        if not folder.exists():
            print(f"\n{sub}: folder not found — skipping")
            filetype_counts[sub] = Counter()
            continue

        types = count_file_types(folder)
        filetype_counts[sub] = types
        print(f"\n{sub}:")
        print(f"  File type breakdown:")
        print_file_type_breakdown(types)

    print("\n" + "=" * 60)
    print("SUMMARY — source PDFs and total files per round")
    print("=" * 60)
    print(f"  {'Round':<10} | {'Adenda PDFs':>12} | {'ICSARA PDFs':>12} | {'Total files':>12}")
    print("  " + "-" * 55)

    adenda_total, icsara_total, files_total = 0, 0, 0
    for r in ["round_1", "round_2", "round_3"]:
        a_types = filetype_counts.get(f"adenda_{r}", Counter())
        i_types = filetype_counts.get(f"icsara_{r}", Counter())
        a_pdf = a_types.get("pdf", 0)
        i_pdf = i_types.get("pdf", 0)
        round_files = sum(a_types.values()) + sum(i_types.values())

        adenda_total += a_pdf
        icsara_total += i_pdf
        files_total  += round_files

        print(f"  {r.replace('_', ' ').title():<10} | {a_pdf:>12} | {i_pdf:>12} | {round_files:>12}")

    print("  " + "-" * 55)
    print(f"  {'Total':<10} | {adenda_total:>12} | {icsara_total:>12} | {files_total:>12}")
    print(f"\n  Grand total: {adenda_total + icsara_total} source PDF(s), {files_total} file(s) overall")

    other_total = sum(
        n for types in filetype_counts.values()
        for ext, n in types.items() if ext != "pdf"
    )
    if other_total:
        print(f"\n  ({other_total} non-PDF file(s) also present, e.g. zip archives "
              f"and metadata — not part of the analysed corpus)")


if __name__ == "__main__":
    main()