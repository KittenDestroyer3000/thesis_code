"""
translate_icsara.py
-------------------
Translates the 'text_es' column of icsara_items CSV from Spanish to English
using the OpenAI API (gpt-4o-mini), one item per call to avoid separator
parsing failures.

Produces a translated CSV with an added 'text_en' column.
The original Spanish text is preserved.

Usage:
    set OPENAI_API_KEY=your_key_here          (Windows)
    python translate_icsara.py --input data\icsara_items_round_1.csv --output data\icsara_items_round_1_translated.csv
    python translate_icsara.py --resume       (continue interrupted run)

Dependencies:
    pip install openai pandas tqdm
"""

import argparse
import os
import time
import pandas as pd
from tqdm import tqdm


# ── Configuration ──────────────────────────────────────────────────────────────

MODEL       = "gpt-4o-mini"
MAX_RETRIES = 4
RETRY_DELAY = 10
SAVE_EVERY  = 20

DEFAULT_INPUT  = r"C:\Users\olesc\PycharmProjects\thesis\icsara\data\icsara_items_round_2.csv"
DEFAULT_OUTPUT = r"C:\Users\olesc\PycharmProjects\thesis\icsara\data\icsara_items_round_2_translated.csv"

SYSTEM_PROMPT = """You are a professional translator specialising in \
environmental impact assessments, mining engineering, and Chilean \
environmental regulation documents.
Translate the Spanish regulatory item into English.

Rules:
- Preserve technical terminology accurately
- Preserve regulation codes exactly as written (e.g. D.S. N°40/2012, NCh 1333, PAS 138)
- Preserve proper nouns, place names, species names, and acronyms (e.g. BNP, GHPPI, ZE, ZD, ZAV, PTAS, SMA, SEA)
- Do not add explanations or commentary
- Output ONLY the translated text, nothing else
"""


# ── Translation ────────────────────────────────────────────────────────────────

def translate_one(client, text: str) -> str:
    """Translate a single ICSARA item. Returns English text or original on failure."""
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                max_tokens=2048,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": text},
                ],
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e):
                if attempt < MAX_RETRIES - 1:
                    print(f"\n  Rate limit — waiting {RETRY_DELAY}s...")
                    time.sleep(RETRY_DELAY)
                else:
                    raise
            elif attempt < MAX_RETRIES - 1:
                print(f"\n  Attempt {attempt+1} failed ({e}) — retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
            else:
                print(f"\n  Item failed after {MAX_RETRIES} attempts. Keeping Spanish.")
                return text


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Translate ICSARA items CSV Spanish->English via OpenAI API."
    )
    parser.add_argument("--input",  default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true",
                        help="Resume from a partially translated output file")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY not set.\n"
            "Run:  set OPENAI_API_KEY=your_key_here   (Windows)"
        )

    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    # Load data
    print(f"Loading '{args.input}'...")
    df = pd.read_csv(args.input)
    total = len(df)
    print(f"  {total} items loaded")

    if "text_es" not in df.columns:
        raise ValueError("Expected column 'text_es' not found. Check input file.")

    # Resume logic — check for existing partial output
    if args.resume and os.path.exists(args.output):
        done = pd.read_csv(args.output)
        if "text_en" in done.columns:
            df["text_en"] = done["text_en"]
            already = df["text_en"].notna().sum()
            print(f"  {already} already translated, {total - already} remaining")
        else:
            df["text_en"] = None
    else:
        df["text_en"] = None

    # Find items still needing translation
    todo_idx = df.index[df["text_en"].isna()].tolist()

    if not todo_idx:
        print("Nothing to translate — all rows already have text_en.")
        return

    print(f"Translating {len(todo_idx)} items one at a time...\n")

    for i, idx in enumerate(tqdm(todo_idx, unit="item")):
        text = str(df.at[idx, "text_es"])
        df.at[idx, "text_en"] = translate_one(client, text)

        if (i + 1) % SAVE_EVERY == 0:
            df.to_csv(args.output, index=False, encoding="utf-8")
            tqdm.write(f"  Progress saved at item {i + 1}")

    # Final save
    df.to_csv(args.output, index=False, encoding="utf-8")
    translated = df["text_en"].notna().sum()
    print(f"\nDone. {translated}/{total} items translated.")
    print(f"Saved to '{args.output}'")

    # Cost estimate (gpt-4o-mini)
    total_words = df["word_count"].sum()
    est_tokens  = total_words * 1.3
    est_cost    = (est_tokens / 1_000_000) * 0.60
    print(f"Estimated cost: ~${est_cost:.3f}")


if __name__ == "__main__":
    main()