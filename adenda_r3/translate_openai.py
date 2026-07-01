"""
translate_openai.py
-------------------
Drop-in replacement for translate.py using the OpenAI API (gpt-4o-mini).
Designed to resume a partially completed Anthropic translation run.

Reads an existing (partial) translated CSV, finds all rows where
text_en is null, and translates only those — leaving already-translated
rows untouched.

Usage:
    set OPENAI_API_KEY=your_key_here          (Windows)
    python translate_openai.py                 (uses defaults below)
    python translate_openai.py --input chunks_r1.csv --output chunks_r3_translated.csv

Dependencies:
    pip install openai pandas tqdm
"""

import argparse
import os
import time
import pandas as pd
from tqdm import tqdm

# ── Configuration ──────────────────────────────────────────────────────────────

MODEL      = "gpt-4o-mini"
BATCH_SIZE = 5
MAX_RETRIES = 4
RETRY_DELAY = 10
SAVE_EVERY  = 50

DEFAULT_INPUT  = r"C:\Users\olesc\PycharmProjects\thesis\adenda_r3\chunks_r3.csv"
DEFAULT_OUTPUT = r"C:\Users\olesc\PycharmProjects\thesis\adenda_r3\chunks_r1_translated.csv"

SYSTEM_PROMPT = """You are a professional translator specialising in \
environmental impact assessments and mining engineering documents.
Translate the Spanish text into English.

Rules:
- Preserve technical terminology accurately (e.g. proper nouns,
  regulation codes like D.S. Nº38/2011, chemical formulas)
- Do not add explanations or commentary
- Output ONLY the translated text, nothing else
"""


# ── Translation ────────────────────────────────────────────────────────────────

def translate_one(client, text: str) -> str:
    """Translate a single chunk. Returns English text or original on failure."""
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
                print(f"\n  Chunk failed after {MAX_RETRIES} attempts. Keeping Spanish.")
                return text


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Translate chunks CSV Spanish->English via OpenAI API."
    )
    parser.add_argument("--input",      default=DEFAULT_INPUT)
    parser.add_argument("--output",     default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    # API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY not set.\n"
            "Run:  set OPENAI_API_KEY=your_key_here   (Windows)"
        )

    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    # Load original chunks
    print(f"Loading '{args.input}'...")
    df = pd.read_csv(args.input)
    total = len(df)
    print(f"  {total} total chunks")

    # Load partial output if it exists, merge text_en into df
    if os.path.exists(args.output):
        done = pd.read_csv(args.output)
        if "text_en" in done.columns:
            df["text_en"] = done["text_en"]
            already = df["text_en"].notna().sum()
            print(f"  {already} already translated, {total - already} remaining")
        else:
            df["text_en"] = None
    else:
        df["text_en"] = None

    # Find indices still needing translation
    todo_idx = df.index[df["text_en"].isna()].tolist()

    if not todo_idx:
        print("Nothing to translate — all rows already have text_en.")
        return

    print(f"Translating {len(todo_idx)} chunks one at a time...\n")

    for i, idx in enumerate(tqdm(todo_idx, unit="chunk")):
        text = df.at[idx, "text"]
        df.at[idx, "text_en"] = translate_one(client, text)

        if (i + 1) % SAVE_EVERY == 0:
            df.to_csv(args.output, index=False, encoding="utf-8")
            tqdm.write(f"  Progress saved at chunk {i + 1}")

    # Final save
    df.to_csv(args.output, index=False, encoding="utf-8")
    translated = df["text_en"].notna().sum()
    print(f"\nDone. {translated}/{total} chunks translated.")
    print(f"Saved to '{args.output}'")

    # Cost estimate (gpt-4o-mini: ~$0.15/1M input, $0.60/1M output tokens)
    remaining_words = df.loc[todo_idx, "text"].str.split().str.len().sum()
    est_tokens = remaining_words * 1.3
    est_cost   = (est_tokens / 1_000_000) * 0.60
    print(f"Estimated cost: ~${est_cost:.2f} (rough estimate)")


if __name__ == "__main__":
    main()
