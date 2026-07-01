"""
translate.py
------------
Translates the 'text' column of chunks.csv from Spanish to English
using the Anthropic API (claude-haiku-4-5).

Produces chunks_translated.csv — identical to chunks.csv but with
an added 'text_en' column. The original Spanish text is preserved.

Usage:
    set ANTHROPIC_API_KEY=your_key_here        (Windows)
    python translate.py --input chunks.csv --output chunks_translated.csv

Dependencies:
    pip install anthropic pandas tqdm
"""

import anthropic
import argparse
import os
import re
import time
import pandas as pd
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL         = "claude-haiku-4-5"
BATCH_SIZE    = 5        # chunks per API call — smaller = more reliable
MAX_RETRIES   = 4
RETRY_DELAY   = 10
SAVE_EVERY    = 50

SEPARATOR = "---TRANSLATION_END---"

SYSTEM_PROMPT = f"""You are a professional translator specialising in 
environmental impact assessments and mining engineering documents. 
Translate each Spanish text chunk into English.

Rules:
- Preserve technical terminology accurately (e.g. proper nouns, 
  regulation codes like D.S. Nº38/2011, chemical formulas)
- Do not add explanations or commentary
- Separate each translated chunk with exactly this string on its own line:
  {SEPARATOR}
- Output ONLY the translations separated by that string, nothing else
"""


def translate_batch(client: anthropic.Anthropic, texts: list[str]) -> list[str]:
    """Send a batch of Spanish texts, return list of English translations."""

    numbered = f"\n{SEPARATOR}\n".join(texts)

    user_message = (
        f"Translate each of these {len(texts)} chunks. "
        f"Separate your translations with {SEPARATOR} on its own line.\n\n"
        f"{numbered}"
    )

    for attempt in range(MAX_RETRIES):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )

            raw = response.content[0].text.strip()

            # Split on separator
            parts = re.split(rf"\s*{re.escape(SEPARATOR)}\s*", raw)
            translations = [p.strip() for p in parts if p.strip()]

            if len(translations) != len(texts):
                raise ValueError(
                    f"Expected {len(texts)} translations, got {len(translations)}"
                )

            return translations

        except anthropic.RateLimitError:
            if attempt < MAX_RETRIES - 1:
                print(f"\n  Rate limit — waiting {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
            else:
                raise

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                print(f"\n  Attempt {attempt+1} failed ({e}) — retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
            else:
                print(f"\n  Batch failed after {MAX_RETRIES} attempts. Keeping Spanish.")
                return texts


def main():
    parser = argparse.ArgumentParser(
        description="Translate chunks.csv Spanish->English via Anthropic API."
    )
    parser.add_argument("--input",      default="chunks.csv")
    parser.add_argument("--output",     default="chunks_translated.csv")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--resume",     action="store_true",
                        help="Resume from a partially translated output file")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # API key
    # ------------------------------------------------------------------
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY not set.\n"
            "Run:  set ANTHROPIC_API_KEY=your_key_here   (Windows)\n"
            "  or: export ANTHROPIC_API_KEY=your_key_here (Mac/Linux)"
        )
    client = anthropic.Anthropic(api_key=api_key)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print(f"Loading '{args.input}'...")
    df = pd.read_csv(args.input)
    total = len(df)
    print(f"  {total} chunks loaded")

    # ------------------------------------------------------------------
    # Resume logic
    # ------------------------------------------------------------------
    start_idx = 0
    if args.resume and os.path.exists(args.output):
        done = pd.read_csv(args.output)
        already_done = done["text_en"].notna().sum()
        df["text_en"] = None
        df.loc[:already_done - 1, "text_en"] = done.loc[:already_done - 1, "text_en"]
        start_idx = already_done
        print(f"  Resuming from chunk {start_idx} ({already_done} already translated)")
    else:
        df["text_en"] = None

    # ------------------------------------------------------------------
    # Translate in batches
    # ------------------------------------------------------------------
    indices  = list(range(start_idx, total))
    batches  = [indices[i: i + args.batch_size]
                for i in range(0, len(indices), args.batch_size)]

    print(f"Translating {len(indices)} chunks in {len(batches)} batches "
          f"(batch size: {args.batch_size})...\n")

    for batch_num, batch_idx in enumerate(tqdm(batches, unit="batch")):
        texts        = df.loc[batch_idx, "text"].tolist()
        translations = translate_batch(client, texts)
        df.loc[batch_idx, "text_en"] = translations

        if (batch_num + 1) % SAVE_EVERY == 0:
            df.to_csv(args.output, index=False, encoding="utf-8")
            tqdm.write(f"  Progress saved at batch {batch_num + 1}")

    # ------------------------------------------------------------------
    # Final save
    # ------------------------------------------------------------------
    df.to_csv(args.output, index=False, encoding="utf-8")
    translated = df["text_en"].notna().sum()
    print(f"\nDone. {translated}/{total} chunks translated.")
    print(f"Saved to '{args.output}'")

    total_words = df["word_count"].sum()
    est_tokens  = total_words * 1.3
    est_cost    = (est_tokens / 1_000_000) * 0.80
    print(f"Estimated cost: ~${est_cost:.2f} (rough estimate)")


if __name__ == "__main__":
    main()