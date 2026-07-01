"""
coverage_icsara.py
------------------
For each ICSARA item, retrieves the top-N most semantically similar adenda
chunks using cosine similarity on pre-computed embeddings, then asks
gpt-4o-mini to score how well the adenda addressed the regulatory concern.

Output columns:
  item_number       : ICSARA item number
  topic_code        : assigned topic
  topic_label       : topic label
  max_similarity    : cosine similarity of the best-matching chunk (0-1)
  coverage_score    : 1 (not addressed) | 2 (partially) | 3 (fully addressed)
  justification     : one sentence explaining the score
  key_passage       : most relevant quoted passage from the adenda (<30 words)
  top_chunk_ids     : pipe-separated list of top-N chunk IDs retrieved

Output file: icsara/data/coverage_round_{N}.csv

Usage:
    set OPENAI_API_KEY=your_key_here
    python coverage_icsara.py --round r1
    python coverage_icsara.py --round r2
    python coverage_icsara.py --round r3
    python coverage_icsara.py --round r1 --resume

Dependencies:
    pip install openai numpy pandas scikit-learn sentence-transformers tqdm
"""

import argparse
import json
import os
import re
import time
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

# ── Configuration ──────────────────────────────────────────────────────────────

MODEL       = "gpt-4o-mini"
TOP_N       = 10      # chunks to retrieve per ICSARA item
MAX_RETRIES = 4
RETRY_DELAY = 10
SAVE_EVERY  = 25

# Paths — adjust base dirs if your structure differs
BASE = Path(r"C:\Users\olesc\PycharmProjects\thesis")

ADENDA_DIRS = {
    "r1": BASE / "adenda_r1",
    "r2": BASE / "adenda_r2",
    "r3": BASE / "adenda_r3",
}

ICSARA_DIR = BASE / "icsara" / "data"

CHUNKS_FILES = {
    "r1": "chunks_r1_translated.csv",
    "r2": "chunks_r2_translated.csv",  # update if named differently
    "r3": "chunks_r3_translated.csv",
}

CLASSIFIED_FILES = {
    "r1": "icsara_classified_round_1.csv",
    "r2": "icsara_classified_round_2.csv",
    "r3": "icsara_classified_round_3.csv",
}

OUTPUT_FILES = {
    "r1": "coverage_round_1.csv",
    "r2": "coverage_round_2.csv",
    "r3": "coverage_round_3.csv",
}

EMBEDDING_MODEL = "all-mpnet-base-v2"

# ── Prompts ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert reviewer of Chilean environmental impact assessment (EIA) 
adenda documents. Your task is to assess how well a company's adenda addressed a specific 
regulatory concern raised in an ICSARA document.

You must respond with valid JSON only — no preamble, no explanation outside the JSON.
"""

USER_TEMPLATE = """You are reviewing whether an adenda (company response) adequately addressed 
a regulatory concern from an ICSARA (regulator's consolidated list of questions).

ICSARA ITEM (regulatory concern):
{icsara_text}

MOST RELEVANT ADENDA PASSAGES (retrieved by semantic similarity):
{chunks_text}

Score how well the adenda addressed this regulatory concern:
  1 = Not addressed: the adenda contains no substantive response to this specific concern
  2 = Partially addressed: the adenda responds to some aspects but leaves significant gaps or is insufficiently specific
  3 = Fully addressed: the adenda provides a clear, substantive, specific response that directly engages with the concern

Respond with this exact JSON structure:
{{
  "coverage_score": <1, 2, or 3>,
  "justification": "<one sentence explaining the score>",
  "key_passage": "<most relevant quote from the adenda passages, under 30 words, or 'None' if not addressed>"
}}"""


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_embeddings(round_key: str) -> tuple[np.ndarray, pd.DataFrame]:
    """Load pre-computed embeddings and chunk ID index."""
    emb_dir   = ADENDA_DIRS[round_key] / "embeddings"
    emb_path  = emb_dir / f"chunks_{round_key}_embeddings.npy"
    ids_path  = emb_dir / f"chunks_{round_key}_chunk_ids.csv"

    if not emb_path.exists():
        raise FileNotFoundError(
            f"Embeddings not found: {emb_path}\n"
            f"Run: python embed_adenda.py --input chunks_{round_key}_translated.csv --round {round_key}"
        )

    embeddings = np.load(emb_path)
    chunk_ids  = pd.read_csv(ids_path)
    print(f"  Embeddings loaded: {embeddings.shape}")
    return embeddings, chunk_ids


def embed_query(text: str, model) -> np.ndarray:
    """Embed a single query text."""
    return model.encode([text], convert_to_numpy=True).astype(np.float32)


def retrieve_top_chunks(query_emb: np.ndarray,
                        corpus_emb: np.ndarray,
                        chunk_ids: pd.DataFrame,
                        chunks_df: pd.DataFrame,
                        top_n: int = TOP_N) -> tuple[float, list[dict]]:
    """
    Compute cosine similarity, return top-N chunks with their text.
    Returns (max_similarity, list of {chunk_id, doc_id, text, similarity})
    """
    sims    = cosine_similarity(query_emb, corpus_emb)[0]
    top_idx = sims.argsort()[-top_n:][::-1]

    max_sim = float(sims[top_idx[0]])
    results = []
    for idx in top_idx:
        row = chunk_ids.iloc[idx]
        # Get text from chunks_df
        match = chunks_df[chunks_df["chunk_id"] == row["chunk_id"]]
        text  = match["text_en"].values[0] if len(match) else ""
        results.append({
            "chunk_id":   row["chunk_id"],
            "doc_id":     row["doc_id"],
            "text":       text,
            "similarity": float(sims[idx]),
        })
    return max_sim, results


def score_coverage(client, icsara_text: str, chunks: list[dict]) -> dict:
    """Send ICSARA item + top chunks to LLM, return coverage assessment."""
    chunks_text = "\n\n---\n\n".join(
        f"[Source: {c['doc_id']} | Similarity: {c['similarity']:.3f}]\n{c['text'][:500]}"
        for c in chunks
    )

    user_message = USER_TEMPLATE.format(
        icsara_text=icsara_text[:2000],
        chunks_text=chunks_text,
    )

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                max_tokens=300,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_message},
                ],
            )

            raw = response.choices[0].message.content.strip()
            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw,
                         flags=re.MULTILINE).strip()
            result = json.loads(raw)

            assert result.get("coverage_score") in (1, 2, 3), \
                f"Invalid score: {result.get('coverage_score')}"

            return result

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
                print(f"\n  Scoring failed after {MAX_RETRIES} attempts. Defaulting to score 1.")
                return {
                    "coverage_score": 1,
                    "justification":  "Fallback — scoring failed after max retries.",
                    "key_passage":    "None",
                }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Score ICSARA item coverage against adenda chunks."
    )
    parser.add_argument("--round", required=True, choices=["r1", "r2", "r3"],
                        help="Round to process (r1, r2, r3)")
    parser.add_argument("--top_n", type=int, default=TOP_N,
                        help=f"Chunks to retrieve per item (default: {TOP_N})")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from partial output file")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set.")
    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    round_key   = args.round
    output_path = ICSARA_DIR / OUTPUT_FILES[round_key]

    # Load ICSARA classified items
    classified_path = ICSARA_DIR / CLASSIFIED_FILES[round_key]
    print(f"Loading ICSARA items: {classified_path.name}")
    icsara_df = pd.read_csv(classified_path)
    icsara_df = icsara_df[icsara_df["item_number"].notna()].copy().reset_index(drop=True)
    total = len(icsara_df)
    print(f"  {total} items")

    # Load adenda chunks
    chunks_path = ADENDA_DIRS[round_key] / CHUNKS_FILES[round_key]
    print(f"Loading adenda chunks: {chunks_path.name}")
    chunks_df = pd.read_csv(chunks_path)
    chunks_df = chunks_df.dropna(subset=["text_en"]).reset_index(drop=True)
    print(f"  {len(chunks_df)} chunks")

    # Load embeddings
    print("Loading embeddings...")
    corpus_emb, chunk_ids = load_embeddings(round_key)

    # Load embedding model
    print(f"Loading embedding model '{EMBEDDING_MODEL}'...")
    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer(EMBEDDING_MODEL)

    # Resume logic
    start_idx = 0
    for col in ["max_similarity", "coverage_score", "justification",
                "key_passage", "top_chunk_ids"]:
        if col not in icsara_df.columns:
            icsara_df[col] = None

    if args.resume and output_path.exists():
        done = pd.read_csv(output_path)
        already = done["coverage_score"].notna().sum()
        for col in ["max_similarity", "coverage_score", "justification",
                    "key_passage", "top_chunk_ids"]:
            if col in done.columns:
                icsara_df.loc[:already-1, col] = done.loc[:already-1, col].values
        start_idx = already
        print(f"  Resuming from item {start_idx} ({already} already scored)")

    # Process items
    print(f"\nScoring {total - start_idx} items (top-{args.top_n} chunks each)...\n")

    for i in tqdm(range(start_idx, total), unit="item"):
        row        = icsara_df.iloc[i]
        icsara_text = str(row.get("text_en") or row.get("text_es", ""))

        # Embed query
        query_emb = embed_query(icsara_text, embed_model)

        # Retrieve top chunks
        max_sim, top_chunks = retrieve_top_chunks(
            query_emb, corpus_emb, chunk_ids, chunks_df, args.top_n
        )

        # Score coverage
        result = score_coverage(client, icsara_text, top_chunks)

        icsara_df.at[i, "max_similarity"]  = round(max_sim, 4)
        icsara_df.at[i, "coverage_score"]  = result["coverage_score"]
        icsara_df.at[i, "justification"]   = result["justification"]
        icsara_df.at[i, "key_passage"]     = result["key_passage"]
        icsara_df.at[i, "top_chunk_ids"]   = "|".join(
            c["chunk_id"] for c in top_chunks
        )

        if (i + 1) % SAVE_EVERY == 0:
            icsara_df.to_csv(output_path, index=False, encoding="utf-8")
            tqdm.write(f"  Progress saved at item {i + 1}")

    # Final save
    icsara_df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"\nCoverage CSV written: {output_path}")

    # Summary
    scored = icsara_df[icsara_df["coverage_score"].notna()]
    counts = scored["coverage_score"].value_counts().sort_index()
    print("\nCoverage distribution:")
    labels = {1: "Not addressed", 2: "Partially addressed", 3: "Fully addressed"}
    for score, count in counts.items():
        pct = 100 * count / len(scored)
        print(f"  {int(score)} — {labels[int(score)]:<22} {count:>4} ({pct:.1f}%)")

    avg_sim = icsara_df["max_similarity"].astype(float).mean()
    print(f"\nAvg max similarity  : {avg_sim:.3f}")

    # Cost estimate
    est_tokens = total * (2000 + args.top_n * 600)
    est_cost   = (est_tokens / 1_000_000) * 0.60
    print(f"Estimated cost      : ~${est_cost:.2f}")


if __name__ == "__main__":
    main()