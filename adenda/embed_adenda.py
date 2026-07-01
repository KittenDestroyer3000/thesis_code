"""
embed_adenda.py
---------------
Embeds all adenda chunks using sentence-transformers and saves to disk.

Output (in embeddings/ subfolder of the adenda directory):
  chunks_r{N}_embeddings.npy   — float32 matrix (n_chunks x embedding_dim)
  chunks_r{N}_chunk_ids.csv    — chunk IDs and doc_ids matching row order

The embedding matrix and chunk ID CSV are paired — row i of the matrix
corresponds to row i of the CSV.

Usage:
    python embed_adenda.py --input chunks_r1_translated.csv --round r1
    python embed_adenda.py --input chunks_r2_translated.csv --round r2
    python embed_adenda.py --input chunks_r3_translated.csv --round r3

    # Custom output directory:
    python embed_adenda.py --input chunks_r1_translated.csv --round r1 ^
        --output_dir C:\\path\\to\\adenda_r1\\embeddings

Dependencies:
    pip install sentence-transformers numpy pandas tqdm
"""

import argparse
import os
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

MODEL_NAME = "all-mpnet-base-v2"
BATCH_SIZE = 64   # chunks per encoding batch — tune down if memory issues


def embed_chunks(texts: list[str], model) -> np.ndarray:
    """Encode texts in batches, return float32 matrix."""
    all_embeddings = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE),
                  desc="Embedding batches", unit="batch"):
        batch = texts[i: i + BATCH_SIZE]
        emb   = model.encode(batch, show_progress_bar=False,
                             convert_to_numpy=True)
        all_embeddings.append(emb)
    return np.vstack(all_embeddings).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Embed adenda chunks with sentence-transformers."
    )
    parser.add_argument("--input", required=True,
                        help="Path to chunks_r{N}_translated.csv")
    parser.add_argument("--round", required=True, choices=["r1", "r2", "r3"],
                        help="Round identifier (r1, r2, r3)")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory (default: embeddings/ next to input file)")
    parser.add_argument("--model", default=MODEL_NAME,
                        help=f"Sentence-transformers model (default: {MODEL_NAME})")
    parser.add_argument("--text_col", default="text_en",
                        help="Column to embed (default: text_en)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Output directory
    out_dir = Path(args.output_dir) if args.output_dir else input_path.parent / "embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)

    emb_path      = out_dir / f"chunks_{args.round}_embeddings.npy"
    chunk_id_path = out_dir / f"chunks_{args.round}_chunk_ids.csv"

    # Load chunks
    print(f"Loading '{input_path.name}'...")
    df = pd.read_csv(input_path)
    df = df.dropna(subset=[args.text_col]).reset_index(drop=True)
    total = len(df)
    print(f"  {total} chunks to embed")

    # Load model
    print(f"\nLoading model '{args.model}'...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model)
    print(f"  Embedding dimension: {model.get_sentence_embedding_dimension()}")

    # Embed
    print(f"\nEmbedding {total} chunks (CPU — this may take 10-20 minutes)...")
    texts      = df[args.text_col].tolist()
    embeddings = embed_chunks(texts, model)

    print(f"  Done. Matrix shape: {embeddings.shape}")

    # Save embeddings
    np.save(emb_path, embeddings)
    print(f"\nEmbeddings saved : {emb_path}")

    # Save chunk ID index
    id_df = df[["chunk_id", "doc_id"]].copy()
    id_df["row_index"] = id_df.index
    id_df.to_csv(chunk_id_path, index=False, encoding="utf-8")
    print(f"Chunk ID index   : {chunk_id_path}")
    print(f"\nDone. {total} chunks embedded.")
    print(f"Next step: python coverage_icsara.py --round {args.round}")


if __name__ == "__main__":
    main()