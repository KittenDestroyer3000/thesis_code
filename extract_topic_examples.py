"""
extract_topic_examples.py
-------------------------
For each topic in each round's final LDA model, extracts the N most
representative chunks — those with the highest proportion for that topic.

Reading these gives a much richer understanding of topic content than
top words alone.

Output:
  topic_examples_r1.txt
  topic_examples_r2.txt
  topic_examples_r3.txt

  Each file contains the top-N chunks per topic with their source document,
  topic proportion, and full text.

Usage:
    python extract_topic_examples.py

Dependencies:
    pip install pandas
"""

import pandas as pd
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────────

BASE = Path(r"C:\Users\olesc\PycharmProjects\thesis")

MODELS = {
    "r1": {
        "chunks":     BASE / "adenda_r1" / "chunks_r1_translated.csv",
        "chunks_lda": BASE / "adenda_r1" / "chunks_topics_r1_k14v2.csv",
        "label":      "Round 1 (k=14, k14v2)",
        "k":          14,
    },
    "r2": {
        "chunks":     BASE / "adenda_r2" / "chunks_r2_translated.csv",
        "chunks_lda": BASE / "adenda_r2" / "chunks_topics_r2_k13v1.csv",
        "label":      "Round 2 (k=13, k13v1)",
        "k":          13,
    },
    "r3": {
        "chunks":     BASE / "adenda_r3" / "chunks_r3_translated.csv",
        "chunks_lda": BASE / "adenda_r3" / "chunks_topics_r3_k12v2.csv",
        "label":      "Round 3 (k=12, k12v2)",
        "k":          12,
    },
}

OUTPUT_DIR = BASE / "icsara"
TOP_N      = 3    # number of representative chunks per topic

TOPIC_LABELS = {
    "r1": {
        0: "Native forest & vegetation",
        1: "Flora monitoring & species conservation",
        2: "Drainage & hydrology",
        3: "Water infrastructure & surface works",
        4: "Processing plant & infrastructure",
        5: "Water process & chemical treatment",
        6: "Environmental impact & measures",
        7: "Hazardous waste & emergency planning",
        8: "Project phases (construction/operation/closure)",
        9: "Soil, revegetation & disposal",
        10: "Fauna baseline & sampling",
        11: "Air quality & health risk",
        12: "Fauna relocation & monitoring",
        13: "Indigenous communities (GHPPI)",
    },
    "r2": {
        0: "Native forest & vegetation",
        1: "Water process & chemical treatment",
        2: "Air quality & emissions",
        3: "Project phases & drainage",
        4: "Processing plant & infrastructure",
        5: "Indigenous communities (GHPPI)",
        6: "Fauna relocation & conservation",
        7: "Soil, emergency & drainage",
        8: "Flora monitoring & planting",
        9: "Environmental planning & measures",
        10: "Health risk assessment",
        11: "Hydrology & water characterisation",
        12: "Air quality & health risk",
    },
    "r3": {
        0: "Project phases & environmental impact",
        1: "Flora species & conservation planning",
        2: "Air quality & emissions monitoring",
        3: "Fauna relocation & baseline sampling",
        4: "Water process & chemical treatment",
        5: "Health risk assessment",
        6: "Indigenous communities (GHPPI)",
        7: "Project phases & emissions (operation/closure)",
        8: "Native forest & species conservation",
        9: "Emergency planning & hazardous waste",
        10: "Drainage & hydrology",
        11: "Soil, vegetation & revegetation",
    },
}


def extract_examples(round_key: str, config: dict) -> str:
    print(f"\n[{round_key.upper()}] {config['label']}")

    # chunks_topics CSV already contains text_en and topic columns — load directly
    df = pd.read_csv(config["chunks_lda"])
    df = df.dropna(subset=["text_en"])
    print(f"  Loaded {len(df)} chunks with {len([c for c in df.columns if c.startswith('topic_')])} topic columns")

    labels = TOPIC_LABELS[round_key]
    k      = config["k"]
    lines  = []

    lines.append("=" * 70)
    lines.append(config["label"])
    lines.append("=" * 70)

    for t in range(k):
        topic_col = f"topic_{t:02d}"
        label     = labels.get(t, f"Topic {t:02d}")

        lines.append(f"\n{'─' * 70}")
        lines.append(f"T{t:02d}  {label}")
        lines.append(f"{'─' * 70}")

        if topic_col not in df.columns:
            lines.append("  [topic column not found]")
            continue

        top = df.nlargest(TOP_N, topic_col)[
            ["doc_id", "chunk_id", topic_col, "text_en"]
        ]

        for rank, (_, row) in enumerate(top.iterrows(), 1):
            prop  = row[topic_col]
            doc   = str(row["doc_id"]).replace(".txt", "")[:60]
            text  = str(row["text_en"])

            lines.append(f"\n  [{rank}] {doc}")
            lines.append(f"      Topic proportion: {prop:.3f}")
            lines.append(f"      {text}")

    return "\n".join(lines)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for round_key, config in MODELS.items():
        report = extract_examples(round_key, config)

        out_path = OUTPUT_DIR / f"topic_examples_{round_key}.txt"
        out_path.write_text(report, encoding="utf-8")
        print(f"  Saved: {out_path.name}")

    print(f"\nDone. Files saved to: {OUTPUT_DIR}")
    print("Open the .txt files to read representative chunks per topic.")


if __name__ == "__main__":
    main()