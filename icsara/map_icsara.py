"""
map_icsara.py
-------------
Assigns each ICSARA item to one of the 15 adenda topics using keyword matching.

For each item in icsara_items_translated.csv, the script counts keyword hits
per topic and assigns the topic with the highest count. Ties are broken by
keyword frequency. Items with zero matches on all topics are assigned T03
(Environmental impact assessment) as a fallback — since T03 keywords appear
in nearly every item, it is excluded from normal matching and reserved as
the catch-all for otherwise unclassifiable items.

Output:
  icsara_mapped.csv       — full item table with assigned topic + scores
  icsara_mapping_report.txt — diagnostic summary for sanity checking

Usage:
    python map_icsara.py
    python map_icsara.py --input icsara_items_translated.csv --output icsara_mapped.csv
"""

import re
import argparse
import pandas as pd
from pathlib import Path
from collections import defaultdict

# ── Configuration ──────────────────────────────────────────────────────────────

DEFAULT_INPUT  = r"C:\Users\olesc\PycharmProjects\thesis\icsara\data\icsara_items_translated.csv"
DEFAULT_OUTPUT = r"C:\Users\olesc\PycharmProjects\thesis\icsara\data\icsara_mapped.csv"
REPORT_OUTPUT  = r"C:\Users\olesc\PycharmProjects\thesis\icsara\data\icsara_mapping_report.txt"

# ── Topic keyword vocabulary ───────────────────────────────────────────────────
# T03 is excluded from normal matching and used as fallback only.
# Each entry is (topic_label, [keywords]).
# Keywords are matched case-insensitively as whole words or substrings.

TOPICS = {
    "T00": (
        "Water infrastructure & drainage",
        ["water", "drainage", "effluent", "runoff", "hydrology", "aquifer",
         "watercourse", "river", "basin", "PTAS", "wastewater", "irrigation",
         "rainfall", "groundwater", "flow", "stream", "discharge", "pond",
         "infiltration", "leachate", "wetland"]
    ),
    "T01": (
        "Noise & acoustic impacts",
        ["noise", "acoustic", "sound", "vibration", "decibel", "dB",
         "receptor", "blast", "detonation", "attenuation"]
    ),
    "T02": (
        "Geotechnical characterisation",
        ["geotechnical", "geology", "slope", "stability", "erosion",
         "sediment", "stratigraphy", "borehole", "permeability", "lithology",
         "subsidence", "landslide", "fault", "seismic", "bearing capacity",
         "subsoil", "bedrock"]
    ),
    # T03 excluded from normal matching — used as fallback only
    "T04": (
        "Processing plant & infrastructure",
        ["plant", "processing", "infrastructure", "facility", "road",
         "pipeline", "tailings", "reagent", "circuit", "crusher", "mill",
         "conveyor", "pond", "heap", "leach", "concentrate", "refinery",
         "access route", "internal road", "camp"]
    ),
    "T05": (
        "Native forest & vegetation",
        ["native forest", "BNP", "vegetation", "forest", "woodland",
         "clearing", "buffer", "canopy", "native species", "bosque nativo",
         "preservation forest", "native tree", "deforestation", "tree cover",
         "understory", "riparian", "30 m", "safety buffer"]
    ),
    "T06": (
        "Flora monitoring & species conservation",
        ["flora", "conservation status", "monitoring", "botanical",
         "inventory", "endangered", "protected species", "queule", "pitao",
         "canelo", "specimen", "threatened", "rare species", "plant survey",
         "floristic", "transect"]
    ),
    "T07": (
        "Indigenous communities (GHPPI)",
        ["indigenous", "GHPPI", "community", "consultation", "FPIC",
         "cultural", "territory", "Mapuche", "Lafkenche", "ancestral",
         "traditional", "nguillatún", "rewe", "ceremonial", "anthropological",
         "ethnic", "native people", "indigenous rights", "ILO 169"]
    ),
    "T08": (
        "Emergency & contingency planning",
        ["emergency", "contingency", "spill", "accident", "fire",
         "response plan", "evacuation", "incident", "hazard response",
         "emergency protocol", "safety plan", "first responder"]
    ),
    "T09": (
        "Fauna, soil & revegetation",
        ["fauna", "soil", "revegetation", "habitat", "rescue",
         "transplant", "restoration", "topsoil", "organic matter",
         "reforestation", "planting", "seedling", "nursery", "substrate",
         "soil quality", "edaphic", "soil profile"]
    ),
    "T10": (
        "Project phases (construction/operation/closure)",
        ["construction", "operation", "closure", "phase", "schedule",
         "timeline", "works", "commissioning", "decommissioning",
         "abandonment", "dismantling", "earthworks", "excavation",
         "preparatory works"]
    ),
    "T11": (
        "Hazardous waste & regulatory compliance",
        ["hazardous", "waste", "disposal", "storage", "compliance",
         "hazardous material", "chemical", "toxic", "residue",
         "waste management", "landfill", "containment", "spill prevention",
         "DS 148", "regulated waste"]
    ),
    "T12": (
        "Environmental law & permits",
        ["PAS", "permit", "decree", "authorisation", "sectoral",
         "RCA", "environmental permit", "legal requirement", "regulation",
         "DS 40", "law 19300", "SEA", "CONAF", "DGA", "SAG", "SMA",
         "sectoral permit", "water rights", "concession"]
    ),
    "T13": (
        "Air quality, emissions & health risk",
        ["air quality", "emission", "dust", "particulate", "PM10",
         "PM2.5", "dispersion", "health risk", "odour", "air pollution",
         "fugitive dust", "stack", "ventilation", "respiratory",
         "contamination", "ambient air", "modelling"]
    ),
    "T14": (
        "Fauna relocation (reptiles & mammals)",
        ["relocation", "rescue", "capture", "mammal", "reptile",
         "translocation", "corridor", "wildlife", "fauna rescue",
         "animal rescue", "herpetology", "amphibian", "bird",
         "fauna management", "species rescue"]
    ),
}

FALLBACK_TOPIC    = "T03"
FALLBACK_LABEL    = "Environmental impact assessment"


# ── Keyword matching ───────────────────────────────────────────────────────────

def count_keyword_hits(text: str, keywords: list[str]) -> int:
    """Count how many keyword occurrences appear in text (case-insensitive)."""
    text_lower = text.lower()
    count = 0
    for kw in keywords:
        # Use word-boundary-aware matching for single words,
        # plain substring for multi-word phrases
        if " " in kw:
            count += text_lower.count(kw.lower())
        else:
            count += len(re.findall(rf"\b{re.escape(kw.lower())}\b", text_lower))
    return count


def assign_topic(text: str) -> tuple[str, str, dict]:
    """
    Assign the best-matching topic to a text.
    Returns (topic_code, topic_label, scores_dict).
    Falls back to T03 if no keywords match.
    """
    scores = {}
    for code, (label, keywords) in TOPICS.items():
        scores[code] = count_keyword_hits(text, keywords)

    best_code  = max(scores, key=lambda c: scores[c])
    best_score = scores[best_code]

    if best_score == 0:
        return FALLBACK_TOPIC, FALLBACK_LABEL, scores

    return best_code, TOPICS[best_code][0], scores


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Keyword-map ICSARA items to adenda topics.")
    parser.add_argument("--input",  default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--report", default=REPORT_OUTPUT)
    args = parser.parse_args()

    print(f"Loading '{args.input}'...")
    df = pd.read_csv(args.input)
    total = len(df)
    print(f"  {total} items loaded")

    # Use text_en for matching; fall back to text_es if translation missing
    df["_match_text"] = df["text_en"].fillna(df["text_es"])

    # Apply topic assignment
    results = df["_match_text"].apply(assign_topic)
    df["topic_code"]  = results.apply(lambda x: x[0])
    df["topic_label"] = results.apply(lambda x: x[1])
    df["match_score"] = results.apply(lambda x: max(x[2].values()))

    # Flag fallbacks and low-confidence assignments
    df["is_fallback"]    = df["match_score"] == 0
    df["low_confidence"] = df["match_score"] <= 2

    # Drop internal column
    df = df.drop(columns=["_match_text"])

    # Save
    df.to_csv(args.output, index=False, encoding="utf-8")
    print(f"\nMapped CSV written: {args.output}")

    # ── Diagnostic report ──────────────────────────────────────────────────────

    numbered = df[df["item_number"].notna()].copy()

    topic_counts    = numbered["topic_code"].value_counts().sort_index()
    fallback_count  = numbered["is_fallback"].sum()
    lowconf_count   = numbered["low_confidence"].sum()

    # Build all topic codes including fallback for complete display
    all_codes = sorted(list(TOPICS.keys()) + [FALLBACK_TOPIC])

    lines = []
    lines.append("=" * 70)
    lines.append("ICSARA TOPIC MAPPING REPORT — keyword matching")
    lines.append("=" * 70)
    lines.append(f"\nTotal items mapped : {len(numbered)}")
    lines.append(f"Fallback (T03)     : {fallback_count} items (zero keyword matches)")
    lines.append(f"Low confidence     : {lowconf_count} items (score ≤ 2)\n")

    lines.append("-" * 70)
    lines.append(f"{'Code':<6} {'Label':<45} {'Count':>5} {'%':>6}")
    lines.append("-" * 70)

    for code in all_codes:
        if code == FALLBACK_TOPIC:
            label = FALLBACK_LABEL + " [fallback]"
        else:
            label = TOPICS[code][0]
        count = topic_counts.get(code, 0)
        pct   = 100 * count / len(numbered) if len(numbered) else 0
        lines.append(f"{code:<6} {label:<45} {count:>5} {pct:>5.1f}%")

    lines.append("-" * 70)

    # Per-section breakdown
    lines.append("\n\nPER-SECTION BREAKDOWN")
    lines.append("-" * 70)
    for section, grp in numbered.groupby("section"):
        lines.append(f"\n{section[:70]}")
        sc = grp["topic_code"].value_counts()
        for code, cnt in sc.items():
            label = TOPICS[code][0] if code in TOPICS else FALLBACK_LABEL
            lines.append(f"  {code} {label:<42} {cnt:>3}")

    # Low confidence items for manual review
    low = numbered[numbered["low_confidence"]].copy()
    if len(low):
        lines.append(f"\n\nLOW CONFIDENCE ITEMS (score ≤ 2) — review these manually")
        lines.append("-" * 70)
        for _, row in low.iterrows():
            lines.append(f"\n  Item {int(row['item_number']):>3} | {row['topic_code']} ({row['match_score']} hits) | "
                         f"{row['section'][:40]}")
            preview = str(row.get("text_en", row.get("text_es", "")))[:120].replace("\n", " ")
            lines.append(f"  {preview}...")

    report_text = "\n".join(lines)
    print("\n" + report_text)

    Path(args.report).write_text(report_text, encoding="utf-8")
    print(f"\nReport written: {args.report}")


if __name__ == "__main__":
    main()