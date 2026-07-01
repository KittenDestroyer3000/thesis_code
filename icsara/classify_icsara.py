"""
classify_icsara.py
------------------
Classifies each ICSARA item into one of 15 adenda topics using
gpt-4o-mini. Each item is classified individually so that a
reasoning field can be captured per item — useful as a methodology
audit trail in a thesis context.

Output columns added to the CSV:
  topic_code    : e.g. "T05"
  topic_label   : e.g. "Native forest & vegetation"
  reasoning     : one sentence from the model explaining the assignment
  confidence    : "high" | "medium" | "low" (model self-assessment)

Output files:
  icsara_classified.csv         — full table with classification columns
  icsara_classification_log.txt — per-item log for audit/review

Usage:
    set OPENAI_API_KEY=your_key_here
    python classify_icsara.py
    python classify_icsara.py --resume   (continue interrupted run)
"""

from openai import OpenAI
import argparse
import json
import os
import re
import time
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ── Configuration ──────────────────────────────────────────────────────────────

MODEL       = "gpt-4o-mini"
MAX_RETRIES = 4
RETRY_DELAY = 10
SAVE_EVERY  = 25

DEFAULT_INPUT  = r"C:\Users\olesc\PycharmProjects\thesis\icsara\data\icsara_items_translated.csv"
DEFAULT_OUTPUT = r"C:\Users\olesc\PycharmProjects\thesis\icsara\data\icsara_classified.csv"
DEFAULT_LOG    = r"C:\Users\olesc\PycharmProjects\thesis\icsara\data\icsara_classification_log.txt"

# ── Topic definitions ──────────────────────────────────────────────────────────
# Descriptions are deliberately richer than the short labels,
# to give the model enough context to distinguish similar topics.

# ── Topic codebooks per round ──────────────────────────────────────────────────
# Each round's ICSARA is classified into that round's adenda topic codebook.
# Round 3 to be filled in once Round 3 LDA is finalised.

TOPICS_R1 = {
    "T00": ("Native forest & vegetation",
            "Native forest (bosque nativo), Native Preservation Forests (BNP), vegetation "
            "clearance, safety buffers from BNP, woodland cover, riparian vegetation, "
            "native tree species, deforestation, canopy, forest law compliance, Ley de Bosque Nativo."),

    "T01": ("Flora monitoring & species conservation",
            "Flora surveys, botanical inventory, plant species in conservation status, "
            "endangered or protected plant species, queule, pitao, canelo, gomortega, "
            "floristic transects, species sampling, flora monitoring plans, CITES species."),

    "T02": ("Drainage & hydrology",
            "Surface drainage design, hydrological characterisation, channel design, "
            "rainwater management, stormwater, drainage network, catchment areas, "
            "precipitation analysis, flood risk, hydraulic modelling."),

    "T03": ("Water infrastructure & surface works",
            "Water supply infrastructure, surface water management, civil works near water, "
            "water extraction, water rights, pipeline routing, water storage, "
            "earthworks affecting watercourses, stream crossings, water balance."),

    "T04": ("Processing plant & infrastructure",
            "Mineral processing plant design, infrastructure layout, internal roads, pipelines, "
            "tailings management, reagents, processing circuits, camps, facilities siting, "
            "rare earth concentration process, heap leach, ion exchange, plant location."),

    "T05": ("Water process & chemical treatment",
            "Water treatment processes, chemical characterisation of process water, "
            "effluent treatment plants (PTAS), wastewater quality, chemical elements in water, "
            "process water reuse, water quality standards, NCh 1333, clays processing water."),

    "T06": ("Environmental impact & measures",
            "Environmental impact significance assessment, impact prediction methodology, "
            "mitigation measures, compensation measures, environmental management plans, "
            "impact significance criteria, residual impacts, cumulative impacts, EIA framework."),

    "T07": ("Hazardous waste & emergency planning",
            "Hazardous materials management, hazardous waste storage and disposal, "
            "emergency response plans, contingency plans, spill response, fire risk, "
            "chemical storage, DS 148, toxic substances, accident protocols, safety planning."),

    "T08": ("Project phases (construction/operation/closure)",
            "Construction phase activities, operation phase, closure and abandonment, "
            "project schedule, phasing, earthworks, commissioning, decommissioning, "
            "phase-specific impacts or measures, works programme, project timeline."),

    "T09": ("Soil, revegetation & disposal",
            "Soil quality and characterisation, topsoil management, revegetation plans, "
            "waste disposal areas, spoil management, soil profiles, edaphic characterisation, "
            "organic matter, reforestation, substrate preparation, planting schedules."),

    "T10": ("Fauna baseline & sampling",
            "Terrestrial fauna surveys, wildlife baseline characterisation, species recorded, "
            "sampling methodology, fauna inventory, biodiversity assessment, species richness, "
            "fauna monitoring, biological surveys, baseline ecological data."),

    "T11": ("Air quality & health risk",
            "Air quality baseline and monitoring, dust emissions, particulate matter (PM10, PM2.5), "
            "dispersion modelling, health risk assessment, fugitive dust, stack emissions, "
            "ambient air standards, odour, respiratory health, air quality monitoring plans."),

    "T12": ("Fauna relocation & monitoring",
            "Wildlife rescue and relocation plans, fauna translocation, reptile rescue, "
            "mammal rescue, amphibians, birds, wildlife corridors, fauna rescue protocols, "
            "herpetology surveys, capture methods, release sites, fauna monitoring plans."),

    "T13": ("Indigenous communities (GHPPI)",
            "Indigenous peoples and human groups (GHPPI), free prior and informed consent, "
            "indigenous consultation, Mapuche, Lafkenche, cultural heritage, ceremonial sites, "
            "nguillatún, anthropological characterisation, ILO Convention 169, "
            "indigenous rights, traditional practices, territory, community engagement."),
}

TOPICS_R2 = {
    "T00": ("Native forest & vegetation",
            "Native forest (bosque nativo), Native Preservation Forests (BNP), vegetation "
            "clearance, safety buffers from BNP, woodland cover, riparian vegetation, "
            "native tree species, deforestation, canopy, forest law compliance, Ley de Bosque Nativo."),

    "T01": ("Water process & chemical treatment",
            "Water treatment processes, chemical characterisation of process water, "
            "effluent treatment plants (PTAS), wastewater quality, chemical elements in water, "
            "process water reuse, water quality standards, clays processing water, "
            "hazardous waste, SPLP/TCLP tests, trace metals, radioactive materials."),

    "T02": ("Air quality & emissions",
            "Air quality baseline and monitoring, dust emissions, particulate matter (PM10, PM2.5), "
            "dispersion modelling, fugitive dust, stack emissions, ambient air standards, "
            "emission inventories, air quality monitoring plans."),

    "T03": ("Project phases & drainage",
            "Construction phase, operation phase, closure and abandonment, project schedule, "
            "drainage design, surface water management, channel design, earthworks, "
            "phase-specific impacts, works programme, stormwater management."),

    "T04": ("Processing plant & infrastructure",
            "Mineral processing plant design, infrastructure layout, internal roads, pipelines, "
            "tailings management, reagents, processing circuits, camps, facilities siting, "
            "rare earth concentration process, heap leach, ion exchange, plant location."),

    "T05": ("Indigenous communities (GHPPI)",
            "Indigenous peoples and human groups (GHPPI), free prior and informed consent, "
            "indigenous consultation, Mapuche, Lafkenche, cultural heritage, ceremonial sites, "
            "nguillatún, anthropological characterisation, ILO Convention 169, "
            "indigenous rights, traditional practices, territory, medicinal practices."),

    "T06": ("Fauna relocation & conservation",
            "Wildlife rescue and relocation plans, fauna translocation, reptile rescue, "
            "mammal rescue, amphibians, birds, wildlife corridors, fauna rescue protocols, "
            "herpetology surveys, capture methods, release sites, fauna monitoring plans, "
            "species conservation status."),

    "T07": ("Soil, emergency & drainage",
            "Soil quality and characterisation, topsoil management, emergency response plans, "
            "contingency plans, spill response, hazardous materials, drainage management, "
            "revegetation plans, waste disposal areas, soil profiles, edaphic characterisation."),

    "T08": ("Flora monitoring & planting",
            "Flora surveys, botanical inventory, plant species in conservation status, "
            "endangered or protected plant species, queule, pitao, canelo, gomortega, "
            "floristic transects, species sampling, planting plans, native species nurseries, "
            "revegetation species selection."),

    "T09": ("Environmental planning & measures",
            "Environmental management plans, mitigation measures, compensation measures, "
            "environmental monitoring, impact significance criteria, residual impacts, "
            "cumulative impacts, EIA framework, environmental commitments, RCA conditions."),

    "T10": ("Health risk assessment",
            "Human health risk assessment, exposure pathways, population receptors, "
            "toxicological assessment, risk characterisation, health risk from emissions, "
            "epidemiological data, health impact assessment."),

    "T11": ("Hydrology & water characterisation",
            "Hydrological characterisation, surface water quality, groundwater monitoring, "
            "water balance, aquifer characterisation, precipitation analysis, hydrogeological "
            "studies, water rights, DGA, water sampling campaigns, baseline hydrology."),

    "T12": ("Air quality & health risk",
            "Combined air quality and health risk assessment, particulate matter health effects, "
            "dispersion modelling linked to health outcomes, PM10 and PM2.5 health standards, "
            "population exposure to air emissions, respiratory health risk."),
}

# Round 3 topic codebook (k=12, final model k12v2)
TOPICS_R3 = {
    "T00": ("Project phases & environmental impact",
            "Construction, operation and closure phase activities, project scheduling, "
            "phase-specific environmental impacts, surface water management during works, "
            "earthworks, general environmental impact assessment across project phases."),

    "T01": ("Flora species & conservation planning",
            "Flora species in conservation status, sensitive native species, botanical surveys, "
            "conservation planning, protected plant species, queule, pitao, canelo, gomortega, "
            "native species management plans, botanical monitoring."),

    "T02": ("Air quality & emissions monitoring",
            "Air quality baseline and monitoring, dust emissions, particulate matter (PM10, PM2.5), "
            "dispersion modelling, fugitive dust, stack emissions, ambient air standards, "
            "emission inventories, air quality monitoring plans, atmospheric emissions."),

    "T03": ("Fauna relocation & baseline sampling",
            "Wildlife rescue and relocation plans, fauna translocation, fauna baseline surveys, "
            "species sampling methodology, reptile and mammal rescue, amphibians, birds, "
            "capture methods, release sites, fauna monitoring, herpetology surveys."),

    "T04": ("Water process & chemical treatment",
            "Water treatment processes, chemical characterisation of process water, "
            "effluent treatment plants (PTAS), wastewater quality, chemical elements in water, "
            "process water reuse, clays processing water, waste water management, "
            "hazardous substances in water, disposal of process waste."),

    "T05": ("Health risk assessment",
            "Human health risk assessment, exposure pathways, population receptors, "
            "toxicological assessment, risk characterisation, cumulative health risk, "
            "particulate matter health effects, epidemiological data, health impact assessment."),

    "T06": ("Indigenous communities (GHPPI)",
            "Indigenous peoples and human groups (GHPPI), free prior and informed consent, "
            "indigenous consultation, Mapuche, Lafkenche, cultural heritage, ceremonial sites, "
            "anthropological characterisation, ILO Convention 169, indigenous rights, "
            "traditional practices, territory, natural resource use by communities."),

    "T07": ("Project phases & emissions (operation/closure)",
            "Operation phase and closure phase activities, phase-specific emissions, "
            "decommissioning, abandonment, operational emissions inventory, closure planning, "
            "emission sources during operation, phase-specific mitigation measures."),

    "T08": ("Native forest & species conservation",
            "Native forest (bosque nativo), Native Preservation Forests (BNP), vegetation "
            "clearance, safety buffers from BNP, woodland cover, riparian vegetation, "
            "native tree species, deforestation, canopy, forest law compliance, "
            "species conservation status in forest context."),

    "T09": ("Emergency planning & hazardous waste",
            "Emergency response plans, contingency plans, spill response, accident protocols, "
            "hazardous materials management, hazardous waste storage and disposal, "
            "chemical storage, DS 148, toxic substances, fire risk, evacuation planning."),

    "T10": ("Drainage & hydrology",
            "Surface drainage design, hydrological characterisation, channel design, "
            "rainwater management, stormwater, drainage network, catchment areas, "
            "precipitation analysis, flood risk, hydraulic modelling, APF calculations, "
            "groundwater interaction with drainage."),

    "T11": ("Soil, vegetation & revegetation",
            "Soil quality and characterisation, topsoil management, revegetation plans, "
            "vegetation surveys, native species planting, soil profiles, edaphic characterisation, "
            "organic matter, reforestation, substrate preparation, planting schedules, "
            "intervened vegetation areas."),
}

ROUND_TOPICS = {
    "r1": TOPICS_R1,
    "r2": TOPICS_R2,
    "r3": TOPICS_R3,
}

# Active codebook — set by --round argument in main()
TOPICS = TOPICS_R1  # default, overridden at runtime



TOPIC_LIST_STR = "\n".join(
    f"  {code}: {label} — {desc}"
    for code, (label, desc) in TOPICS.items()
)

SYSTEM_PROMPT = """You are an expert in Chilean environmental impact assessment (EIA) law 
and rare earth mining projects. Your task is to classify regulatory questions from a 
Round 1 ICSARA document into exactly one thematic topic from the Round 1 adenda topic codebook.

You must respond with valid JSON only — no preamble, no explanation outside the JSON.
"""

USER_TEMPLATE = """Classify the following ICSARA regulatory item into exactly one topic.

TOPICS:
{topic_list}

ICSARA ITEM (section: {section}):
{text}

Respond with this exact JSON structure:
{{
  "topic_code": "<code, e.g. T05>",
  "topic_label": "<label matching the code>",
  "reasoning": "<one sentence explaining why this topic fits best>",
  "confidence": "<high|medium|low>"
}}

Choose the topic that best captures the PRIMARY regulatory concern of the item.
If the item is purely administrative/procedural with no specific environmental theme, use T03."""


# ── Classification ─────────────────────────────────────────────────────────────

def classify_item(client, section: str, text: str) -> dict:
    """Classify a single ICSARA item. Returns dict with topic_code, topic_label,
    reasoning, confidence. Falls back to T03 on repeated failure."""

    user_message = USER_TEMPLATE.format(
        topic_list=TOPIC_LIST_STR,
        section=section,
        text=text[:3000]  # cap at 3000 chars — haiku context is ample but be safe
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

            # Strip markdown code fences if present
            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()

            result = json.loads(raw)

            # Validate expected fields
            assert result.get("topic_code") in TOPICS, f"Unknown topic code: {result.get('topic_code')}"
            assert result.get("confidence") in ("high", "medium", "low")

            return result

        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e):
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
                print(f"\n  Classification failed after {MAX_RETRIES} attempts. Assigning T03 fallback.")
                return {
                    "topic_code": "T03",
                    "topic_label": TOPICS["T03"][0],
                    "reasoning": "Fallback — classification failed after max retries.",
                    "confidence": "low",
                }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LLM-classify ICSARA items into adenda topics.")
    parser.add_argument("--input",  default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--log",    default=DEFAULT_LOG)
    parser.add_argument("--resume", action="store_true",
                        help="Resume from a partially classified output file")
    parser.add_argument("--round", default="r1", choices=["r1", "r2", "r3"],
                        help="Which round\'s topic codebook to use (default: r1)")
    args = parser.parse_args()

    # Set active topic codebook based on round
    global TOPICS, TOPIC_LIST_STR
    TOPICS = ROUND_TOPICS.get(args.round, TOPICS_R1)
    if not TOPICS:
        raise ValueError(f"Topic codebook for --round {args.round} is empty. "
                         f"Fill in TOPICS_R3 once Round 3 LDA is finalised.")
    TOPIC_LIST_STR = "\n".join(
        f"  {code}: {label} — {desc}"
        for code, (label, desc) in TOPICS.items()
    )
    print(f"Using Round {args.round.upper()} topic codebook ({len(TOPICS)} topics)")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY not set.\n"
            "Run:  set OPENAI_API_KEY=your_key_here   (Windows)"
        )
    client = OpenAI(api_key=api_key)

    # Load data
    print(f"Loading '{args.input}'...")
    df = pd.read_csv(args.input)

    # Skip preamble row (no item_number)
    df_items = df[df["item_number"].notna()].copy().reset_index(drop=True)
    total = len(df_items)
    print(f"  {total} numbered items to classify")

    # Resume logic
    start_idx = 0
    if args.resume and os.path.exists(args.output):
        done = pd.read_csv(args.output)
        done_items = done[done["item_number"].notna()]
        already_done = done_items["topic_code"].notna().sum()
        for col in ["topic_code", "topic_label", "reasoning", "confidence"]:
            df_items[col] = None
            if col in done_items.columns:
                df_items.loc[:already_done - 1, col] = done_items.loc[:already_done - 1, col].values
        start_idx = already_done
        print(f"  Resuming from item index {start_idx} ({already_done} already classified)")
    else:
        for col in ["topic_code", "topic_label", "reasoning", "confidence"]:
            df_items[col] = None

    # Classify
    log_lines = []
    indices = list(range(start_idx, total))

    print(f"\nClassifying {len(indices)} items...\n")

    for i in tqdm(indices, unit="item"):
        row = df_items.iloc[i]
        text = str(row.get("text_en") or row.get("text_es", ""))
        section = str(row.get("section", ""))

        result = classify_item(client, section, text)

        df_items.at[i, "topic_code"]  = result["topic_code"]
        df_items.at[i, "topic_label"] = result["topic_label"]
        df_items.at[i, "reasoning"]   = result["reasoning"]
        df_items.at[i, "confidence"]  = result["confidence"]

        log_lines.append(
            f"Item {int(row['item_number']):>3} | {result['topic_code']} | "
            f"{result['confidence']:<6} | {result['reasoning']}"
        )

        if (i + 1) % SAVE_EVERY == 0:
            df_items.to_csv(args.output, index=False, encoding="utf-8")
            tqdm.write(f"  Progress saved at item {i + 1}")

    # Final save
    df_items.to_csv(args.output, index=False, encoding="utf-8")
    print(f"\nClassified CSV written: {args.output}")

    # Write log
    Path(args.log).write_text("\n".join(log_lines), encoding="utf-8")
    print(f"Audit log written     : {args.log}")

    # Summary
    counts = df_items["topic_code"].value_counts().sort_index()
    conf   = df_items["confidence"].value_counts()

    print("\nTopic distribution:")
    for code, count in counts.items():
        label = TOPICS[code][0] if code in TOPICS else "EIA methodology"
        pct = 100 * count / total
        print(f"  {code} {label:<45} {count:>3} ({pct:.1f}%)")

    print(f"\nConfidence breakdown:")
    for level in ["high", "medium", "low"]:
        n = conf.get(level, 0)
        print(f"  {level:<6} {n:>3} ({100*n/total:.1f}%)")

    est_tokens = total * 800  # ~800 tokens per item (prompt + response)
    est_cost = (est_tokens / 1_000_000) * 0.60
    print(f"\nEstimated cost: ~${est_cost:.3f}")


if __name__ == "__main__":
    main()