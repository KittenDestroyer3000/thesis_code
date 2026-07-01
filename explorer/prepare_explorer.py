"""
prepare_explorer.py
-------------------
Joins coverage and chunks data for all three rounds and produces a single
fully self-contained HTML explorer with all rounds embedded.

Round switcher buttons let the user toggle between rounds in the browser.
No server needed — double-click explorer_all_rounds.html to open.

Output: explorer_all_rounds.html in the explorer/ directory

Usage:
    python prepare_explorer.py

Dependencies:
    pip install pandas tqdm
"""

import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ── Configuration ──────────────────────────────────────────────────────────────

BASE = Path(r"C:\Users\olesc\PycharmProjects\thesis")

COVERAGE_FILES = {
    "r1": BASE / "icsara" / "data" / "coverage_round_1.csv",
    "r2": BASE / "icsara" / "data" / "coverage_round_2.csv",
    "r3": BASE / "icsara" / "data" / "coverage_round_3.csv",
}

CHUNKS_FILES = {
    "r1": BASE / "adenda_r1" / "chunks_r1_translated.csv",
    "r2": BASE / "adenda_r2" / "chunks_r2_translated.csv",
    "r3": BASE / "adenda_r3" / "chunks_r3_translated.csv",
}

OUTPUT_DIR  = BASE / "explorer"  # self-contained HTML output directory
OUTPUT_FILE = "explorer_all_rounds.html"

ROUND_LABELS = {
    "r1": "Round 1 (394 items)",
    "r2": "Round 2 (205 items)",
    "r3": "Round 3 (114 items)",
}


def build_records(round_key):
    coverage_path = COVERAGE_FILES[round_key]
    chunks_path   = CHUNKS_FILES[round_key]

    print(f"\n[{round_key.upper()}] Loading coverage: {coverage_path.name}")
    cov = pd.read_csv(coverage_path)
    cov = cov[cov["item_number"].notna()].copy()
    cov["item_number"] = cov["item_number"].astype(int)
    print(f"  {len(cov)} items")

    print(f"[{round_key.upper()}] Loading chunks: {chunks_path.name}")
    chunks_df = pd.read_csv(chunks_path,
                            usecols=["chunk_id", "doc_id", "text_en", "start_page", "end_page"])
    chunks_df = chunks_df.dropna(subset=["text_en"])
    dupes = chunks_df["chunk_id"].duplicated().sum()
    if dupes:
        chunks_df = chunks_df.drop_duplicates(subset=["chunk_id"], keep="first")
    chunk_lookup = chunks_df.set_index("chunk_id").to_dict("index")
    print(f"  {len(chunk_lookup)} chunks indexed")

    records = []
    for _, row in tqdm(cov.iterrows(), total=len(cov), unit="item",
                       desc=f"  Building {round_key}"):
        raw_ids    = str(row.get("top_chunk_ids", "") or "")
        chunk_ids  = [c.strip() for c in raw_ids.split("|") if c.strip()]
        chunks_out = []
        for cid in chunk_ids:
            chunk = chunk_lookup.get(cid)
            if chunk:
                chunks_out.append({
                    "chunk_id":   cid,
                    "doc_id":     str(chunk.get("doc_id", "")),
                    "text_en":    str(chunk.get("text_en", ""))[:1500],
                    "start_page": int(chunk["start_page"]) if pd.notna(chunk.get("start_page")) else None,
                    "end_page":   int(chunk["end_page"])   if pd.notna(chunk.get("end_page"))   else None,
                })

        records.append({
            "item_number":    int(row["item_number"]),
            "section":        str(row.get("section", "")),
            "text_en":        str(row.get("text_en", "")),
            "topic_code":     str(row.get("topic_code", "")),
            "topic_label":    str(row.get("topic_label", "")),
            "coverage_score": int(row["coverage_score"]) if pd.notna(row.get("coverage_score")) else None,
            "justification":  str(row.get("justification", "")),
            "key_passage":    str(row.get("key_passage", "")),
            "max_similarity": round(float(row["max_similarity"]), 4) if pd.notna(row.get("max_similarity")) else None,
            "chunks":         chunks_out,
        })

    return records


def build_html(all_data):
    """Build a single self-contained HTML with all rounds embedded."""

    # Serialise all rounds as JS variables
    data_js = "\n".join(
        f"const DATA_{key.upper()} = {json.dumps({'round': key, 'round_label': ROUND_LABELS[key], 'total_items': len(records), 'items': records}, ensure_ascii=False)};"
        for key, records in all_data.items()
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ICSARA Coverage Explorer — REE UNO SpA</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  :root {{
    --bg: #f8f8f6; --surface: #ffffff; --border: #e2e1db;
    --text: #2d2c29; --text-muted: #888780; --accent: #2563eb;
    --score-1-bg: #fef2f2; --score-1-text: #991b1b; --score-1-border: #fca5a5;
    --score-2-bg: #fffbeb; --score-2-text: #92400e; --score-2-border: #fcd34d;
    --score-3-bg: #f0fdf4; --score-3-text: #166534; --score-3-border: #86efac;
    --radius: 8px;
  }}
  html, body {{ height: 100%; overflow: hidden; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          font-size: 14px; background: var(--bg); color: var(--text); }}
  .app {{ display: flex; flex-direction: column; height: 100vh; overflow: hidden; }}
  header {{ background: var(--surface); border-bottom: 1px solid var(--border);
            padding: 10px 20px; flex-shrink: 0; }}
  header h1 {{ font-size: 15px; font-weight: 600; }}
  header p {{ font-size: 12px; color: var(--text-muted); margin-top: 2px; }}
  .round-switcher {{ display: flex; gap: 6px; margin-top: 8px; }}
  .round-btn {{ padding: 4px 14px; border-radius: 99px; border: 1px solid var(--border);
                background: var(--bg); font-size: 12px; cursor: pointer;
                color: var(--text-muted); transition: all 0.15s; }}
  .round-btn.active {{ background: var(--accent); border-color: var(--accent);
                       color: white; font-weight: 600; }}
  .main {{ display: flex; flex: 1; overflow: hidden; min-height: 0; }}
  .sidebar {{ width: 340px; flex-shrink: 0; background: var(--surface);
              border-right: 1px solid var(--border); display: flex;
              flex-direction: column; overflow: hidden; min-height: 0; }}
  .filters {{ padding: 12px; border-bottom: 1px solid var(--border);
              display: flex; flex-direction: column; gap: 8px; flex-shrink: 0; }}
  .filters input, .filters select {{
    width: 100%; padding: 7px 10px; border: 1px solid var(--border);
    border-radius: var(--radius); font-size: 13px; background: var(--bg);
    color: var(--text); outline: none; }}
  .filters input:focus, .filters select:focus {{ border-color: var(--accent); }}
  .filter-row {{ display: flex; gap: 6px; }}
  .filter-row select {{ flex: 1; }}
  .results-count {{ padding: 6px 12px; font-size: 11px; color: var(--text-muted);
                    border-bottom: 1px solid var(--border); flex-shrink: 0; }}
  .item-list {{ overflow-y: auto; flex: 1; min-height: 0; }}
  .item-card {{ padding: 10px 12px; border-bottom: 1px solid var(--border);
                cursor: pointer; transition: background 0.1s; }}
  .item-card:hover {{ background: var(--bg); }}
  .item-card.active {{ background: #eff6ff; border-left: 3px solid var(--accent); }}
  .item-card-header {{ display: flex; justify-content: space-between;
                       align-items: flex-start; gap: 8px; margin-bottom: 4px; }}
  .item-num {{ font-size: 11px; font-weight: 600; color: var(--text-muted); white-space: nowrap; }}
  .score-badge {{ font-size: 10px; font-weight: 600; padding: 2px 7px;
                  border-radius: 99px; white-space: nowrap; flex-shrink: 0; }}
  .score-1 {{ background: var(--score-1-bg); color: var(--score-1-text); border: 1px solid var(--score-1-border); }}
  .score-2 {{ background: var(--score-2-bg); color: var(--score-2-text); border: 1px solid var(--score-2-border); }}
  .score-3 {{ background: var(--score-3-bg); color: var(--score-3-text); border: 1px solid var(--score-3-border); }}
  .item-preview {{ font-size: 12px; color: var(--text-muted); overflow: hidden;
                   display: -webkit-box; -webkit-line-clamp: 2;
                   -webkit-box-orient: vertical; line-height: 1.4; }}
  .item-topic {{ font-size: 10px; color: var(--accent); margin-top: 3px; }}
  .detail {{ flex: 1; overflow-y: auto; padding: 20px; display: flex;
             flex-direction: column; gap: 16px; min-height: 0; align-content: flex-start; }}
  .empty-state {{ display: flex; align-items: center; justify-content: center;
                  height: 100%; color: var(--text-muted); font-size: 13px; text-align: center; }}
  .card {{ background: var(--surface); border: 1px solid var(--border);
           border-radius: var(--radius); overflow: visible; }}
  .card > .card-header {{ border-radius: var(--radius) var(--radius) 0 0; overflow: hidden; }}
  .card-header {{ padding: 10px 14px; background: var(--bg);
                  border-bottom: 1px solid var(--border); display: flex;
                  justify-content: space-between; align-items: center; gap: 10px; }}
  .card-header h3 {{ font-size: 12px; font-weight: 600; color: var(--text-muted);
                     text-transform: uppercase; letter-spacing: 0.05em; }}
  .card-body {{ padding: 14px; }}
  .meta-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 12px; }}
  .meta-item label {{ display: block; font-size: 10px; font-weight: 600;
                      color: var(--text-muted); text-transform: uppercase;
                      letter-spacing: 0.05em; margin-bottom: 2px; }}
  .meta-item span {{ font-size: 13px; color: var(--text); }}
  .concern-text {{ font-size: 13px; line-height: 1.6; color: var(--text); white-space: pre-wrap; }}
  .coverage-box {{ border-radius: var(--radius); padding: 12px 14px; }}
  .coverage-box.score-1 {{ background: var(--score-1-bg); border: 1px solid var(--score-1-border); }}
  .coverage-box.score-2 {{ background: var(--score-2-bg); border: 1px solid var(--score-2-border); }}
  .coverage-box.score-3 {{ background: var(--score-3-bg); border: 1px solid var(--score-3-border); }}
  .coverage-score-label {{ font-size: 13px; font-weight: 700; margin-bottom: 6px; }}
  .score-1 .coverage-score-label {{ color: var(--score-1-text); }}
  .score-2 .coverage-score-label {{ color: var(--score-2-text); }}
  .score-3 .coverage-score-label {{ color: var(--score-3-text); }}
  .coverage-justification {{ font-size: 12px; line-height: 1.5; color: var(--text); margin-bottom: 8px; }}
  .key-passage {{ font-size: 12px; font-style: italic; color: var(--text-muted);
                  border-left: 3px solid var(--border); padding-left: 10px; line-height: 1.5; }}
  .sim-bar-container {{ display: flex; align-items: center; gap: 8px; margin-top: 12px; }}
  .sim-bar-track {{ flex: 1; height: 6px; background: var(--border);
                    border-radius: 99px; overflow: hidden; }}
  .sim-bar-fill {{ height: 100%; background: var(--accent); border-radius: 99px; }}
  .sim-value {{ font-size: 11px; color: var(--text-muted); white-space: nowrap; }}
  .chunk-list {{ display: flex; flex-direction: column; gap: 10px; }}
  .chunk-item {{ border: 1px solid var(--border); border-radius: var(--radius); overflow: visible; }}
  .chunk-item > .chunk-header {{ border-radius: var(--radius) var(--radius) 0 0; }}
  .chunk-header {{ padding: 7px 12px; background: var(--bg); display: flex;
                   justify-content: space-between; align-items: center;
                   gap: 8px; cursor: pointer; }}
  .chunk-doc {{ font-size: 11px; font-weight: 600; color: var(--text);
                overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
  .chunk-meta {{ display: flex; gap: 8px; align-items: center; flex-shrink: 0; }}
  .chunk-toggle {{ font-size: 11px; color: var(--accent); cursor: pointer;
                   border: none; background: none; padding: 0; }}
  .chunk-text {{ padding: 10px 12px; font-size: 12px; line-height: 1.6;
                 color: var(--text); display: none; border-top: 1px solid var(--border);
                 white-space: pre-wrap; word-break: break-word; }}
  .chunk-text.open {{ display: block; }}
  .loading {{ padding: 20px; color: var(--text-muted); font-size: 13px; text-align: center; }}
  mark {{ background: #fef08a; border-radius: 2px; padding: 0 1px; }}

  /* ── Mobile responsive ── */
  @media (max-width: 768px) {{
    html, body {{ overflow: auto; height: auto; }}
    .app {{ height: auto; overflow: visible; }}
    header h1 {{ font-size: 13px; }}
    .round-btn {{ padding: 6px 10px; font-size: 11px; }}
    .main {{ flex-direction: column; overflow: visible; height: auto; }}
    .sidebar {{ width: 100%; border-right: none;
                border-bottom: 1px solid var(--border);
                max-height: 50vh; overflow: hidden; }}
    .item-list {{ max-height: 35vh; overflow-y: auto; }}
    .detail {{ overflow: visible; padding: 12px; }}
    .filters input, .filters select {{ font-size: 16px; padding: 8px 10px; }}
    .filter-row {{ flex-direction: column; }}
    .filter-row select {{ width: 100%; }}
    .round-switcher {{ flex-wrap: wrap; gap: 4px; }}
    .meta-grid {{ grid-template-columns: 1fr; }}
    .card-body {{ padding: 10px; }}
    .concern-text {{ font-size: 12px; }}
    .chunk-doc {{ font-size: 10px; }}
  }}
</style>
</head>
<body>
<div class="app">
  <header>
    <h1>ICSARA Coverage Explorer — REE UNO SpA, Biobío</h1>
    <p id="hdr-meta">Select a round to begin</p>
    <div class="round-switcher">
      <button class="round-btn active" onclick="setRound('r1')">Round 1 — 394 items</button>
      <button class="round-btn" onclick="setRound('r2')">Round 2 — 205 items</button>
      <button class="round-btn" onclick="setRound('r3')">Round 3 — 114 items</button>
    </div>
  </header>
  <div class="main">
    <div class="sidebar">
      <div class="filters">
        <input type="text" id="search" placeholder="Search regulatory concerns…" oninput="applyFilters()">
        <select id="fscore" onchange="applyFilters()">
          <option value="">All coverage scores</option>
          <option value="1">1 — Not addressed</option>
          <option value="2">2 — Partially addressed</option>
          <option value="3">3 — Fully addressed</option>
        </select>
        <div class="filter-row">
          <select id="ftopic" onchange="applyFilters()"></select>
          <select id="fsec" onchange="applyFilters()"></select>
        </div>
      </div>
      <div class="results-count" id="cnt"></div>
      <div class="item-list" id="list"></div>
    </div>
    <div class="detail" id="detail">
      <div class="empty-state">
        <div>
          <p style="font-size:15px;margin-bottom:6px;">Select a regulatory concern</p>
          <p>Choose a round above, filter on the left,<br>then click an item to view details.</p>
        </div>
      </div>
    </div>
  </div>
</div>
<script>
{data_js}

const ROUND_DATA = {{ r1: DATA_R1, r2: DATA_R2, r3: DATA_R3 }};
const SCORE_LABELS = {{1:'Not addressed',2:'Partially addressed',3:'Fully addressed'}};

let activeRound = 'r1';
let filtered    = [];
let activeItem  = null;

function setRound(round) {{
  activeRound = round;
  activeItem  = null;
  document.querySelectorAll('.round-btn').forEach((b,i) => {{
    b.classList.toggle('active', ['r1','r2','r3'][i] === round);
  }});
  const data = ROUND_DATA[round];
  document.getElementById('hdr-meta').textContent =
    data.round_label + ' · ' + data.total_items + ' regulatory items';
  populateFilters(data.items);
  applyFilters();
  document.getElementById('detail').innerHTML = `
    <div class="empty-state">
      <div>
        <p style="font-size:15px;margin-bottom:6px;">Select a regulatory concern</p>
        <p>Filter on the left, then click an item.</p>
      </div>
    </div>`;
}}

function populateFilters(items) {{
  const topics = [...new Set(items.map(i => i.topic_code).filter(Boolean))].sort();
  const sections = [...new Set(items.map(i => shortenSection(i.section)).filter(Boolean))]
    .sort((a,b) => sectionSortKey(a) - sectionSortKey(b));
  document.getElementById('ftopic').innerHTML =
    '<option value="">All topics</option>' +
    topics.map(t => {{
      const item = items.find(i => i.topic_code === t);
      return `<option value="${{t}}">${{t}} ${{item ? item.topic_label : ''}}</option>`;
    }}).join('');
  document.getElementById('fsec').innerHTML =
    '<option value="">All sections</option>' +
    sections.map(s => `<option value="${{s}}">${{s}}</option>`).join('');
  document.getElementById('search').value = '';
  document.getElementById('fscore').value = '';
}}

function applyFilters() {{
  const items = ROUND_DATA[activeRound].items;
  const query = document.getElementById('search').value.toLowerCase().trim();
  const score = document.getElementById('fscore').value;
  const topic = document.getElementById('ftopic').value;
  const sec   = document.getElementById('fsec').value;

  filtered = items.filter(item => {{
    if (score && String(item.coverage_score) !== score) return false;
    if (topic && item.topic_code !== topic) return false;
    if (sec   && shortenSection(item.section) !== sec) return false;
    if (query) {{
      const text = (item.text_en + ' ' + item.topic_label + ' ' + item.justification).toLowerCase();
      if (!text.includes(query)) return false;
    }}
    return true;
  }});

  renderList(query);
  document.getElementById('cnt').textContent =
    `${{filtered.length}} of ${{items.length}} items`;
}}

function renderList(query) {{
  const list = document.getElementById('list');
  if (!filtered.length) {{
    list.innerHTML = '<div class="loading">No items match your filters.</div>';
    return;
  }}
  list.innerHTML = filtered.map(item => {{
    const preview = highlight(item.text_en.slice(0,120) + '…', query);
    const isActive = activeItem && activeItem.item_number === item.item_number;
    return `<div class="item-card ${{isActive?'active':''}}" onclick="selectItem(${{item.item_number}})">
      <div class="item-card-header">
        <span class="item-num">Item ${{item.item_number}}</span>
        <span class="score-badge score-${{item.coverage_score}}">${{SCORE_LABELS[item.coverage_score]}}</span>
      </div>
      <div class="item-preview">${{preview}}</div>
      <div class="item-topic">${{item.topic_code}} · ${{item.topic_label}}</div>
    </div>`;
  }}).join('');
}}

function selectItem(itemNum) {{
  activeItem = ROUND_DATA[activeRound].items.find(i => i.item_number === itemNum);
  if (!activeItem) return;
  const query = document.getElementById('search').value.toLowerCase().trim();
  renderList(query);
  const item = activeItem;
  const sim  = item.max_similarity ? (item.max_similarity * 100).toFixed(1) : '—';

  const chunksHTML = item.chunks.map((c, idx) => `
    <div class="chunk-item">
      <div class="chunk-header" onclick="toggleChunk(${{idx}})">
        <span class="chunk-doc">${{c.doc_id.replace('.txt','')}} · pp.${{c.start_page}}–${{c.end_page}}</span>
        <div class="chunk-meta">
          <button class="chunk-toggle" id="toggle-${{idx}}">Show ▾</button>
        </div>
      </div>
      <div class="chunk-text" id="chunk-text-${{idx}}">${{escHtml(c.text_en)}}</div>
    </div>`).join('');

  const keyPassage = item.key_passage && item.key_passage !== 'None'
    ? `<div class="key-passage">"${{escHtml(item.key_passage)}}"</div>` : '';

  document.getElementById('detail').innerHTML = `
    <div class="card">
      <div class="card-header">
        <h3>Item ${{item.item_number}}</h3>
        <span class="score-badge score-${{item.coverage_score}}">${{SCORE_LABELS[item.coverage_score]}}</span>
      </div>
      <div class="card-body">
        <div class="meta-grid">
          <div class="meta-item"><label>Topic</label><span>${{item.topic_code}} · ${{item.topic_label}}</span></div>
          <div class="meta-item"><label>Section</label><span>${{shortenSection(item.section)}}</span></div>
        </div>
        <div class="concern-text">${{escHtml(item.text_en)}}</div>
      </div>
    </div>
    <div class="card">
      <div class="card-header"><h3>Coverage assessment</h3></div>
      <div class="card-body">
        <div class="coverage-box score-${{item.coverage_score}}">
          <div class="coverage-score-label">${{item.coverage_score}} — ${{SCORE_LABELS[item.coverage_score]}}</div>
          <div class="coverage-justification">${{escHtml(item.justification)}}</div>
          ${{keyPassage}}
        </div>
        <div class="sim-bar-container">
          <span class="sim-value" style="width:130px;font-size:11px;">Best chunk similarity</span>
          <div class="sim-bar-track"><div class="sim-bar-fill" style="width:${{sim}}%"></div></div>
          <span class="sim-value">${{sim}}%</span>
        </div>
      </div>
    </div>
    <div class="card">
      <div class="card-header"><h3>Retrieved adenda passages (top ${{item.chunks.length}})</h3></div>
      <div class="card-body"><div class="chunk-list">${{chunksHTML}}</div></div>
    </div>`;
}}

function toggleChunk(idx) {{
  const text = document.getElementById(`chunk-text-${{idx}}`);
  const btn  = document.getElementById(`toggle-${{idx}}`);
  const open = text.classList.toggle('open');
  btn.textContent = open ? 'Hide ▴' : 'Show ▾';
}}

function highlight(text, query) {{
  if (!query) return escHtml(text);
  return escHtml(text).replace(new RegExp(`(${{escRe(query)}})`, 'gi'), '<mark>$1</mark>');
}}

function escHtml(s) {{
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}}
function escRe(s) {{ return s.replace(/[.*+?^${{}}()|[\]\\\\]/g,'\\\\$&'); }}

function shortenSection(s) {{
  if (!s) return '';
  s = s.trim();
  const parts = s.split('.');
  if (parts.length >= 2) {{
    const roman = parts[0].trim();
    const rest  = parts.slice(1).join('.').trim().split(/\s+/).slice(0,5).join(' ');
    return `${{roman}}. ${{rest}}`;
  }}
  return s.slice(0,50);
}}

function sectionSortKey(s) {{
  const m = {{I:1,II:2,III:3,IV:4,V:5,VI:6,VII:7,VIII:8,IX:9,X:10,
             XI:11,XII:12,XIII:13,XIV:14,XV:15,XVI:16,XVII:17}};
  return m[s.split('.')[0].trim()] || 99;
}}

// Initialise with Round 1
setRound('r1');
</script>
</body>
</html>"""


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / OUTPUT_FILE

    all_data = {}
    for round_key in ["r1", "r2", "r3"]:
        all_data[round_key] = build_records(round_key)

    print("\nBuilding combined HTML...")
    html = build_html(all_data)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    size_mb = output_path.stat().st_size / 1_000_000
    total_items = sum(len(r) for r in all_data.values())
    print(f"\nCombined HTML written : {output_path}")
    print(f"Size                  : {size_mb:.1f} MB")
    print(f"Total items           : {total_items} across 3 rounds")
    print(f"\nDouble-click {OUTPUT_FILE} to open in any browser — no server needed.")


if __name__ == "__main__":
    main()