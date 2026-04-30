"""Build a single self-contained HTML report.

Reads data/*.mp3 + data/results/*.qwen3_diarize.json, emits
data/results/report.html with embedded audio (base64) and JSON data.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA_DIR = REPO / "data"
RESULTS_DIR = DATA_DIR / "results"

FILES = [
    {
        "key": "上海话",
        "label": "上海话.mp3",
        "kicker": "卷 一",
        "blurb": "沪语演讲片段。一位主讲人引用经济宏观叙事，偶有助理短促插话。",
        "subtitle": "Shanghainese monologue · single dominant speaker",
    },
    {
        "key": "武定路",
        "label": "武定路.mp3",
        "kicker": "卷 二",
        "blurb": "双人产品对谈。讲解方介绍会议音频转写功能与多网部署细节，询问方频繁追问。",
        "subtitle": "Two-speaker product walkthrough · dense turn-taking",
    },
]


def to_js_json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False).replace("</", "<\\/")


def build() -> Path:
    payload = []
    for f in FILES:
        audio_path = DATA_DIR / f["label"]
        json_path = RESULTS_DIR / f"{f['key']}.qwen3_diarize.json"
        with json_path.open(encoding="utf-8") as fp:
            obj = json.load(fp)
        audio_b64 = base64.b64encode(audio_path.read_bytes()).decode("ascii")
        payload.append({
            "key": f["key"],
            "label": f["label"],
            "kicker": f["kicker"],
            "blurb": f["blurb"],
            "subtitle": f["subtitle"],
            "audio_b64": audio_b64,
            "data": obj,
        })

    html = TEMPLATE.replace("__DATA__", to_js_json(payload))
    out = RESULTS_DIR / "report.html"
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out}  ({len(html) / 1024:.1f} KB)")
    return out


TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>双段中文音频卷宗 · Speaker × ASR</title>
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
<link href="https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400;0,500;0,600;1,400;1,600&family=Noto+Serif+SC:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet" />
<style>
  :root {
    --paper: #f4ecdb;
    --paper-2: #f9f3e6;
    --surface: #fffaee;
    --ink: #221911;
    --ink-2: #4a3a28;
    --ink-3: #7a6849;
    --rule: #d6c6a6;
    --rule-soft: #e7dcc1;
    --gold: #a87632;
    --gold-soft: #d6b07a;
    --sp0: #b3543a;          /* terracotta */
    --sp0-soft: #efd6c8;
    --sp1: #2c5957;          /* deep pine teal */
    --sp1-soft: #c9d9d6;
    --warn: #8a4a1f;
    --serif-display: "EB Garamond", "Noto Serif SC", "Songti SC", "STSong", serif;
    --serif-body: "EB Garamond", "Noto Serif SC", "Songti SC", "STSong", serif;
    --mono: "IBM Plex Mono", ui-monospace, "JetBrains Mono", Menlo, monospace;
  }

  /* ─── reset + base ─── */
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; }
  body {
    background: var(--paper);
    color: var(--ink);
    font-family: var(--serif-body);
    font-size: 17px;
    line-height: 1.55;
    -webkit-font-smoothing: antialiased;
    text-rendering: optimizeLegibility;
    background-image:
      radial-gradient(1200px 600px at 20% -10%, rgba(168, 118, 50, 0.08), transparent 70%),
      radial-gradient(800px 500px at 110% 30%, rgba(44, 89, 87, 0.06), transparent 65%);
  }

  .grain::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 100;
    opacity: 0.06;
    mix-blend-mode: multiply;
    background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='200' height='200'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='2'/><feColorMatrix values='0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0.5 0'/></filter><rect width='200' height='200' filter='url(%23n)'/></svg>");
  }

  .wrap {
    max-width: 1080px;
    margin: 0 auto;
    padding: 64px 48px 96px;
  }
  @media (max-width: 720px) {
    .wrap { padding: 36px 22px 64px; }
    body { font-size: 16px; }
  }

  /* ─── header ─── */
  header.cover {
    border-top: 2px solid var(--ink);
    border-bottom: 1px solid var(--rule);
    padding: 28px 0 36px;
    position: relative;
  }
  .cover::before {
    content: "";
    position: absolute;
    top: -2px; left: 0; right: 0;
    height: 7px;
    border-top: 1px solid var(--ink);
    pointer-events: none;
  }
  .kicker {
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: 0.32em;
    text-transform: uppercase;
    color: var(--ink-3);
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    border-bottom: 1px dashed var(--rule);
    padding-bottom: 10px;
    margin-bottom: 24px;
  }
  .kicker .ornament { color: var(--gold); letter-spacing: 0; }
  h1.title {
    font-family: var(--serif-display);
    font-weight: 600;
    font-size: clamp(46px, 7vw, 84px);
    line-height: 0.98;
    margin: 8px 0 0;
    letter-spacing: -0.01em;
  }
  h1.title .cn {
    display: block;
    font-weight: 600;
    letter-spacing: 0.18em;
  }
  h1.title em {
    font-style: italic;
    font-weight: 400;
    color: var(--gold);
  }
  .lede {
    font-size: 18px;
    color: var(--ink-2);
    max-width: 64ch;
    margin: 22px 0 0;
    font-style: italic;
  }
  .meta-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0;
    margin-top: 32px;
    border-top: 1px solid var(--rule);
    border-bottom: 1px solid var(--rule);
  }
  @media (max-width: 720px) { .meta-row { grid-template-columns: repeat(2, 1fr); } }
  .meta-cell {
    padding: 14px 16px;
    border-right: 1px solid var(--rule-soft);
    font-family: var(--mono);
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--ink-3);
  }
  .meta-cell:last-child { border-right: none; }
  .meta-cell strong {
    display: block;
    font-family: var(--serif-display);
    font-weight: 500;
    font-size: 22px;
    color: var(--ink);
    text-transform: none;
    letter-spacing: 0;
    margin-top: 4px;
  }

  /* ─── methodology ─── */
  section.methodology {
    margin-top: 56px;
  }
  .section-label {
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: 0.32em;
    text-transform: uppercase;
    color: var(--ink-3);
    display: flex;
    align-items: center;
    gap: 16px;
  }
  .section-label::after {
    content: "";
    flex: 1;
    height: 1px;
    background: var(--rule);
  }
  .section-label .num { color: var(--gold); }
  h2.h-section {
    font-family: var(--serif-display);
    font-weight: 600;
    font-size: 36px;
    line-height: 1.05;
    margin: 14px 0 28px;
    letter-spacing: -0.005em;
  }
  .method-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0;
    border-top: 1px solid var(--rule);
    border-bottom: 1px solid var(--rule);
  }
  @media (max-width: 720px) { .method-grid { grid-template-columns: repeat(2, 1fr); } }
  .method-step {
    padding: 22px 18px 24px;
    border-right: 1px solid var(--rule-soft);
    background: var(--paper-2);
  }
  .method-step:last-child { border-right: none; }
  .method-step .step-no {
    font-family: var(--mono);
    font-size: 11px;
    color: var(--gold);
    letter-spacing: 0.32em;
  }
  .method-step h4 {
    font-family: var(--serif-display);
    font-weight: 600;
    font-size: 18px;
    margin: 6px 0 10px;
    letter-spacing: 0;
  }
  .method-step p {
    font-size: 14px;
    line-height: 1.5;
    color: var(--ink-2);
    margin: 0;
  }
  .method-step code {
    font-family: var(--mono);
    font-size: 12px;
    color: var(--ink);
    background: var(--surface);
    padding: 1px 6px;
    border: 1px solid var(--rule-soft);
    border-radius: 2px;
  }

  /* ─── highlight comparison ─── */
  section.compare {
    margin-top: 64px;
    padding: 28px 32px 32px;
    background: linear-gradient(180deg, var(--surface), #fff8e9);
    border: 1px solid var(--rule);
    border-left: 4px solid var(--gold);
    position: relative;
  }
  section.compare::before {
    content: "✦";
    position: absolute;
    top: -14px; left: 24px;
    background: var(--paper);
    padding: 0 10px;
    color: var(--gold);
    font-size: 18px;
  }
  .compare-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 18px;
    margin-top: 16px;
  }
  @media (max-width: 720px) { .compare-grid { grid-template-columns: 1fr; } }
  .compare-card {
    border: 1px solid var(--rule);
    background: var(--surface);
    padding: 16px 18px;
  }
  .compare-card h5 {
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: 0.24em;
    color: var(--ink-3);
    text-transform: uppercase;
    margin: 0 0 6px;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .compare-card.qwen3 h5 { color: var(--sp1); }
  .compare-card.whisper h5 { color: var(--warn); }
  .compare-card .verdict {
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: 0.16em;
    padding: 2px 8px;
    border: 1px solid currentColor;
    border-radius: 999px;
  }
  .compare-card p {
    font-size: 16px;
    margin: 8px 0 0;
    color: var(--ink);
  }

  /* ─── audio section ─── */
  article.audio-section {
    margin-top: 96px;
    border-top: 2px solid var(--ink);
    position: relative;
  }
  article.audio-section::before {
    content: "";
    position: absolute;
    top: -2px; left: 0; right: 0;
    height: 6px;
    border-top: 1px solid var(--ink);
    pointer-events: none;
  }
  .audio-header {
    padding: 28px 0 24px;
    display: grid;
    grid-template-columns: 1.4fr 1fr;
    gap: 32px;
    align-items: end;
  }
  @media (max-width: 720px) { .audio-header { grid-template-columns: 1fr; gap: 16px; } }
  .audio-title-block .audio-kicker {
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: 0.4em;
    text-transform: uppercase;
    color: var(--gold);
  }
  .audio-title-block h2 {
    font-family: var(--serif-display);
    font-weight: 600;
    font-size: clamp(40px, 6vw, 64px);
    margin: 6px 0 6px;
    line-height: 1;
    letter-spacing: 0.04em;
  }
  .audio-subtitle {
    font-style: italic;
    color: var(--ink-3);
    font-size: 15px;
  }
  .audio-blurb {
    color: var(--ink-2);
    font-size: 16px;
    line-height: 1.6;
    margin: 0;
  }

  .audio-player-wrap {
    background: var(--surface);
    border: 1px solid var(--rule);
    padding: 14px 16px;
    margin: 4px 0 28px;
    display: flex;
    align-items: center;
    gap: 16px;
    position: sticky;
    top: 8px;
    z-index: 10;
    box-shadow: 0 4px 14px rgba(34, 25, 17, 0.06);
  }
  .audio-player-wrap .now {
    font-family: var(--mono);
    font-size: 11px;
    color: var(--ink-3);
    letter-spacing: 0.16em;
    text-transform: uppercase;
    white-space: nowrap;
  }
  .audio-player-wrap audio {
    flex: 1;
    min-width: 0;
    height: 36px;
  }

  /* ─── stats ─── */
  .stats-row {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 0;
    border-top: 1px solid var(--rule);
    border-bottom: 1px solid var(--rule);
    margin: 8px 0 32px;
  }
  @media (max-width: 720px) { .stats-row { grid-template-columns: repeat(2, 1fr); } }
  .stat {
    padding: 14px 14px 16px;
    border-right: 1px solid var(--rule-soft);
  }
  .stat:last-child { border-right: none; }
  .stat .lbl {
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: 0.24em;
    text-transform: uppercase;
    color: var(--ink-3);
  }
  .stat .val {
    font-family: var(--serif-display);
    font-weight: 600;
    font-size: 26px;
    color: var(--ink);
    line-height: 1.1;
    margin-top: 2px;
  }
  .stat .val .unit {
    font-family: var(--mono);
    font-size: 12px;
    font-weight: 400;
    color: var(--ink-3);
    margin-left: 4px;
  }

  /* ─── timeline ─── */
  .timeline-block { margin-bottom: 36px; }
  .timeline-svg {
    width: 100%;
    height: auto;
    display: block;
    background: var(--paper-2);
    border: 1px solid var(--rule);
  }
  .timeline-axis {
    display: flex;
    justify-content: space-between;
    font-family: var(--mono);
    font-size: 10px;
    color: var(--ink-3);
    margin-top: 4px;
    letter-spacing: 0.06em;
  }

  /* ─── speaker breakdown ─── */
  .breakdown { margin-bottom: 36px; }
  .breakdown-rows {
    display: flex;
    flex-direction: column;
    gap: 10px;
    margin-top: 14px;
  }
  .br-row {
    display: grid;
    grid-template-columns: 130px 1fr 110px;
    gap: 14px;
    align-items: center;
    font-family: var(--mono);
    font-size: 12px;
  }
  @media (max-width: 720px) { .br-row { grid-template-columns: 110px 1fr 90px; } }
  .br-row .name { color: var(--ink); letter-spacing: 0.08em; }
  .br-row .name .swatch {
    display: inline-block;
    width: 10px; height: 10px;
    margin-right: 8px;
    vertical-align: -1px;
    border-radius: 1px;
  }
  .br-bar {
    height: 14px;
    background: var(--rule-soft);
    position: relative;
    overflow: hidden;
  }
  .br-bar > div { height: 100%; }
  .br-row .num { color: var(--ink-2); text-align: right; }

  /* ─── notes / observations ─── */
  .notes {
    margin: 0 0 40px;
    padding: 22px 26px 24px;
    background: var(--paper-2);
    border-left: 3px solid var(--gold);
    border-top: 1px solid var(--rule-soft);
    border-bottom: 1px solid var(--rule-soft);
  }
  .notes h4 {
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: 0.32em;
    text-transform: uppercase;
    color: var(--ink-3);
    margin: 0 0 12px;
  }
  .notes ul {
    margin: 0;
    padding-left: 20px;
    font-size: 16px;
    line-height: 1.7;
    color: var(--ink-2);
  }
  .notes li { margin-bottom: 6px; }
  .notes em {
    background: rgba(168, 118, 50, 0.18);
    font-style: normal;
    padding: 0 3px;
  }
  .notes code {
    font-family: var(--mono);
    background: var(--surface);
    padding: 1px 6px;
    border: 1px solid var(--rule-soft);
    font-size: 13px;
  }

  /* ─── turns ─── */
  .turns-block .section-label { margin-bottom: 18px; }
  .turn {
    display: grid;
    grid-template-columns: 4px 1fr;
    gap: 16px;
    padding: 18px 0;
    border-bottom: 1px solid var(--rule-soft);
    transition: background 120ms ease;
  }
  .turn:last-child { border-bottom: none; }
  .turn:hover { background: rgba(255, 250, 238, 0.55); }
  .turn .band { background: var(--sp0); }
  .turn[data-speaker="SPEAKER_01"] .band { background: var(--sp1); }
  .turn[data-speaker="SPEAKER_00"] .speaker-tag { color: var(--sp0); }
  .turn[data-speaker="SPEAKER_01"] .speaker-tag { color: var(--sp1); }
  .turn-meta {
    display: flex;
    align-items: center;
    gap: 14px;
    flex-wrap: wrap;
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: 0.06em;
    color: var(--ink-3);
  }
  .turn-meta .idx {
    width: 32px;
    color: var(--ink-3);
    font-size: 11px;
    letter-spacing: 0.04em;
  }
  .speaker-tag {
    font-weight: 500;
    letter-spacing: 0.18em;
    text-transform: uppercase;
  }
  .turn-time {
    background: transparent;
    border: 1px solid var(--rule);
    color: var(--ink);
    padding: 2px 8px;
    cursor: pointer;
    font: inherit;
    transition: all 120ms;
  }
  .turn-time:hover {
    background: var(--ink);
    color: var(--paper);
    border-color: var(--ink);
  }
  .turn-dur { color: var(--ink-3); }
  .lang-tag {
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: 0.16em;
    padding: 1px 7px;
    border: 1px solid var(--rule);
    color: var(--ink-2);
    background: var(--surface);
  }
  .lang-tag.cantonese { color: var(--gold); border-color: var(--gold-soft); }
  .play-btn {
    margin-left: auto;
    background: transparent;
    border: 1px solid var(--ink);
    color: var(--ink);
    padding: 3px 12px;
    font: inherit;
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: 0.12em;
    cursor: pointer;
    transition: all 120ms;
  }
  .play-btn:hover { background: var(--ink); color: var(--paper); }
  .play-btn.playing { background: var(--gold); color: var(--paper); border-color: var(--gold); }
  .turn-text {
    font-family: var(--serif-body);
    font-size: 19px;
    line-height: 1.65;
    color: var(--ink);
    margin: 8px 0 0;
    max-width: 64ch;
  }
  .turn-text .empty {
    color: var(--ink-3);
    font-style: italic;
    font-size: 15px;
  }

  /* ─── skipped ─── */
  details.skipped {
    margin-top: 36px;
    border: 1px dashed var(--rule);
    padding: 14px 18px;
    background: var(--paper-2);
  }
  details.skipped summary {
    cursor: pointer;
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: 0.28em;
    text-transform: uppercase;
    color: var(--ink-3);
  }
  details.skipped summary::marker { color: var(--gold); }
  details.skipped table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 14px;
    font-family: var(--mono);
    font-size: 12px;
  }
  details.skipped td {
    padding: 4px 10px 4px 0;
    color: var(--ink-2);
  }
  details.skipped td.sp { font-weight: 500; }

  /* ─── footer ─── */
  footer.colophon {
    margin-top: 96px;
    padding-top: 18px;
    border-top: 1px solid var(--rule);
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--ink-3);
    display: flex;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 8px;
  }
  footer .gold { color: var(--gold); }
</style>
</head>
<body class="grain">
<div class="wrap">

  <!-- ════════ COVER ════════ -->
  <header class="cover">
    <div class="kicker">
      <span>Transcript Dossier · 2026.04.30</span>
      <span class="ornament">✦  ✧  ✦</span>
      <span>volume 01</span>
    </div>
    <h1 class="title">
      <span class="cn">双 段 中 文 卷 宗</span>
      <em>speaker × asr · two clips, turn by turn</em>
    </h1>
    <p class="lede">两段普通话/沪语短录音, 经本地 pyannote diarization 切分话轮, 再交由远端 Qwen3-ASR-1.7B 逐段转写。本文逐 turn 呈现, 时间码可点击跳播, 每段右侧 ▶ 按钮仅播该段。</p>
    <div class="meta-row">
      <div class="meta-cell">pipeline<strong>pyannote × Qwen3</strong></div>
      <div class="meta-cell">diarize<strong>community-1</strong></div>
      <div class="meta-cell">asr<strong>Qwen3-ASR 1.7B</strong></div>
      <div class="meta-cell">device<strong>cuda · vLLM</strong></div>
    </div>
  </header>

  <!-- ════════ METHODOLOGY ════════ -->
  <section class="methodology">
    <div class="section-label"><span class="num">01</span><span>METHODOLOGY · 流水线</span></div>
    <h2 class="h-section">从音频到带说话人的文本, 共四步</h2>
    <div class="method-grid">
      <div class="method-step">
        <div class="step-no">STEP 01</div>
        <h4>说话人切分</h4>
        <p>把整段音频送本地 <code>POST /diarize</code>, pyannote-community-1 模型返回 <code>{start, end, speaker}</code> 时间线。</p>
      </div>
      <div class="method-step">
        <div class="step-no">STEP 02</div>
        <h4>话轮合并</h4>
        <p>同一 speaker 相邻段, 间隙 ≤ 1s 自动并入, 让一个 turn 对应一句完整发言, 减少碎片。</p>
      </div>
      <div class="method-step">
        <div class="step-no">STEP 03</div>
        <h4>切片转码</h4>
        <p>对每个 turn 用 ffmpeg 切出 16k 单声道 wav, 写入临时目录。短于 0.5s 的段会被标记跳过。</p>
      </div>
      <div class="method-step">
        <div class="step-no">STEP 04</div>
        <h4>远端 ASR</h4>
        <p>切片 multipart POST 到 <code>10.0.0.32:6001</code> 上 vLLM 跑的 Qwen3-ASR-1.7B, 解析返回的 <code>language X&lt;asr_text&gt;…</code>。</p>
      </div>
    </div>
  </section>

  <!-- ════════ COMPARISON ════════ -->
  <section class="compare">
    <div class="section-label"><span class="num">02</span><span>HEADLINE FINDING · 关键对比</span></div>
    <h2 class="h-section" style="margin-bottom:6px">沪语段上, Qwen3 大幅压制 Whisper</h2>
    <p style="color:var(--ink-2);margin:0;font-size:15px">同一段 <code style="font-family:var(--mono);font-size:13px;background:var(--surface);padding:1px 6px;border:1px solid var(--rule-soft)">上海话.mp3</code> 开头 0–3 秒, 两条管线给出的转写:</p>
    <div class="compare-grid">
      <div class="compare-card whisper">
        <h5><span>Whisper large-v3</span><span class="verdict">幻觉</span></h5>
        <p>正常一点聊的你看我现在都开始录了就我中间讲点话也不要紧的过去十三年已经发生<br/><span style="color:var(--warn)">非非非非非非非非非非非非非非非非非非非非非非非非非非非非非非非非非非非非非非</span>…</p>
      </div>
      <div class="compare-card qwen3">
        <h5><span>Qwen3-ASR-1.7B</span><span class="verdict">通顺</span></h5>
        <p>至少也得了了, 你看, 我现在都开始录了, 就我中间讲点话也不要紧 ⋯⋯ 过去四十年, 已经发生翻天覆地的变化了。</p>
      </div>
    </div>
    <p style="margin-top:18px;color:var(--ink-3);font-size:13px;font-style:italic">注: Whisper 在 OOD 方言上 token 分布坍塌, 进入重复幻觉; Qwen3 训练里覆盖了沪语词条, 后文还能识别出 <em style="background:rgba(168,118,50,0.18);font-style:normal;padding:0 3px">侬</em>、<em style="background:rgba(168,118,50,0.18);font-style:normal;padding:0 3px">啥物事</em>、<em style="background:rgba(168,118,50,0.18);font-style:normal;padding:0 3px">辣个种辰光</em>。</p>
  </section>

  <!-- ════════ AUDIO SECTIONS (rendered by JS) ════════ -->
  <div id="sections"></div>

  <footer class="colophon">
    <span>SPEAKER × ASR · 2026 <span class="gold">·</span> private build</span>
    <span>pyannote-community-1 ⊗ qwen3-asr-1.7b</span>
  </footer>

</div>

<script id="payload" type="application/json">__DATA__</script>
<script>
(() => {
  const PAYLOAD = JSON.parse(document.getElementById("payload").textContent);
  const SP_COLORS = { SPEAKER_00: "#b3543a", SPEAKER_01: "#2c5957" };
  const SP_LABELS = { SPEAKER_00: "S · 一", SPEAKER_01: "S · 二" };

  // ── observations curated per file (analysis text) ──
  const NOTES = {
    "上海话": [
      "<strong>独白格局:</strong> SPEAKER_00 占 7/8 turns, 几乎独占发言, SPEAKER_01 仅在 02:16 插话一次 (0.59s), 形如旁人接话。",
      "<strong>沪语词汇被吃下了:</strong> 长 turn 里识别到 <em>侬</em>、<em>嘅思维</em>、<em>讲啥物事</em>、<em>辣辣发生</em> 等沪语口语成分, 这是 Whisper 在同段崩成 <em>非非非非</em> 的部分。",
      "<strong>语种识别有偏差:</strong> Qwen3 把 turn 5 / turn 7 标为 <em>Cantonese</em>, 实际是上海话, 推测沪粤特征字符在模型里距离较近, 不影响转写质量。",
      "<strong>叙事内容:</strong> 主讲在引用 \"过去四十年中国/美国/日本/东南亚都不是过去的样子\" 这类宏观叙事, 末段切到 \"差不多了呀\" 像是录制结束语。",
    ],
    "武定路": [
      "<strong>双角色清晰:</strong> SPEAKER_01 是<em>讲解方</em>(产品经理/技术), 几乎全程在解说音频转写一体机的功能; SPEAKER_00 是<em>询问方</em>(客户/数据局对接人), 主要追问部署细节。",
      "<strong>话轮密集:</strong> 64 个有效 turn 分布在 ~204s, 平均 ~3.2s 一次切换, 多处出现 \"边讲边接\" 的重叠 (SPEAKER_00 / SPEAKER_01 时间窗交叠)。",
      "<strong>15 个超短段被跳过:</strong> 全部是 <0.5s 的 \"嗯/啊/对\" 类反馈词, 不影响主线; 见文末 <em>Skipped Segments</em> 表。",
      "<strong>主题脉络:</strong> 一体机会议音频功能 → 多语种(普通话/英语/粤语/沪语)训练 → 热词机制(可加千个) → 多网部署(公安网/感知网) → license 售卖模式 → GPU 硬件 (4090)。",
      "<strong>专业术语命中:</strong> Qwen3 准确切出 <code>GPU</code>/<code>license</code>/<code>4090</code>/<code>demo</code>/<code>GPT</code> 等中英混入。",
    ],
  };

  // ── helpers ──
  const fmt = (sec) => {
    const m = Math.floor(sec / 60).toString().padStart(2, "0");
    const s = (sec % 60).toFixed(2).padStart(5, "0");
    return `${m}:${s}`;
  };
  const fmtShort = (sec) => `${sec.toFixed(2)}s`;
  const el = (tag, attrs = {}, ...kids) => {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "class") e.className = v;
      else if (k === "html") e.innerHTML = v;
      else if (k === "data") {
        for (const [dk, dv] of Object.entries(v)) e.dataset[dk] = dv;
      } else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
      else e.setAttribute(k, v);
    }
    for (const kid of kids) {
      if (kid == null) continue;
      e.appendChild(typeof kid === "string" ? document.createTextNode(kid) : kid);
    }
    return e;
  };

  // ── per-file rendering ──
  function renderSection(rec) {
    const turns = rec.data.turns;
    const real = turns.filter((t) => !t.skipped_reason);
    const skipped = turns.filter((t) => t.skipped_reason);
    const lastEnd = Math.max(...turns.map((t) => t.end));
    const speakerTimes = {};
    real.forEach((t) => {
      speakerTimes[t.speaker] = (speakerTimes[t.speaker] || 0) + (t.end - t.start);
    });
    const speakerCounts = {};
    real.forEach((t) => {
      speakerCounts[t.speaker] = (speakerCounts[t.speaker] || 0) + 1;
    });
    const totalSpoken = Object.values(speakerTimes).reduce((a, b) => a + b, 0);
    const speakers = Array.from(new Set(real.map((t) => t.speaker))).sort();
    const langs = Array.from(new Set(real.map((t) => t.language).filter(Boolean))).sort();

    // ── article skeleton ──
    const art = el("article", { class: "audio-section", id: `sec-${rec.key}` });

    art.appendChild(el("div", { class: "audio-header" },
      el("div", { class: "audio-title-block" },
        el("div", { class: "audio-kicker" }, rec.kicker + " · " + rec.label),
        el("h2", {}, rec.key),
        el("div", { class: "audio-subtitle" }, rec.subtitle),
      ),
      el("p", { class: "audio-blurb" }, rec.blurb),
    ));

    // ── audio player (sticky) ──
    const audio = el("audio", { controls: "", preload: "metadata" });
    const src = el("source", { src: `data:audio/mpeg;base64,${rec.audio_b64}`, type: "audio/mpeg" });
    audio.appendChild(src);
    const nowLabel = el("span", { class: "now" }, "─ ─ : ─ ─");
    audio.addEventListener("timeupdate", () => { nowLabel.textContent = fmt(audio.currentTime); });
    audio.addEventListener("seeked",      () => { nowLabel.textContent = fmt(audio.currentTime); });
    art.appendChild(el("div", { class: "audio-player-wrap" }, nowLabel, audio));

    // ── stats ──
    const stats = el("div", { class: "stats-row" });
    const addStat = (lbl, val, unit = "") => {
      stats.appendChild(el("div", { class: "stat" },
        el("div", { class: "lbl" }, lbl),
        el("div", { class: "val", html: `${val}${unit ? `<span class='unit'>${unit}</span>` : ""}` }),
      ));
    };
    addStat("DURATION", fmt(lastEnd));
    addStat("TURNS", real.length, "valid");
    addStat("SKIPPED", skipped.length, "&lt;0.5s");
    addStat("SPEAKERS", speakers.length);
    addStat("LANGUAGES", langs.length, langs.length === 1 ? "" : "tags");
    art.appendChild(stats);

    // ── timeline svg ──
    const tl = renderTimeline(turns, lastEnd, audio);
    const tlBlock = el("div", { class: "timeline-block" },
      el("div", { class: "section-label" },
        el("span", { class: "num" }, "·"),
        el("span", {}, "TIMELINE · 时 间 线"),
      ),
      el("h2", { class: "h-section", style: "font-size:24px;margin:8px 0 14px" }, "话轮分布"),
      tl,
      el("div", { class: "timeline-axis" },
        el("span", {}, "00:00"),
        el("span", {}, fmt(lastEnd / 2)),
        el("span", {}, fmt(lastEnd)),
      ),
    );
    art.appendChild(tlBlock);

    // ── speaker breakdown ──
    const brBlock = el("div", { class: "breakdown" },
      el("div", { class: "section-label" },
        el("span", { class: "num" }, "·"),
        el("span", {}, "SPEAKER SHARE · 占 比"),
      ),
      el("h2", { class: "h-section", style: "font-size:24px;margin:8px 0 14px" }, "谁说得更多"),
    );
    const rows = el("div", { class: "breakdown-rows" });
    speakers.forEach((sp) => {
      const dur = speakerTimes[sp] || 0;
      const pct = totalSpoken ? (dur / totalSpoken) * 100 : 0;
      const cnt = speakerCounts[sp] || 0;
      const bar = el("div", { class: "br-bar" },
        el("div", { style: `width:${pct.toFixed(2)}%;background:${SP_COLORS[sp] || "#888"}` }),
      );
      const nameSpan = el("span", { class: "name", html:
        `<span class='swatch' style='background:${SP_COLORS[sp]}'></span>${sp}` });
      rows.appendChild(el("div", { class: "br-row" },
        nameSpan,
        bar,
        el("span", { class: "num" }, `${dur.toFixed(2)}s · ${pct.toFixed(1)}% · ${cnt} turns`),
      ));
    });
    brBlock.appendChild(rows);
    art.appendChild(brBlock);

    // ── notes ──
    const noteList = el("ul");
    (NOTES[rec.key] || []).forEach((html) => noteList.appendChild(el("li", { html })));
    art.appendChild(el("div", { class: "notes" },
      el("h4", {}, "OBSERVATIONS · 解读"),
      noteList,
    ));

    // ── turns ──
    const turnsBlock = el("div", { class: "turns-block" });
    turnsBlock.appendChild(el("div", { class: "section-label" },
      el("span", { class: "num" }, "·"),
      el("span", {}, "TURNS · 逐 段 转 写"),
    ));
    turnsBlock.appendChild(el("h2", { class: "h-section", style: "font-size:24px;margin:8px 0 18px" },
      `共 ${real.length} 段 · 跳过 ${skipped.length} 段`));
    let visIdx = 0;
    turns.forEach((t) => {
      if (t.skipped_reason) return;
      visIdx++;
      const dur = t.end - t.start;
      const langClass = "lang-tag" + (t.language && t.language.toLowerCase() === "cantonese" ? " cantonese" : "");
      const timeBtn = el("button", { class: "turn-time", type: "button" },
        `${fmt(t.start)} → ${fmt(t.end)}`);
      timeBtn.addEventListener("click", () => {
        audio.currentTime = t.start;
        audio.play().catch(() => {});
      });
      const playBtn = el("button", { class: "play-btn", type: "button" }, "▶  segment");
      let stopAt = null;
      const onTime = () => {
        if (stopAt != null && audio.currentTime >= stopAt) {
          audio.pause();
          playBtn.classList.remove("playing");
          playBtn.textContent = "▶  segment";
          stopAt = null;
          audio.removeEventListener("timeupdate", onTime);
        }
      };
      playBtn.addEventListener("click", () => {
        // stop any other segment playback
        document.querySelectorAll(".play-btn.playing").forEach((b) => {
          b.classList.remove("playing");
          b.textContent = "▶  segment";
        });
        audio.currentTime = t.start;
        stopAt = t.end;
        audio.removeEventListener("timeupdate", onTime);
        audio.addEventListener("timeupdate", onTime);
        playBtn.classList.add("playing");
        playBtn.textContent = "■  playing";
        audio.play().catch(() => {});
      });

      const meta = el("div", { class: "turn-meta" },
        el("span", { class: "idx" }, String(visIdx).padStart(2, "0")),
        el("span", { class: "speaker-tag" }, t.speaker),
        timeBtn,
        el("span", { class: "turn-dur" }, fmtShort(dur)),
        el("span", { class: langClass }, t.language || "—"),
        playBtn,
      );

      const text = el("p", { class: "turn-text" },
        t.text || el("span", { class: "empty" }, "(空, 未识别)"));

      art.dataset.appended = "1";
      const turnEl = el("div", { class: "turn", data: { speaker: t.speaker } },
        el("div", { class: "band" }),
        el("div", {}, meta, text),
      );
      turnsBlock.appendChild(turnEl);
    });
    art.appendChild(turnsBlock);

    // ── skipped table ──
    if (skipped.length) {
      const det = el("details", { class: "skipped" });
      det.appendChild(el("summary", {}, `Skipped Segments · ${skipped.length} 个 < 0.5s 的反馈词被丢弃`));
      const tbl = el("table");
      skipped.forEach((s) => {
        tbl.appendChild(el("tr", {},
          el("td", { class: "sp" }, s.speaker),
          el("td", {}, `${fmt(s.start)} → ${fmt(s.end)}`),
          el("td", {}, `${(s.end - s.start).toFixed(2)}s`),
          el("td", {}, s.skipped_reason || ""),
        ));
      });
      det.appendChild(tbl);
      art.appendChild(det);
    }

    return art;
  }

  function renderTimeline(turns, totalDur, audio) {
    const W = 1080, H = 110, PAD_X = 8, ROW_H = 36, ROW_GAP = 10, AXIS = 20;
    const rowsTop = AXIS;
    const speakers = ["SPEAKER_00", "SPEAKER_01"];
    const ns = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(ns, "svg");
    svg.setAttribute("class", "timeline-svg");
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svg.setAttribute("preserveAspectRatio", "none");
    // axis ticks
    const ticks = 8;
    for (let i = 0; i <= ticks; i++) {
      const x = PAD_X + (W - 2 * PAD_X) * (i / ticks);
      const ln = document.createElementNS(ns, "line");
      ln.setAttribute("x1", x); ln.setAttribute("x2", x);
      ln.setAttribute("y1", AXIS - 4); ln.setAttribute("y2", H - 6);
      ln.setAttribute("stroke", "#d6c6a6");
      ln.setAttribute("stroke-dasharray", "2 4");
      svg.appendChild(ln);
    }
    // labels for rows
    speakers.forEach((sp, i) => {
      const y = rowsTop + i * (ROW_H + ROW_GAP);
      const lbl = document.createElementNS(ns, "text");
      lbl.setAttribute("x", PAD_X + 4);
      lbl.setAttribute("y", y - 4);
      lbl.setAttribute("font-family", "IBM Plex Mono, monospace");
      lbl.setAttribute("font-size", "10");
      lbl.setAttribute("fill", "#7a6849");
      lbl.setAttribute("letter-spacing", "0.1em");
      lbl.textContent = sp;
      svg.appendChild(lbl);
      // baseline
      const base = document.createElementNS(ns, "rect");
      base.setAttribute("x", PAD_X);
      base.setAttribute("y", y);
      base.setAttribute("width", W - 2 * PAD_X);
      base.setAttribute("height", ROW_H);
      base.setAttribute("fill", "#fffaee");
      base.setAttribute("stroke", "#e7dcc1");
      svg.appendChild(base);
    });
    // segments
    turns.forEach((t) => {
      const i = speakers.indexOf(t.speaker);
      if (i < 0) return;
      const y = rowsTop + i * (ROW_H + ROW_GAP);
      const x = PAD_X + ((W - 2 * PAD_X) * (t.start / totalDur));
      const w = Math.max(2, ((W - 2 * PAD_X) * ((t.end - t.start) / totalDur)));
      const rect = document.createElementNS(ns, "rect");
      rect.setAttribute("x", x);
      rect.setAttribute("y", y + 3);
      rect.setAttribute("width", w);
      rect.setAttribute("height", ROW_H - 6);
      rect.setAttribute("fill", t.speaker === "SPEAKER_00" ? "#b3543a" : "#2c5957");
      rect.setAttribute("opacity", t.skipped_reason ? "0.25" : "0.92");
      rect.setAttribute("rx", "1");
      rect.style.cursor = "pointer";
      const title = document.createElementNS(ns, "title");
      title.textContent = `${t.speaker}  ${fmt(t.start)} → ${fmt(t.end)}  ${(t.end-t.start).toFixed(2)}s\n${t.text || "(skipped)"}`;
      rect.appendChild(title);
      rect.addEventListener("click", () => { audio.currentTime = t.start; audio.play().catch(() => {}); });
      svg.appendChild(rect);
    });
    return svg;
  }

  // ── mount ──
  const host = document.getElementById("sections");
  PAYLOAD.forEach((rec) => host.appendChild(renderSection(rec)));
})();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    build()
