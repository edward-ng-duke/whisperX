"""Speaker-attributed transcription using local /diarize + remote Qwen3-ASR.

Steps:
  1. POST audio to local WhisperX /diarize -> raw pyannote segments.
  2. Merge consecutive same-speaker rows (small inter-segment gaps absorbed).
  3. ffmpeg-slice each merged turn into a 16k mono wav in a temp dir.
  4. POST each slice to Qwen3-ASR (OpenAI-compatible /v1/audio/transcriptions).
  5. Emit one JSON file per input with the assembled turns.

Usage:
  python deploy/scripts/qwen3_diarize.py [audio.mp3 ...] [--out-dir DIR] [...]
Defaults to data/*.mp3 -> data/results/<basename>.qwen3_diarize.json.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import requests

DIARIZE_URL_DEFAULT = "http://localhost:8000/diarize"
ASR_URL_DEFAULT = "http://10.0.0.32:6001/v1/audio/transcriptions"
ASR_MODEL = "models/Qwen3-ASR-1.7B"
ASR_MIN_SEC = 0.5
ASR_MAX_SEC = 300.0


def diarize(audio_path: Path, url: str) -> list[dict]:
    with audio_path.open("rb") as fh:
        resp = requests.post(url, files={"file": (audio_path.name, fh, "audio/mpeg")}, timeout=600)
    resp.raise_for_status()
    return resp.json()["segments"]


def merge_turns(segments: list[dict], gap_tol: float = 1.0) -> list[dict]:
    """Collapse consecutive same-speaker segments separated by < gap_tol seconds."""
    if not segments:
        return []
    segs = sorted(segments, key=lambda s: s["start"])
    merged = [dict(segs[0])]
    for s in segs[1:]:
        last = merged[-1]
        if s["speaker"] == last["speaker"] and s["start"] - last["end"] <= gap_tol:
            last["end"] = max(last["end"], s["end"])
        else:
            merged.append(dict(s))
    return merged


def slice_wav(audio_path: Path, start: float, end: float, out_wav: Path) -> None:
    """Cut [start, end) from audio_path, downmix to 16k mono, write to out_wav."""
    duration = max(0.0, end - start)
    cmd = [
        "ffmpeg", "-nostdin", "-y", "-loglevel", "error",
        "-ss", f"{start:.3f}", "-t", f"{duration:.3f}",
        "-i", str(audio_path),
        "-ar", "16000", "-ac", "1", "-f", "wav",
        str(out_wav),
    ]
    subprocess.run(cmd, check=True)


_ASR_RE = re.compile(r"language\s+(\S+)", re.IGNORECASE)


def parse_asr_text(raw: str) -> tuple[str, str]:
    """Split 'language <name><asr_text>real text' into (lang_code_or_empty, text)."""
    if "<asr_text>" not in raw:
        return "", raw.strip()
    meta, text = raw.split("<asr_text>", 1)
    m = _ASR_RE.match(meta.strip())
    lang = ""
    if m:
        name = m.group(1)
        if name.lower() not in ("none", ""):
            lang = name
    return lang, text.strip()


def asr_call(wav_path: Path, url: str, language: str | None, prompt: str | None) -> tuple[str, str]:
    files = {"file": (wav_path.name, wav_path.open("rb"), "audio/wav")}
    data = {"model": ASR_MODEL}
    if language:
        data["language"] = language
    if prompt:
        data["prompt"] = prompt
    try:
        resp = requests.post(url, files=files, data=data, headers={"Authorization": "Bearer EMPTY"}, timeout=600)
    finally:
        files["file"][1].close()
    resp.raise_for_status()
    raw = resp.json().get("text", "")
    return parse_asr_text(raw)


def process(audio: Path, *, diarize_url: str, asr_url: str, language: str | None, prompt: str | None,
            out_dir: Path) -> Path:
    print(f"[{audio.name}] diarize ...", flush=True)
    raw_segments = diarize(audio, diarize_url)
    turns = merge_turns(raw_segments)
    print(f"[{audio.name}] {len(raw_segments)} raw segs -> {len(turns)} turns", flush=True)

    out_turns: list[dict] = []
    skipped = 0
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for i, t in enumerate(turns):
            dur = t["end"] - t["start"]
            if dur < ASR_MIN_SEC or dur > ASR_MAX_SEC:
                skipped += 1
                out_turns.append({**t, "language": "", "text": "",
                                  "skipped_reason": f"duration {dur:.2f}s out of [{ASR_MIN_SEC}, {ASR_MAX_SEC}]"})
                continue
            wav = td / f"slice_{i:03d}.wav"
            slice_wav(audio, t["start"], t["end"], wav)
            try:
                lang, text = asr_call(wav, asr_url, language=language, prompt=prompt)
            except requests.HTTPError as e:
                print(f"  turn {i} ASR error: {e.response.status_code} {e.response.text[:200]}", flush=True)
                out_turns.append({**t, "language": "", "text": "", "asr_error": str(e)})
                continue
            out_turns.append({**t, "language": lang, "text": text})
            print(f"  turn {i:>3} {t['speaker']} [{t['start']:.2f}-{t['end']:.2f}] {dur:.1f}s lang={lang} :: {text[:60]}",
                  flush=True)

    speakers = sorted({t["speaker"] for t in out_turns})
    payload = {
        "audio": audio.name,
        "asr_model": ASR_MODEL,
        "asr_url": asr_url,
        "diarize_url": diarize_url,
        "language_hint": language,
        "prompt": prompt,
        "speakers": speakers,
        "n_turns": len(out_turns),
        "n_skipped": skipped,
        "turns": out_turns,
    }
    out_path = out_dir / f"{audio.stem}.qwen3_diarize.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{audio.name}] -> {out_path} ({len(out_turns)} turns, skipped {skipped})\n", flush=True)
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser()
    repo_root = Path(__file__).resolve().parents[2]
    default_inputs = sorted((repo_root / "data").glob("*.mp3"))
    ap.add_argument("inputs", nargs="*", type=Path, default=default_inputs,
                    help="Audio files (default: data/*.mp3 under repo root).")
    ap.add_argument("--diarize-url", default=DIARIZE_URL_DEFAULT)
    ap.add_argument("--asr-url", default=ASR_URL_DEFAULT)
    ap.add_argument("--language", default="zh", help="ISO 639-1 lang hint for ASR; '' to disable.")
    ap.add_argument("--prompt", default=None, help="Optional ASR hot-words.")
    ap.add_argument("--out-dir", type=Path, default=repo_root / "data" / "results")
    args = ap.parse_args()

    if not args.inputs:
        print("no input audio files found", file=sys.stderr)
        return 1

    lang = args.language or None
    for audio in args.inputs:
        process(Path(audio), diarize_url=args.diarize_url, asr_url=args.asr_url,
                language=lang, prompt=args.prompt, out_dir=args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
