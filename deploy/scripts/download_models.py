"""Pre-download every model whisperX needs at runtime, into ./models/.

After this script finishes successfully you can run the API server with the
network disabled and everything will still work.

Layout produced:
    models/whisper/   faster-whisper (CTranslate2) snapshots
    models/hf/        HuggingFace cache: wav2vec2 align models + pyannote diarization
    models/torch/     torch.hub cache: torchaudio align models (en/fr/de/es/it) + (optional) silero VAD
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_ROOT = REPO_ROOT / "models"
WHISPER_DIR = MODELS_ROOT / "whisper"
HF_DIR = MODELS_ROOT / "hf"
TORCH_DIR = MODELS_ROOT / "torch"
NLTK_DIR = MODELS_ROOT / "nltk_data"

# Set cache dirs BEFORE importing torch / huggingface_hub / transformers so the
# libraries pick them up. We do NOT set HF_HUB_OFFLINE here because we *want*
# to hit the network during download.
os.environ.setdefault("HF_HOME", str(HF_DIR))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_DIR / "hub"))
os.environ.setdefault("TORCH_HOME", str(TORCH_DIR))
os.environ.setdefault("NLTK_DATA", str(NLTK_DIR))

# Make sure we can `import whisperx` from the repo root.
sys.path.insert(0, str(REPO_ROOT))


def load_env_local() -> None:
    """Load .env.local into os.environ (no python-dotenv dep needed)."""
    env_file = REPO_ROOT / ".env.local"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


WHISPER_SIZES_DEFAULT = ["tiny", "base", "small", "medium", "large-v3"]
DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"


def download_whisper(sizes: list[str]) -> list[tuple[str, bool, str]]:
    from faster_whisper import WhisperModel

    results: list[tuple[str, bool, str]] = []
    for size in sizes:
        t0 = time.time()
        print(f"\n[whisper] downloading {size!r} -> {WHISPER_DIR}")
        try:
            # Initializing WhisperModel triggers the download.
            # device="cpu" + compute_type="int8" is the cheapest way to materialize
            # the snapshot without allocating GPU memory.
            WhisperModel(
                size,
                device="cpu",
                compute_type="int8",
                download_root=str(WHISPER_DIR),
            )
            dt = time.time() - t0
            results.append((f"whisper:{size}", True, f"ok in {dt:.1f}s"))
        except Exception as exc:
            results.append((f"whisper:{size}", False, repr(exc)))
            print(f"  FAILED: {exc}")
    return results


def download_align_hf(languages: list[str], hf_token: str | None) -> list[tuple[str, bool, str]]:
    from huggingface_hub import snapshot_download

    from whisperx.alignment import DEFAULT_ALIGN_MODELS_HF

    results: list[tuple[str, bool, str]] = []
    for lang in languages:
        repo = DEFAULT_ALIGN_MODELS_HF.get(lang)
        if repo is None:
            print(f"\n[align-hf] skipping {lang!r}: not in DEFAULT_ALIGN_MODELS_HF")
            results.append((f"align-hf:{lang}", True, "skipped (no entry)"))
            continue
        t0 = time.time()
        print(f"\n[align-hf] downloading {lang!r} -> {repo}")
        try:
            snapshot_download(
                repo_id=repo,
                cache_dir=str(HF_DIR / "hub"),
                token=hf_token,
            )
            dt = time.time() - t0
            results.append((f"align-hf:{lang}={repo}", True, f"ok in {dt:.1f}s"))
        except Exception as exc:
            results.append((f"align-hf:{lang}={repo}", False, repr(exc)))
            print(f"  FAILED: {exc}")
    return results


def download_align_torch(languages: list[str]) -> list[tuple[str, bool, str]]:
    import torchaudio

    from whisperx.alignment import DEFAULT_ALIGN_MODELS_TORCH

    results: list[tuple[str, bool, str]] = []
    for lang in languages:
        bundle_name = DEFAULT_ALIGN_MODELS_TORCH.get(lang)
        if bundle_name is None:
            print(f"\n[align-torch] skipping {lang!r}: not in DEFAULT_ALIGN_MODELS_TORCH")
            results.append((f"align-torch:{lang}", True, "skipped (no entry)"))
            continue
        t0 = time.time()
        print(f"\n[align-torch] downloading {lang!r} -> {bundle_name}")
        try:
            bundle = torchaudio.pipelines.__dict__[bundle_name]
            bundle.get_model(dl_kwargs={"model_dir": str(TORCH_DIR / "hub" / "checkpoints")})
            dt = time.time() - t0
            results.append((f"align-torch:{lang}={bundle_name}", True, f"ok in {dt:.1f}s"))
        except Exception as exc:
            results.append((f"align-torch:{lang}={bundle_name}", False, repr(exc)))
            print(f"  FAILED: {exc}")
    return results


def download_diarization(hf_token: str) -> list[tuple[str, bool, str]]:
    from pyannote.audio import Pipeline

    print(f"\n[diarize] downloading {DIARIZATION_MODEL}")
    t0 = time.time()
    try:
        Pipeline.from_pretrained(
            DIARIZATION_MODEL,
            token=hf_token,
            cache_dir=str(HF_DIR / "hub"),
        )
        dt = time.time() - t0
        return [(f"diarize:{DIARIZATION_MODEL}", True, f"ok in {dt:.1f}s")]
    except Exception as exc:
        return [(f"diarize:{DIARIZATION_MODEL}", False, repr(exc))]


def _nltk_fallback_from_system(resource: str) -> bool:
    """If nltk.download fails, copy the resource from common system paths."""
    import shutil

    candidate_roots = [
        Path.home() / "nltk_data",
        Path("/usr/share/nltk_data"),
        Path("/usr/local/share/nltk_data"),
        Path("/usr/local/lib/nltk_data"),
    ]
    # Resource paths within NLTK_DATA: tokenizers/punkt_tab, corpora/<name>, etc.
    for root in candidate_roots:
        for sub in ("tokenizers", "corpora", "taggers"):
            src = root / sub / resource
            if src.exists():
                dst = NLTK_DIR / sub / resource
                dst.parent.mkdir(parents=True, exist_ok=True)
                if dst.exists():
                    return True
                shutil.copytree(src, dst)
                print(f"  fallback: copied {src} -> {dst}")
                return True
    return False


def download_nltk() -> list[tuple[str, bool, str]]:
    """whisperx.align uses NLTK punkt_tab for sentence splitting.

    NLTK's download server has a history of SSL hiccups; fall back to copying
    from a system nltk_data directory if available.
    """
    import nltk

    NLTK_DIR.mkdir(parents=True, exist_ok=True)
    results: list[tuple[str, bool, str]] = []
    for resource in ("punkt_tab",):
        t0 = time.time()
        print(f"\n[nltk] downloading {resource} -> {NLTK_DIR}")
        ok = False
        msg = ""
        try:
            ok = bool(nltk.download(resource, download_dir=str(NLTK_DIR), quiet=True))
            msg = "downloaded"
        except Exception as exc:
            msg = repr(exc)
        if not ok:
            print(f"  primary download failed ({msg}); trying system fallback")
            ok = _nltk_fallback_from_system(resource)
            msg = "from system" if ok else f"all sources failed ({msg})"
        dt = time.time() - t0
        results.append((f"nltk:{resource}", ok, f"{msg} in {dt:.1f}s"))
    return results


def download_silero() -> list[tuple[str, bool, str]]:
    import torch

    print("\n[vad-silero] downloading snakers4/silero-vad")
    t0 = time.time()
    try:
        torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            onnx=False,
            trust_repo=True,
        )
        dt = time.time() - t0
        return [("vad-silero", True, f"ok in {dt:.1f}s")]
    except Exception as exc:
        return [("vad-silero", False, repr(exc))]


def main() -> int:
    parser = argparse.ArgumentParser(description="Pre-download all whisperX models.")
    parser.add_argument(
        "--whisper-sizes",
        nargs="*",
        default=WHISPER_SIZES_DEFAULT,
        help=f"Whisper sizes to download (default: {' '.join(WHISPER_SIZES_DEFAULT)}).",
    )
    parser.add_argument(
        "--languages",
        nargs="*",
        default=None,
        help="Alignment languages to download. Default: a curated multilingual pack.",
    )
    parser.add_argument(
        "--all-languages",
        action="store_true",
        help="Download every language in DEFAULT_ALIGN_MODELS_HF + _TORCH.",
    )
    parser.add_argument("--skip-whisper", action="store_true")
    parser.add_argument("--skip-align", action="store_true")
    parser.add_argument("--skip-diarization", action="store_true")
    parser.add_argument("--skip-nltk", action="store_true")
    parser.add_argument("--include-silero", action="store_true",
                        help="Also download Silero VAD (default: skip; pyannote VAD ships in repo).")
    args = parser.parse_args()

    load_env_local()
    hf_token = os.environ.get("HF_TOKEN", "").strip() or None

    # Default multilingual pack: cover the most-used 16 languages out of the box.
    default_pack = [
        "en", "zh", "ja", "ko", "fr", "de", "es", "it",
        "ru", "pt", "ar", "vi", "hi", "nl", "pl", "tr",
    ]

    if args.all_languages:
        from whisperx.alignment import DEFAULT_ALIGN_MODELS_HF, DEFAULT_ALIGN_MODELS_TORCH
        languages = sorted(set(DEFAULT_ALIGN_MODELS_HF) | set(DEFAULT_ALIGN_MODELS_TORCH))
    elif args.languages:
        languages = args.languages
    else:
        languages = default_pack

    WHISPER_DIR.mkdir(parents=True, exist_ok=True)
    HF_DIR.mkdir(parents=True, exist_ok=True)
    (HF_DIR / "hub").mkdir(parents=True, exist_ok=True)
    TORCH_DIR.mkdir(parents=True, exist_ok=True)

    print(f"REPO_ROOT       = {REPO_ROOT}")
    print(f"WHISPER_DIR     = {WHISPER_DIR}")
    print(f"HF_DIR          = {HF_DIR}")
    print(f"TORCH_DIR       = {TORCH_DIR}")
    print(f"HF_TOKEN        = {'set' if hf_token else 'NOT SET'}")
    print(f"whisper sizes   = {args.whisper_sizes}")
    print(f"align languages = {languages}")
    print(f"diarization     = {'skip' if args.skip_diarization else DIARIZATION_MODEL}")
    print(f"silero VAD      = {'yes' if args.include_silero else 'no'}")

    summary: list[tuple[str, bool, str]] = []

    if not args.skip_whisper:
        summary += download_whisper(args.whisper_sizes)

    if not args.skip_align:
        summary += download_align_torch(languages)
        summary += download_align_hf(languages, hf_token)

    if not args.skip_diarization:
        if not hf_token:
            summary.append(
                (f"diarize:{DIARIZATION_MODEL}", False,
                 "HF_TOKEN missing; set HF_TOKEN in .env.local or use --skip-diarization")
            )
        else:
            summary += download_diarization(hf_token)

    if not args.skip_nltk:
        summary += download_nltk()

    if args.include_silero:
        summary += download_silero()

    print("\n" + "=" * 78)
    print("Download summary")
    print("=" * 78)
    failures = 0
    for name, ok, msg in summary:
        status = "OK  " if ok else "FAIL"
        print(f"  [{status}] {name}: {msg}")
        if not ok:
            failures += 1
    print("=" * 78)
    print(f"Total: {len(summary)}, failures: {failures}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
