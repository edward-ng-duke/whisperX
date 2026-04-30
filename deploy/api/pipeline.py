"""WhisperX inference pipeline with model singletons.

All models are loaded from ./models/ in offline mode. The first call for each
(asr_model, language, diarize) combination is slow because the model is
materialized into memory; subsequent calls reuse the cached instance.
"""
from __future__ import annotations

import ctypes
import os
import threading
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_ROOT = REPO_ROOT / "models"
WHISPER_DIR = MODELS_ROOT / "whisper"
HF_DIR = MODELS_ROOT / "hf"
TORCH_DIR = MODELS_ROOT / "torch"
NLTK_DIR = MODELS_ROOT / "nltk_data"


def _preload_cudnn() -> None:
    """ctranslate2 / pyannote dlopen libcudnn_*.so.9 by short name; the pip-installed
    cuDNN under nvidia/cudnn/lib is not on the dynamic-linker search path. Preloading
    each library globally with ctypes registers the symbols so subsequent dlopens
    succeed without the user having to set LD_LIBRARY_PATH.
    """
    try:
        import nvidia.cudnn  # type: ignore[import-not-found]
    except ImportError:
        return
    # nvidia.cudnn is a namespace package; __file__ is None, so resolve via __path__.
    paths = list(getattr(nvidia.cudnn, "__path__", []))
    if not paths:
        return
    cudnn_dir = Path(paths[0]) / "lib"
    if not cudnn_dir.is_dir():
        return
    # Order matters: load deps before dependents.
    for libname in (
        "libcudnn_graph.so.9",
        "libcudnn_ops.so.9",
        "libcudnn_engines_runtime_compiled.so.9",
        "libcudnn_engines_precompiled.so.9",
        "libcudnn_heuristic.so.9",
        "libcudnn_cnn.so.9",
        "libcudnn_adv.so.9",
        "libcudnn.so.9",
    ):
        path = cudnn_dir / libname
        if path.exists():
            try:
                ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass


_preload_cudnn()

# Force everything offline. Set BEFORE importing torch / huggingface_hub /
# transformers / pyannote.
os.environ.setdefault("HF_HOME", str(HF_DIR))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_DIR / "hub"))
os.environ.setdefault("TORCH_HOME", str(TORCH_DIR))
os.environ.setdefault("NLTK_DATA", str(NLTK_DIR))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import whisperx  # noqa: E402
from whisperx.alignment import DEFAULT_ALIGN_MODELS_TORCH  # noqa: E402
from whisperx.diarize import DiarizationPipeline  # noqa: E402

DEVICE = os.environ.get("DEVICE", "cuda").strip() or "cuda"
COMPUTE_TYPE = os.environ.get("COMPUTE_TYPE", "float16" if DEVICE == "cuda" else "int8").strip()
DEFAULT_WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small").strip() or "small"
DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"

_asr_lock = threading.Lock()
_align_lock = threading.Lock()
_diarize_lock = threading.Lock()
_transcribe_lock = threading.Lock()  # whisperX is not thread-safe per process

_asr_models: dict[str, Any] = {}
_align_models: dict[str, tuple[Any, dict]] = {}
_diarize_pipeline: Any = None


def get_asr_model(name: str) -> Any:
    with _asr_lock:
        if name not in _asr_models:
            _asr_models[name] = whisperx.load_model(
                name,
                DEVICE,
                compute_type=COMPUTE_TYPE,
                download_root=str(WHISPER_DIR),
                local_files_only=True,
            )
        return _asr_models[name]


def _align_model_dir(language: str) -> str:
    """torchaudio bundles want the directory containing the .pt; HF wav2vec2 want the HF hub root."""
    if language in DEFAULT_ALIGN_MODELS_TORCH:
        return str(TORCH_DIR / "hub" / "checkpoints")
    return str(HF_DIR / "hub")


def get_align_model(language: str) -> tuple[Any, dict]:
    with _align_lock:
        if language not in _align_models:
            _align_models[language] = whisperx.load_align_model(
                language_code=language,
                device=DEVICE,
                model_dir=_align_model_dir(language),
                model_cache_only=True,
            )
        return _align_models[language]


def get_diarize_pipeline() -> Any:
    global _diarize_pipeline
    with _diarize_lock:
        if _diarize_pipeline is None:
            _diarize_pipeline = DiarizationPipeline(
                model_name=DIARIZATION_MODEL,
                token=os.environ.get("HF_TOKEN") or None,  # may be unused offline
                device=DEVICE,
                cache_dir=str(HF_DIR / "hub"),
            )
        return _diarize_pipeline


def list_local_whisper_models() -> list[str]:
    """Inspect ./models/whisper/ and return the model size names that look ready.

    faster-whisper's snapshot layout under download_root is
        <download_root>/models--Systran--faster-whisper-<size>/snapshots/<rev>/...
    """
    if not WHISPER_DIR.exists():
        return []
    found: list[str] = []
    for entry in WHISPER_DIR.iterdir():
        name = entry.name
        if name.startswith("models--Systran--faster-whisper-"):
            size = name.replace("models--Systran--faster-whisper-", "")
            # require at least one snapshot dir
            snap_dir = entry / "snapshots"
            if snap_dir.exists() and any(snap_dir.iterdir()):
                found.append(size)
    return sorted(set(found))


def list_local_align_languages() -> list[str]:
    """Heuristic: map cached HF wav2vec2 repos back to language codes."""
    from whisperx.alignment import DEFAULT_ALIGN_MODELS_HF, DEFAULT_ALIGN_MODELS_TORCH

    hub = HF_DIR / "hub"
    cached: list[str] = []
    if hub.exists():
        repo_dirs = {p.name for p in hub.iterdir() if p.is_dir()}
        for lang, repo in DEFAULT_ALIGN_MODELS_HF.items():
            if f"models--{repo.replace('/', '--')}" in repo_dirs:
                cached.append(lang)

    # Torchaudio-based bundles cache files under torch/hub/checkpoints
    ta_root = TORCH_DIR / "hub" / "checkpoints"
    if ta_root.exists() and any(ta_root.iterdir()):
        for lang in DEFAULT_ALIGN_MODELS_TORCH:
            if lang not in cached:
                cached.append(lang)

    return sorted(cached)


def diarization_ready() -> bool:
    repo = DIARIZATION_MODEL.replace("/", "--")
    return (HF_DIR / "hub" / f"models--{repo}").exists()


def transcribe(
    audio_path: str,
    *,
    model: str | None = None,
    language: str | None = None,
    batch_size: int = 16,
    diarize: bool = False,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    num_speakers: int | None = None,
) -> dict[str, Any]:
    """Run the full whisperX pipeline on a single audio file."""
    model_name = (model or DEFAULT_WHISPER_MODEL).strip()

    with _transcribe_lock:
        asr = get_asr_model(model_name)
        audio = whisperx.load_audio(audio_path)

        result = asr.transcribe(audio, batch_size=batch_size, language=language)
        detected_language = result.get("language", language or "unknown")

        align_model, align_metadata = get_align_model(detected_language)
        aligned = whisperx.align(
            result["segments"],
            align_model,
            align_metadata,
            audio,
            DEVICE,
            return_char_alignments=False,
        )

        out: dict[str, Any] = {
            "language": detected_language,
            "model": model_name,
            "segments": aligned.get("segments", []),
            "word_segments": aligned.get("word_segments", []),
        }

        if diarize:
            pipe = get_diarize_pipeline()
            diarize_segments = pipe(
                audio,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )
            with_speakers = whisperx.assign_word_speakers(diarize_segments, aligned)
            out["segments"] = with_speakers.get("segments", out["segments"])
            out["word_segments"] = with_speakers.get("word_segments", out["word_segments"])
            out["diarization"] = True
        else:
            out["diarization"] = False

        return out


def diarize_only(
    audio_path: str,
    *,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    num_speakers: int | None = None,
) -> list[dict[str, Any]]:
    """Speaker diarization without ASR. Returns [{start, end, speaker}, ...]."""
    pipe = get_diarize_pipeline()
    audio = whisperx.load_audio(audio_path)
    with _diarize_lock:
        df = pipe(
            audio,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
    return [
        {"start": float(r["start"]), "end": float(r["end"]), "speaker": str(r["speaker"])}
        for _, r in df.iterrows()
    ]
