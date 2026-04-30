"""WhisperX inference pipeline with bounded model caches.

Models are loaded from ./models/ in offline mode and cached as singletons keyed
by ASR size / align language. Caches are bounded LRUs so long-running servers
that see varied requests don't grow GPU memory monotonically. The default ASR
model is pinned in the cache so the warm path stays warm.
"""
from __future__ import annotations

import ctypes
import gc
import os
import threading
from collections import OrderedDict
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


def _int_env(name: str, default: int, minimum: int = 1) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        v = int(raw.strip())
    except ValueError:
        return default
    return max(minimum, v)


# LRU capacities. Each new ASR size/language permanently parks weights on the
# GPU until evicted, so on small cards keep these small. Override via env.
ASR_CACHE_SIZE = _int_env("ASR_CACHE_SIZE", 2)
ALIGN_CACHE_SIZE = _int_env("ALIGN_CACHE_SIZE", 1)

_asr_lock = threading.Lock()
_align_lock = threading.Lock()
_diarize_lock = threading.Lock()
_transcribe_lock = threading.Lock()  # whisperX is not thread-safe per process

_asr_models: "OrderedDict[str, Any]" = OrderedDict()
_align_models: "OrderedDict[str, tuple[Any, dict]]" = OrderedDict()
_diarize_pipeline: Any = None


def _empty_cuda_cache() -> None:
    """Release PyTorch's caching allocator reservations back to the OS.

    Called after evictions and after each request so peak-sized scratch
    allocations don't stay reserved between requests.
    """
    if DEVICE != "cuda":
        return
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _evict_oldest(cache: "OrderedDict[str, Any]", keep_keys: set[str]) -> bool:
    """Pop the oldest entry from `cache` whose key is not in `keep_keys`.

    Returns True if something was evicted. Used by the LRU helpers below.
    """
    for key in list(cache.keys()):
        if key in keep_keys:
            continue
        cache.pop(key)
        return True
    return False


def get_asr_model(name: str) -> Any:
    with _asr_lock:
        if name in _asr_models:
            _asr_models.move_to_end(name)
            return _asr_models[name]

        # Always preserve the configured default model so the warm path stays warm.
        keep = {DEFAULT_WHISPER_MODEL} if DEFAULT_WHISPER_MODEL != name else set()
        evicted = False
        while len(_asr_models) >= ASR_CACHE_SIZE and _evict_oldest(_asr_models, keep):
            evicted = True
        if evicted:
            gc.collect()
            _empty_cuda_cache()

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
        if language in _align_models:
            _align_models.move_to_end(language)
            return _align_models[language]

        evicted = False
        while len(_align_models) >= ALIGN_CACHE_SIZE and _evict_oldest(_align_models, set()):
            evicted = True
        if evicted:
            gc.collect()
            _empty_cuda_cache()

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


def unload_models(*, asr: bool = True, align: bool = True, diarize: bool = False) -> dict[str, list[str] | bool]:
    """Drop cached models and free GPU memory. Useful when the server has
    accumulated long-tail languages/sizes and operators want to reclaim VRAM
    without restarting the container.
    """
    global _diarize_pipeline
    dropped_asr: list[str] = []
    dropped_align: list[str] = []
    dropped_diarize = False

    if asr:
        with _asr_lock:
            dropped_asr = list(_asr_models.keys())
            _asr_models.clear()
    if align:
        with _align_lock:
            dropped_align = list(_align_models.keys())
            _align_models.clear()
    if diarize:
        with _diarize_lock:
            if _diarize_pipeline is not None:
                _diarize_pipeline = None
                dropped_diarize = True

    gc.collect()
    _empty_cuda_cache()
    return {"asr": dropped_asr, "align": dropped_align, "diarize": dropped_diarize}


def cuda_memory_stats() -> dict[str, int] | None:
    """Return PyTorch CUDA allocator stats (bytes) or None when not on CUDA."""
    if DEVICE != "cuda":
        return None
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
    }


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
        try:
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
        finally:
            _empty_cuda_cache()


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
    try:
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
    finally:
        _empty_cuda_cache()
