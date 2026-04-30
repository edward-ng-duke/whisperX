"""FastAPI server exposing whisperX as an HTTP API.

Run with:
    uv run uvicorn deploy.api.server:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

# Load .env.local before importing pipeline so DEVICE / WHISPER_MODEL etc.
# are honored.
_ENV_FILE = Path(__file__).resolve().parents[2] / ".env.local"
if _ENV_FILE.exists():
    for _line in _ENV_FILE.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        _k, _, _v = _line.partition("=")
        os.environ.setdefault(_k.strip(), _v.strip())

from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402

from deploy.api import pipeline  # noqa: E402
from deploy.api.schemas import (  # noqa: E402
    DiarizeResponse,
    ErrorResponse,
    HealthResponse,
    ModelsResponse,
    TranscribeResponse,
    UnloadRequest,
    UnloadResponse,
)

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("whisperx-api")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info("WhisperX API starting")
    logger.info("  REPO_ROOT     = %s", pipeline.REPO_ROOT)
    logger.info("  MODELS_ROOT   = %s", pipeline.MODELS_ROOT)
    logger.info("  DEVICE        = %s", pipeline.DEVICE)
    logger.info("  COMPUTE_TYPE  = %s", pipeline.COMPUTE_TYPE)
    logger.info("  WHISPER_MODEL = %s", pipeline.DEFAULT_WHISPER_MODEL)
    logger.info("  ASR_CACHE_SIZE   = %d (LRU; oldest non-default size evicted)", pipeline.ASR_CACHE_SIZE)
    logger.info("  ALIGN_CACHE_SIZE = %d (LRU; oldest language evicted)", pipeline.ALIGN_CACHE_SIZE)
    logger.info("  whisper sizes on disk: %s", pipeline.list_local_whisper_models())
    logger.info("  align languages on disk: %s", pipeline.list_local_align_languages())
    logger.info("  diarization ready: %s", pipeline.diarization_ready())
    yield
    logger.info("WhisperX API shutting down")


app = FastAPI(
    title="WhisperX API",
    version="1.0.0",
    description="Private, fully offline HTTP wrapper around whisperX (ASR + word-level alignment + speaker diarization).",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        device=pipeline.DEVICE,
        compute_type=pipeline.COMPUTE_TYPE,
        default_model=pipeline.DEFAULT_WHISPER_MODEL,
        models_root=str(pipeline.MODELS_ROOT),
        whisper_models_loaded_in_memory=sorted(pipeline._asr_models.keys()),
        align_languages_loaded_in_memory=sorted(pipeline._align_models.keys()),
        asr_cache_size=pipeline.ASR_CACHE_SIZE,
        align_cache_size=pipeline.ALIGN_CACHE_SIZE,
        cuda_memory=pipeline.cuda_memory_stats(),
    )


@app.post("/admin/unload", response_model=UnloadResponse)
def unload(req: UnloadRequest | None = None) -> UnloadResponse:
    """Drop cached models and call torch.cuda.empty_cache(). Operators can use
    this to reclaim VRAM after long-running multi-language traffic without
    restarting the container.
    """
    body = req or UnloadRequest()
    result = pipeline.unload_models(asr=body.asr, align=body.align, diarize=body.diarize)
    return UnloadResponse(
        asr=result["asr"],  # type: ignore[arg-type]
        align=result["align"],  # type: ignore[arg-type]
        diarize=bool(result["diarize"]),
        cuda_memory=pipeline.cuda_memory_stats(),
    )


@app.get("/models", response_model=ModelsResponse)
def models() -> ModelsResponse:
    return ModelsResponse(
        whisper_sizes_available=pipeline.list_local_whisper_models(),
        align_languages_available=pipeline.list_local_align_languages(),
        diarization_ready=pipeline.diarization_ready(),
        default_model=pipeline.DEFAULT_WHISPER_MODEL,
    )


@app.post(
    "/transcribe",
    response_model=TranscribeResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def transcribe_endpoint(
    file: UploadFile = File(..., description="Audio file (wav/mp3/m4a/flac/ogg/webm/...)."),
    model: Optional[str] = Form(None, description="Whisper size (defaults to env WHISPER_MODEL)."),
    language: Optional[str] = Form(None, description="ISO 639-1 code; auto-detect if omitted."),
    diarize: bool = Form(False, description="Run speaker diarization."),
    min_speakers: Optional[int] = Form(None, description="Lower bound for diarization."),
    max_speakers: Optional[int] = Form(None, description="Upper bound for diarization."),
    num_speakers: Optional[int] = Form(None, description="Exact speaker count if known."),
    batch_size: int = Form(16, description="ASR batch size."),
) -> TranscribeResponse:
    if file.filename is None or file.size == 0:
        raise HTTPException(status_code=400, detail="empty upload")

    suffix = Path(file.filename).suffix or ".wav"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        try:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                tmp.write(chunk)
        finally:
            tmp.close()

        try:
            result = pipeline.transcribe(
                tmp.name,
                model=model,
                language=language,
                batch_size=batch_size,
                diarize=diarize,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                num_speakers=num_speakers,
            )
        except FileNotFoundError as exc:
            logger.exception("model file missing")
            raise HTTPException(status_code=500, detail=f"model file missing: {exc}") from exc
        except ValueError as exc:
            # whisperx raises ValueError for unsupported language etc.
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("transcription failed")
            raise HTTPException(status_code=500, detail=repr(exc)) from exc

        return JSONResponse(content=_jsonable(result))
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


@app.post(
    "/diarize",
    response_model=DiarizeResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def diarize_endpoint(
    file: UploadFile = File(..., description="Audio file (wav/mp3/m4a/flac/ogg/webm/...)."),
    min_speakers: Optional[int] = Form(None, description="Lower bound for diarization."),
    max_speakers: Optional[int] = Form(None, description="Upper bound for diarization."),
    num_speakers: Optional[int] = Form(None, description="Exact speaker count if known."),
) -> DiarizeResponse:
    if file.filename is None or file.size == 0:
        raise HTTPException(status_code=400, detail="empty upload")

    suffix = Path(file.filename).suffix or ".wav"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        try:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                tmp.write(chunk)
        finally:
            tmp.close()

        try:
            segments = pipeline.diarize_only(
                tmp.name,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                num_speakers=num_speakers,
            )
        except FileNotFoundError as exc:
            logger.exception("diarization model file missing")
            raise HTTPException(status_code=500, detail=f"model file missing: {exc}") from exc
        except Exception as exc:
            logger.exception("diarization failed")
            raise HTTPException(status_code=500, detail=repr(exc)) from exc

        return JSONResponse(content={"segments": _jsonable(segments)})
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


def _jsonable(obj):
    """whisperX returns numpy arrays / floats inside dicts; coerce to JSON-safe."""
    import numpy as np

    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
