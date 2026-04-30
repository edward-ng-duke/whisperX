"""Pydantic response models for the WhisperX API."""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class WordSegment(BaseModel):
    word: str
    start: Optional[float] = None
    end: Optional[float] = None
    score: Optional[float] = None
    speaker: Optional[str] = None


class Segment(BaseModel):
    start: float
    end: float
    text: str
    words: list[WordSegment] = Field(default_factory=list)
    speaker: Optional[str] = None


class TranscribeResponse(BaseModel):
    language: str
    model: str
    diarization: bool
    segments: list[Segment] = Field(default_factory=list)
    word_segments: list[WordSegment] = Field(default_factory=list)


class DiarizeSegment(BaseModel):
    start: float
    end: float
    speaker: str


class DiarizeResponse(BaseModel):
    segments: list[DiarizeSegment] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    device: str
    compute_type: str
    default_model: str
    models_root: str
    whisper_models_loaded_in_memory: list[str]
    align_languages_loaded_in_memory: list[str] = Field(default_factory=list)
    asr_cache_size: int = 0
    align_cache_size: int = 0
    cuda_memory: Optional[dict[str, int]] = None


class ModelsResponse(BaseModel):
    whisper_sizes_available: list[str]
    align_languages_available: list[str]
    diarization_ready: bool
    default_model: str


class UnloadRequest(BaseModel):
    asr: bool = True
    align: bool = True
    diarize: bool = False


class UnloadResponse(BaseModel):
    asr: list[str]
    align: list[str]
    diarize: bool
    cuda_memory: Optional[dict[str, int]] = None


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[Any] = None
