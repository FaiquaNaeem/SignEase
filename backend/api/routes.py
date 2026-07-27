"""Single unified API contract. Replaces the two incompatible request/response
shapes the frontend audit found (`{gesture,...}` vs `{prediction,...}`) with
one schema used everywhere, and replaces every hardcoded `/metrics` number
from the old repo with values computed from real state.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException, Request, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

from backend.services.stt import STTError, transcribe_with_fallback
from backend.services.tts import TTSError, speak_with_fallback

router = APIRouter(prefix="/api")

SIGN_REFERENCES_PATH = Path("backend/checkpoints/sign_references.json")


class Landmark(BaseModel):
    x: float
    y: float
    z: float


class PredictLetterRequest(BaseModel):
    landmarks: list[Landmark]


class WordFrame(BaseModel):
    leftHand: Optional[list[Landmark]] = None
    rightHand: Optional[list[Landmark]] = None
    pose: Optional[list[Landmark]] = None


class PredictWordRequest(BaseModel):
    frames: list[WordFrame]


class Alternative(BaseModel):
    label: str
    confidence: float


class PredictResponse(BaseModel):
    label: str
    confidence: float
    alternatives: list[Alternative]
    inference_ms: float


class SpeakRequest(BaseModel):
    text: str
    language: Literal["en-IN", "hi-IN"] = "en-IN"


class TranscribeResponse(BaseModel):
    transcript: str
    language_code: Optional[str] = None
    provider: str


@router.get("/health")
def health(request: Request):
    engine = request.app.state.engine
    return {
        "status": "healthy",
        "letter_model_loaded": engine.letter_model is not None,
        "word_model_loaded": engine.word_model is not None,
        "device": str(engine.device),
        "uptime_seconds": time.time() - engine.started_at,
        "sarvam_configured": bool(os.environ.get("SARVAM_API_KEY")),
    }


@router.post("/predict/letter", response_model=PredictResponse)
def predict_letter(body: PredictLetterRequest, request: Request):
    engine = request.app.state.engine
    if len(body.landmarks) != 21:
        raise HTTPException(422, "Expected exactly 21 hand landmarks.")
    try:
        return engine.predict_letter([lm.model_dump() for lm in body.landmarks])
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc


@router.post("/predict/word", response_model=PredictResponse)
def predict_word(body: PredictWordRequest, request: Request):
    engine = request.app.state.engine
    if not body.frames:
        raise HTTPException(422, "Expected at least one frame.")
    frames = [
        {
            "leftHand": [lm.model_dump() for lm in f.leftHand] if f.leftHand else None,
            "rightHand": [lm.model_dump() for lm in f.rightHand] if f.rightHand else None,
            "pose": [lm.model_dump() for lm in f.pose] if f.pose else None,
        }
        for f in body.frames
    ]
    try:
        return engine.predict_word(frames)
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc


@router.get("/sign-references")
def sign_references():
    if not SIGN_REFERENCES_PATH.exists():
        raise HTTPException(503, "Sign references not generated yet — run export_sign_references.py.")
    return json.loads(SIGN_REFERENCES_PATH.read_text())


@router.post("/speak")
async def speak(body: SpeakRequest):
    try:
        audio_bytes, provider = await speak_with_fallback(body.text, body.language)
    except TTSError as exc:
        raise HTTPException(503, str(exc)) from exc
    return Response(content=audio_bytes, media_type="audio/wav", headers={"X-TTS-Provider": provider})


@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(file: UploadFile, language_code: str = "unknown"):
    audio_bytes = await file.read()
    try:
        result, provider = await transcribe_with_fallback(audio_bytes, file.filename or "audio.webm", language_code)
    except STTError as exc:
        raise HTTPException(503, str(exc)) from exc
    return {**result, "provider": provider}
