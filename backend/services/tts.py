"""Text-to-speech with ordered fallback: Sarvam (cloud, best quality) first,
then a local Piper model (always available, zero cost/keys, runs on-device)
if Sarvam is unconfigured or errors. The browser's Web Speech API is the
final fallback tier, handled client-side when both of these fail.
"""
from __future__ import annotations

import base64
import io
import logging
import os
import wave
from pathlib import Path
from typing import Literal

import httpx

logger = logging.getLogger(__name__)

SignLanguage = Literal["en-IN", "hi-IN"]

SARVAM_TTS_URL = "https://api.sarvam.ai/text-to-speech"
MODELS_DIR = Path(__file__).resolve().parents[1] / "tts_models"

PIPER_VOICE_PATHS: dict[SignLanguage, Path] = {
    "en-IN": MODELS_DIR / "en_US-amy-medium.onnx",
    "hi-IN": MODELS_DIR / "hi_IN-pratham-medium.onnx",
}

_piper_voices: dict[SignLanguage, object] = {}


class TTSError(Exception):
    pass


async def _speak_sarvam(text: str, language: SignLanguage) -> bytes:
    api_key = os.environ.get("SARVAM_API_KEY")
    if not api_key:
        raise TTSError("SARVAM_API_KEY not configured")

    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(
            SARVAM_TTS_URL,
            headers={"api-subscription-key": api_key, "Content-Type": "application/json"},
            json={
                "text": text,
                "target_language_code": language,
                "model": "bulbul:v2",
                "output_audio_codec": "wav",
            },
        )
    if resp.status_code != 200:
        raise TTSError(f"Sarvam TTS error: {resp.status_code} {resp.text}")
    data = resp.json()
    return base64.b64decode(data["audios"][0])


def _get_piper_voice(language: SignLanguage):
    if language not in _piper_voices:
        from piper import PiperVoice

        model_path = PIPER_VOICE_PATHS.get(language)
        if model_path is None or not model_path.exists():
            raise TTSError(f"No local Piper voice available for {language}")
        _piper_voices[language] = PiperVoice.load(str(model_path))
    return _piper_voices[language]


def _speak_piper(text: str, language: SignLanguage) -> bytes:
    voice = _get_piper_voice(language)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        voice.synthesize_wav(text, wav_file)
    return buffer.getvalue()


async def speak_with_fallback(text: str, language: SignLanguage) -> tuple[bytes, str]:
    """Returns (wav_bytes, provider_used). Raises TTSError only if every
    tier fails — at that point the client falls back to Web Speech API."""
    try:
        audio = await _speak_sarvam(text, language)
        return audio, "sarvam"
    except Exception as exc:  # noqa: BLE001 - deliberately broad: any Sarvam failure falls through
        logger.warning("Sarvam TTS failed, falling back to local Piper: %s", exc)

    try:
        audio = _speak_piper(text, language)
        return audio, "piper"
    except Exception as exc:  # noqa: BLE001
        logger.error("Local Piper TTS also failed: %s", exc)
        raise TTSError("All TTS providers failed") from exc
