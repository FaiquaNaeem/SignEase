"""Speech-to-text with ordered fallback: Sarvam (cloud, best quality for
Indian languages) first, then a local faster-whisper model (GPU-accelerated
on this machine, zero cost/keys) if Sarvam is unconfigured or errors.
"""
from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"

_whisper_model = None


class STTError(Exception):
    pass


async def _transcribe_sarvam(audio_bytes: bytes, filename: str, language_code: str) -> dict:
    api_key = os.environ.get("SARVAM_API_KEY")
    if not api_key:
        raise STTError("SARVAM_API_KEY not configured")

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            SARVAM_STT_URL,
            headers={"api-subscription-key": api_key},
            data={"model": "saaras:v3", "language_code": language_code},
            files={"file": (filename, audio_bytes, "audio/webm")},
        )
    if resp.status_code != 200:
        raise STTError(f"Sarvam STT error: {resp.status_code} {resp.text}")
    data = resp.json()
    return {"transcript": data.get("transcript", ""), "language_code": data.get("language_code")}


def _ensure_cuda_libs_on_path() -> None:
    """faster-whisper's CTranslate2 backend dlopen()s libcublas/libcudnn at
    runtime but doesn't know to look inside the pip-installed nvidia-*-cu12
    wheels (as opposed to a system CUDA install) — without this, GPU mode
    fails with 'libcublas.so.12 is not found or cannot be loaded' even
    though the .so files are present on disk.
    """
    import importlib.util

    ld_paths = []
    for pkg in ("nvidia.cublas.lib", "nvidia.cudnn.lib"):
        spec = importlib.util.find_spec(pkg)
        if spec and spec.submodule_search_locations:
            ld_paths.append(next(iter(spec.submodule_search_locations)))
    if ld_paths:
        existing = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = ":".join(ld_paths + ([existing] if existing else []))


def _get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            _ensure_cuda_libs_on_path()
        from faster_whisper import WhisperModel

        compute_type = "float16" if device == "cuda" else "int8"
        _whisper_model = WhisperModel("small", device=device, compute_type=compute_type)
    return _whisper_model


def _transcribe_whisper(audio_bytes: bytes, language_code: str) -> dict:
    model = _get_whisper_model()
    # faster-whisper needs a real file path (it shells out to ffmpeg/PyAV
    # for decoding); write the upload to a temp file rather than trying to
    # guess a fixed input format.
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=True) as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        lang = None if language_code == "unknown" else language_code.split("-")[0]
        segments, info = model.transcribe(tmp.name, language=lang, beam_size=5)
        transcript = " ".join(segment.text.strip() for segment in segments)
    return {"transcript": transcript.strip(), "language_code": info.language}


async def transcribe_with_fallback(audio_bytes: bytes, filename: str, language_code: str) -> tuple[dict, str]:
    """Returns (result_dict, provider_used). Raises STTError only if every
    tier fails."""
    try:
        result = await _transcribe_sarvam(audio_bytes, filename, language_code)
        return result, "sarvam"
    except Exception as exc:  # noqa: BLE001 - any Sarvam failure falls through
        logger.warning("Sarvam STT failed, falling back to local faster-whisper: %s", exc)

    try:
        result = _transcribe_whisper(audio_bytes, language_code)
        return result, "whisper-local"
    except Exception as exc:  # noqa: BLE001
        logger.error("Local faster-whisper also failed: %s", exc)
        raise STTError("All STT providers failed") from exc
