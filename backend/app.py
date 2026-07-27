"""SignEase backend — single FastAPI app. Replaces every duplicate Flask
server from the old repo (simple_backend.py, working_backend.py,
production_backend.py, rtx5060_backend.py, pytorch_backend.py, ...).

Run from the repo root:
    uvicorn backend.app:app --reload --port 8000
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Explicit path, not the auto-discovery find_dotenv() default: that walks up
# from the *caller's* stack frame, which is unreliable depending on how the
# app is invoked (uvicorn CLI vs `python -c` vs test runner) and silently
# found nothing in exactly that situation during testing.
load_dotenv(Path(__file__).resolve().parent / ".env")

from backend.api.routes import router  # noqa: E402
from backend.inference.engine import InferenceEngine  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.engine = InferenceEngine()
    yield


app = FastAPI(title="SignEase API", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_origin_regex=r"chrome-extension://.*",
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
