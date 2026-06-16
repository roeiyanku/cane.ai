"""
cane.ai — local inference server.

POST an image, get back fused navigation perception (walkable region, centerline,
obstacles with relative distance). Runs SegFormer + Depth Anything locally; no
third-party inference API.

Run:
    pip install -r requirements.txt
    uvicorn server.api:app --host 0.0.0.0 --port 8000

First request downloads model weights (a few hundred MB) and is slow; subsequent
requests are fast. A GPU is strongly recommended for real-time use.
"""

from __future__ import annotations
import io
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from .perception import Perception

app = FastAPI(title="cane.ai Perception", version="0.1")

# Allow the local browser demo (file:// or localhost) to call this.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # tighten for production
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

_model: Perception | None = None


def get_model() -> Perception:
    global _model
    if _model is None:
        _model = Perception()   # loads weights on first call
    return _model


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/segment")
async def segment(file: UploadFile = File(...)):
    """Full perception pass: segmentation + depth + fusion."""
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(400, "Expected an image upload")
    raw = await file.read()
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Could not decode image")

    # Downscale very large frames for latency; perception is robust to this.
    max_side = 768
    if max(img.size) > max_side:
        scale = max_side / max(img.size)
        img = img.resize((int(img.width * scale), int(img.height * scale)))

    result = get_model().perceive(img)
    return JSONResponse(result)
