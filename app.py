"""Local research API: .venv/Scripts/python -m uvicorn app:app"""
import os
import threading
from contextlib import asynccontextmanager
import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from verify import FaceVerifier

MAX_BYTES = 10 * 1024 * 1024
lock = threading.Lock()


@asynccontextmanager
async def lifespan(app):
    app.state.verifier = FaceVerifier(os.getenv("FACE_DEVICE", "cuda"))
    yield


app = FastAPI(title="LVFace face verification", lifespan=lifespan)


def decode(upload):
    content = upload.file.read(MAX_BYTES + 1)
    if len(content) > MAX_BYTES:
        raise HTTPException(413, "Each image must be at most 10 MiB.")
    if not content:
        raise ValueError("Uploaded image is empty.")
    try:
        image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
    except cv2.error as exc:
        raise ValueError("Uploaded file is not a readable image.") from exc
    if image is None:
        raise ValueError("Uploaded file is not a readable image.")
    return image


@app.get("/health")
def health():
    return {"status": "ok", "model": "LVFace-B_Glint360K", "providers": app.state.verifier.model.get_providers()}


@app.post("/verify")
def verify(reference: UploadFile = File(...), candidate: UploadFile = File(...),
           threshold: float = Query(0.4, ge=-1, le=1)):
    try:
        a, b = decode(reference), decode(candidate)
        with lock:
            return app.state.verifier.verify(a, b, threshold)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
