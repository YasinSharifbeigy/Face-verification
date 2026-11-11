"""LVFace-B face verification. Run with --help for CLI usage."""
import argparse
import importlib.util
import json
import math
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from insightface.model_zoo import get_model
from insightface.utils.face_align import norm_crop

ROOT = Path(__file__).resolve().parent


class FaceVerifier:
    def __init__(self, device="cuda"):
        if device == "cuda":
            # Importing a CUDA-enabled PyTorch build loads its runtime libraries
            # before ONNX Runtime creates the CUDA session. Local environments
            # without torch can still use installed CUDA wheels or local DLLs.
            if importlib.util.find_spec("torch") is not None:
                importlib.import_module("torch")
            local_runtime = ROOT / ".venv/cuda"
            ort.preload_dlls(directory=str(local_runtime) if local_runtime.exists() else None)
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        path = ROOT / "models/LVFace-B_Glint360K/LVFace-B_Glint360K.onnx"
        detector_path = ROOT / "models/det_10g.onnx"
        if not path.exists() or not detector_path.exists():
            raise FileNotFoundError("Models missing. Run python download_models.py first.")
        self.model = ort.InferenceSession(str(path), sess_options=options, providers=providers)
        if device == "cuda" and "CUDAExecutionProvider" not in self.model.get_providers():
            raise RuntimeError("CUDA provider failed to load. Use --device cpu or repair CUDA dependencies.")
        # Detection is inexpensive on CPU and leaves GPU memory for recognition.
        self.detector = get_model(str(detector_path), providers=["CPUExecutionProvider"])
        self.detector.prepare(ctx_id=-1, input_size=(640, 640), det_thresh=0.5)

    def embedding(self, image):
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Expected a readable color image.")
        boxes, landmarks = self.detector.detect(image, max_num=0)
        if len(boxes) != 1:
            raise ValueError(f"Expected exactly one face; detected {len(boxes)}.")
        face = norm_crop(image, landmark=landmarks[0], image_size=112)
        # Match the official LVFace ONNX preprocessing: RGB, NCHW, [-1, 1].
        tensor = (face[:, :, ::-1].astype(np.float32) / 127.5 - 1).transpose(2, 0, 1)[None]
        vector = self.model.run(None, {self.model.get_inputs()[0].name: np.ascontiguousarray(tensor)})[0].reshape(-1)
        norm = np.linalg.norm(vector)
        if not np.isfinite(vector).all() or norm <= 0:
            raise RuntimeError("Model returned an invalid embedding.")
        return vector / norm

    def verify(self, reference, candidate, threshold=0.4):
        if not math.isfinite(threshold) or not -1 <= threshold <= 1:
            raise ValueError("Threshold must be finite and between -1 and 1.")
        a, b = self.embedding(reference), self.embedding(candidate)
        score = float(np.clip(np.dot(a, b), -1, 1))
        return {"score": score, "same_person": score >= threshold, "threshold": threshold,
                "threshold_calibrated": False, "model": "LVFace-B_Glint360K",
                "provider": self.model.get_providers()[0]}


def read_image(path):
    content = np.fromfile(path, dtype=np.uint8)
    if content.size == 0:
        raise ValueError(f"Image is empty: {path}")
    try:
        image = cv2.imdecode(content, cv2.IMREAD_COLOR)
    except cv2.error as exc:
        raise ValueError(f"Cannot decode image: {path}") from exc
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--threshold", type=float, default=0.4, help="Uncalibrated starting value; cosine score is not a probability.")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    try:
        result = FaceVerifier(args.device).verify(read_image(args.reference), read_image(args.candidate), args.threshold)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
