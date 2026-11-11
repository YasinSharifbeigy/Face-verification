"""Run actual GPU and API checks using the public sample images in samples/."""
import json
import os
import cv2
import numpy as np
from fastapi.testclient import TestClient
from verify import ROOT, FaceVerifier, read_image


def main():
    engine = FaceVerifier(os.getenv("FACE_DEVICE", "cuda"))
    a = read_image(ROOT / "samples/astronaut.png")
    b = read_image(ROOT / "samples/grace_hopper.jpg")
    same = engine.verify(a, a)
    different = engine.verify(a, b)
    transformed = engine.verify(a, cv2.convertScaleAbs(a, alpha=0.8, beta=10))
    assert same["score"] > 0.999 and same["same_person"]
    assert transformed["same_person"]
    assert not different["same_person"]
    for image in [np.zeros((640, 640, 3), np.uint8), np.concatenate([a, a], axis=1)]:
        try:
            engine.embedding(image)
        except ValueError:
            pass
        else:
            raise AssertionError("Expected rejection for zero or multiple faces")
    # Reuse the validated engine, avoiding a second GPU allocation.
    from app import app
    app.state.verifier = engine
    client = TestClient(app)
    assert client.get("/health").status_code == 200
    content = (ROOT / "samples/astronaut.png").read_bytes()
    response = client.post("/verify", files={"reference": ("a.png", content, "image/png"), "candidate": ("b.png", content, "image/png")})
    assert response.status_code == 200, response.text
    assert response.json()["same_person"]
    response = client.post("/verify", files={"reference": ("bad", b"invalid"), "candidate": ("a.png", content)})
    assert response.status_code == 422
    response = client.post("/verify", files={"reference": ("empty", b""), "candidate": ("a.png", content)})
    assert response.status_code == 422
    print(json.dumps({"identical": same, "brightness_change": transformed,
                      "different_people": different, "api": "passed", "invalid_inputs": "passed"}, indent=2))


if __name__ == "__main__":
    main()
