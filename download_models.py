"""Download official LVFace-B weights and InsightFace SCRFD detector."""
import hashlib
from pathlib import Path
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent


def main():
    models = ROOT / "models"
    models.mkdir(exist_ok=True)
    file = models / "LVFace-B_Glint360K/LVFace-B_Glint360K.onnx"
    if not file.exists():
        file = hf_hub_download("bytedance-research/LVFace", "LVFace-B_Glint360K/LVFace-B_Glint360K.onnx", local_dir=models)
    expected = "9d834ed8e927fd35b9123b2bf97c40aad05785b1f9ecfb1c4c1f6242d38d1382"
    with open(file, "rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected:
        raise RuntimeError("LVFace checkpoint SHA256 mismatch.")
    detector = models / "det_10g.onnx"
    if not detector.exists():
        # Immich mirrors the original InsightFace SCRFD-10G detector alone,
        # avoiding downloading unused recognition and attribute models.
        source = hf_hub_download("immich-app/buffalo_l", "detection/model.onnx", local_dir=models / "detector_download")
        import shutil
        shutil.copyfile(source, detector)
    with detector.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91":
        raise RuntimeError("SCRFD checkpoint SHA256 mismatch.")
    print(f"Models ready in {models}")


if __name__ == "__main__":
    main()
