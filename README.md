# LVFace face-verification API

A research API that compares a reference photograph with a candidate photograph and returns a cosine similarity score and a same-person decision. It performs **1:1 verification**, not a search through an identity database.

## Model and pipeline

The recognition model is **LVFace-B**, a Vision Transformer trained on Glint360K, from ByteDance's ICCV 2025 work, *LVFace: Progressive Cluster Optimization for Large Vision Models in Face Recognition*. This service uses the official FP32 ONNX checkpoint with ONNX Runtime. Its authors report **97.70% TAR at FAR = 0.01% on IJB-C**. That template-based benchmark is not a guarantee of accuracy on arbitrary two-photo inputs.

For each of the two images:

1. SCRFD-10G detects faces and five facial landmarks on the CPU.
2. The service requires exactly one detected face per image.
3. Landmark alignment produces a 112 x 112 face crop.
4. Pixels are converted to RGB, normalized to [-1, 1], and passed to LVFace-B on the GPU.
5. The resulting embeddings are normalized and compared using cosine similarity.
6. `same_person` is `true` when `score >= threshold`.

Models load once at startup. One worker processes verifications sequentially through a lock, with one face at a time sent to the recognition model. There is no training step, liveness detection, or face-quality rejection gate.

Sources: [official implementation](https://github.com/bytedance/LVFace), [official weights](https://huggingface.co/bytedance-research/LVFace), [InsightFace detection/alignment](https://github.com/deepinsight/insightface).

## Hardware and GPU VRAM

| Item | Guidance |
| --- | --- |
| GPU | NVIDIA GPU compatible with CUDA 12.8; the existing RTX 3090 works |
| Deployment starting budget | **4 GiB of free VRAM per API worker**, with load testing on the target machine |
| Observed memory | Approximately **1,301 MiB additional GPU memory** on this Windows RTX 3090 during loading and five sequential verifications |
| Recognition checkpoint | Approximately 434 MiB on disk; runtime allocations also require memory |
| CPU mode | Available without a GPU, with slower recognition |

The memory observation came from sampling total GPU usage with `nvidia-smi` before and during inference. Other workloads were running, so it is an approximate increment, not an isolated process peak or a proven minimum. **4 GiB is conservative planning guidance, not a tested minimum.** Container/Linux memory consumption has not been measured. Leave headroom for other processes and startup allocations; 24 GiB is not required by this service. More workers create additional model sessions and can increase VRAM consumption.

## GitHub and clone-directory layout

Commit the source code, Docker files, dependency lists, and this README. **Do not commit model binaries, `.venv/`, or personal test photos.** `models/` and `.venv/` are already ignored by Git and excluded from the Docker build context. Git ignore rules do not remove files that were already tracked; review `git status` and the staged files before pushing.

After cloning, run the Docker quick-start commands below from the clone directory. The downloader creates this layout automatically:

```text
<repository-clone>/
  models/
    LVFace-B_Glint360K/
      LVFace-B_Glint360K.onnx
    det_10g.onnx
  Dockerfile
  compose.yaml
  download_models.py
```

No local Python installation is required for the Docker download command. Alternatively, if you already have a Python environment with `huggingface-hub` installed, run `python download_models.py` from any directory: files are stored beside the script under `models/`, not relative to your shell's current directory. On this machine they already exist at `C:\Yasin\face-verfication\models`.

The local Windows Python environment remains at `<repository-clone>/.venv/`. Docker's Linux Python environment stays at `/opt/venv` inside the image; a Windows `.venv` cannot be shared with a Linux container. Teammates do not need to create `.venv` when using Docker.

## Docker quick start (GPU)

Requirements:

- Docker Engine and Compose v2, using Linux containers on an x86-64 host.
- A compatible NVIDIA driver. Linux hosts need NVIDIA Container Toolkit configured for Docker. On Windows, use Docker Desktop's WSL2 backend and working NVIDIA GPU support.
- Internet access during the first image build for packages and during the separate model download step.

Stop any existing local API using port 8000 first, or change the host port in Compose. From this project directory:

```bash
docker compose run --rm --build download-models
docker compose up -d face-verification
docker compose logs -f face-verification
```

The first command builds the application image and downloads the official LVFace-B checkpoint and SCRFD detector (approximately 451 MiB combined) into **`models/` in your cloned repository**, verifying both SHA256 hashes. Existing valid files are reused. The API mounts that folder at `/app/models` **read-only**; weights are not embedded in the image, and neither image building nor API startup downloads them. Rebuilds reuse the same host files. The base image supplies CUDA 12.8 and cuDNN 9; no host Python environment, Windows DLLs, or local photos are copied into the image. The API container runs as a non-root user. The one-off downloader runs as root to initialize the bind-mounted directory and exits when finished; it does not need GPU access.

- Interactive documentation: http://localhost:8000/docs
- Health: http://localhost:8000/health
- OpenAPI schema: http://localhost:8000/openapi.json

Check and stop:

```bash
docker compose ps
docker compose down
```

The default port mapping exposes the service only on the host's loopback interface. For integration into your main project's Compose network, add the service to that network; other containers can call `http://face-verification:8000/verify` without publishing a host port. A separate Compose project must join a shared network to use that hostname. Keep one worker per container initially.

Docker references: [GPU reservations in Compose](https://docs.docker.com/compose/how-tos/gpu-support/), [ONNX Runtime CUDA compatibility](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html).

### CPU-only execution

Use the standalone CPU configuration, which does not request an NVIDIA device:

```bash
docker compose -f compose.cpu.yaml run --rm --build download-models
docker compose -f compose.cpu.yaml up -d face-verification
```

It uses the same image and installed GPU libraries, but runs inference with `CPUExecutionProvider`. Do not combine this file with `compose.yaml`. Stop the GPU service first if it is already using port 8000.

### Build and share an image

```bash
docker build -t face-verification:1.0 .
docker run --rm --gpus all -p 127.0.0.1:8000:8000 --mount "type=bind,source=<ABSOLUTE_CLONE_PATH>/models,target=/app/models,readonly" face-verification:1.0
```

Replace `<ABSOLUTE_CLONE_PATH>` with the absolute path of the repository clone. Download models using Compose as above before this command.

For offline transfer after building:

```bash
docker save -o face-verification.tar face-verification:1.0
# On the recipient's machine:
docker load -i face-verification.tar
```

The image archive does not contain model weights. Recipients must also download models in their clone (or separately transfer the verified `models/` folder if permitted by the weight licenses) and mount that folder as shown above. The recipient still needs compatible GPU container support for GPU execution. Model-weight licensing applies to sharing and deployment; see below. Nothing is automatically published to a registry.

## API input

### `POST /verify`

Content type: **multipart/form-data**. Send image file contents, not JSON paths, URLs, or base64 strings.

| Parameter | Location | Required | Meaning |
| --- | --- | --- | --- |
| `reference` | File form field | Yes | Reference image |
| `candidate` | File form field | Yes | Image to compare against the reference |
| `threshold` | Query string | No | Finite value in [-1, 1]; default **0.4** |

Input guidance:

- JPEG and PNG are recommended; images must decode successfully through OpenCV.
- Each file must be at most **10 MiB**. This is an application check after upload parsing, not a limit on the entire HTTP request; configure an ingress request-body limit as appropriate.
- Each image must contain **exactly one detectable face**. Zero or multiple detected faces produce an error rather than selecting someone automatically.
- Face crops, head-and-shoulders pictures, upper-body pictures, and full-body pictures are accepted if the face is detectable. Manual cropping is optional; alignment is automatic.
- Prefer sharp, well-lit faces with visible eyes, nose, and mouth. Avoid heavy occlusion, extreme head rotation, or motion blur.
- A distant face may fail detection or produce an unreliable score because it contains little detail. Resizing a tiny face does not recover that detail. There is currently **no minimum face-size or blur gate**, so detection success does not establish adequate image quality.

Example with curl (use `curl.exe` in Windows PowerShell):

```bash
curl -X POST "http://localhost:8000/verify?threshold=0.4" -F "reference=@reference.jpg" -F "candidate=@candidate.jpg"
```

Python client:

```python
import requests

with open("reference.jpg", "rb") as reference, open("candidate.jpg", "rb") as candidate:
    response = requests.post(
        "http://localhost:8000/verify",
        files={"reference": reference, "candidate": candidate},
        params={"threshold": 0.4},
        timeout=120,
    )
response.raise_for_status()
print(response.json())
```

## API output

Successful response: **HTTP 200**, JSON. Example values:

```json
{
  "score": 0.82,
  "same_person": true,
  "threshold": 0.4,
  "threshold_calibrated": false,
  "model": "LVFace-B_Glint360K",
  "provider": "CUDAExecutionProvider"
}
```

| Field | Meaning |
| --- | --- |
| `score` | Cosine similarity in [-1, 1]; higher means more similar embeddings. **Not a probability or accuracy percentage.** |
| `same_person` | Boolean threshold decision: `score >= threshold`; not proof of identity |
| `threshold` | The threshold used for this request |
| `threshold_calibrated` | Always `false` in this implementation; calibration provenance is not tracked |
| `model` | Recognition checkpoint identifier |
| `provider` | Primary ONNX Runtime execution provider; CUDA for GPU mode, CPU for CPU mode |

**The default 0.4 threshold is an uncalibrated demonstration value.** Select a threshold on representative validation pairs for your desired false-accept rate, then evaluate on a separate test set. Passing a different threshold does not change `threshold_calibrated`.

Error responses:

| Status | Condition |
| --- | --- |
| 413 | An individual image exceeds 10 MiB |
| 422 | Missing file, invalid/empty image, zero or multiple detected faces, or invalid threshold |
| 500 | Unexpected inference/server failure |

Application validation errors return JSON such as:

```json
{"detail": "Expected exactly one face; detected 0."}
```

FastAPI request-validation errors (for example a missing field) use a structured `detail` list instead of a string. Invalid inputs do not return a fabricated similarity score.

### `GET /health`

Returns HTTP 200 once startup has loaded the models:

```json
{
  "status": "ok",
  "model": "LVFace-B_Glint360K",
  "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"]
}
```

Docker uses this endpoint for health checks. It checks service/model readiness; it does not run a verification on every health request. In CUDA mode, startup fails if CUDA cannot load instead of silently choosing CPU recognition.

## Local execution without Docker

The existing Windows environment is in `.venv`:

```powershell
.\.venv\Scripts\python.exe verify.py reference.jpg candidate.jpg
.\.venv\Scripts\python.exe verify.py reference.jpg candidate.jpg --device cpu
.\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
```

For a new Python 3.12 environment, install `requirements.txt`, then remove the CPU `onnxruntime` distribution pulled in by InsightFace and reinstall GPU ONNX Runtime to restore its shared files:

```bash
python -m venv .venv
# Activate .venv using your shell's activation command, then:
python -m pip install -r requirements.txt
python -m pip uninstall -y onnxruntime
python -m pip install --force-reinstall --no-deps onnxruntime-gpu==1.23.2
python download_models.py
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Set `FACE_DEVICE=cpu` to run the API on CPU. The existing Windows environment has local CUDA DLLs in `.venv/cuda`, copied from a cached PyTorch 2.11.0+cu128 distribution with its LICENSE and NOTICE retained. These DLLs are excluded from Docker; the Linux image uses its own runtime libraries. `requirements-lock.txt` is a historical Windows snapshot containing a local wheel URL; use `requirements-container.txt` for Docker.

## Validation and current limits

Local RTX 3090 CPU/GPU functionality checks passed: identical-image similarity 1.0000, brightness-adjusted same-image similarity about 0.9818, different-person sample pair about -0.0331. The five unit tests and API checks also passed. These samples are not an accuracy benchmark or calibration set.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe smoke_test.py
```

The smoke test requires the ignored public sample files in `samples/` from scikit-image (`astronaut.png`, v0.24.0) and Matplotlib (`grace_hopper.jpg`). They are excluded from the Docker image, along with all local personal photos.

**Docker build and container execution have not been tested on the authoring machine because Docker is not installed.** Validate the image on a Docker/NVIDIA host before integrating it into your main deployment.

There is no authentication, rate limiting, liveness/spoof detection, or automated threshold calibration. Uploaded images are not explicitly saved by application code; multipart parsing may spool them to temporary storage. Put access control and request limits at your main project's gateway. The service requires facial similarity only and cannot establish that a live person is present.

## Model licensing

The LVFace model card is tagged MIT, but the official repository README restricts downloaded models to **non-commercial research**. Treat the weights as research-only pending clarification from the authors. InsightFace pretrained SCRFD weights also have non-commercial research restrictions. Source-code licenses do not override model-weight terms. Do not assume this image is cleared for commercial deployment or unrestricted redistribution.
