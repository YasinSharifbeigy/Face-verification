# LVFace face-verification API

Compare two photographs to determine whether their faces belong to the same person. The API returns a cosine similarity score and a threshold-based match decision.

## Requirements

- Git, Docker, and Docker Compose v2.
- Linux containers on an x86-64 machine.
- For GPU inference: an NVIDIA GPU and driver compatible with CUDA 12.8. Linux hosts need NVIDIA Container Toolkit configured for Docker; Windows hosts need Docker Desktop with the WSL2 backend and working NVIDIA GPU support.
- Internet access for the initial build and model download. The first build downloads several gigabytes of Python and CUDA runtime dependencies.
- Port 8000 available, or a different host port configured in Compose.

Docker installs Python and its dependencies inside the image. You do not need a host Python environment or a separately installed CUDA toolkit for the Docker setup. CPU inference is also available.

## Quick start

### 1. Clone the repository

```bash
git clone https://github.com/YasinSharifbeigy/Face-verification.git
cd Face-verification
```

Run the following commands from this directory.

### 2. Build the image and download the models

```bash
docker compose run --rm --build download-models
```

This builds the application image, downloads LVFace-B and SCRFD, checks their SHA256 hashes, and exits. Model files are saved in **`models/` inside your clone**. Existing valid files are reused.

### 3. Start the API

```bash
docker compose up -d face-verification
docker compose logs -f face-verification
```

Wait for application startup to finish. Open:

- **http://localhost:8000/docs** ? upload images and try the API interactively.
- **http://localhost:8000/health** ? check model readiness and execution providers.
- **http://localhost:8000/openapi.json** ? API schema for integration.

In the interactive documentation, expand **POST /verify**, click **Try it out**, choose the `reference` and `candidate` image files, and click **Execute**.

### 4. Send a verification request

Use JPEG or PNG images containing exactly one detectable face each:

```bash
curl -X POST "http://localhost:8000/verify?threshold=0.4" -F "reference=@reference.jpg" -F "candidate=@candidate.jpg"
```

On Windows PowerShell, use `curl.exe` instead of `curl`. The image paths refer to files on the machine sending the request.

### Stop the service

```bash
docker compose down
```

Stopping the service does not delete the downloaded models.

## CPU-only setup

Use these commands instead of the GPU quick-start commands:

```bash
docker compose -f compose.cpu.yaml run --rm --build download-models
docker compose -f compose.cpu.yaml up -d face-verification
```

This configuration does not request GPU access. It uses the same image and downloads the same dependencies, but performs inference with `CPUExecutionProvider`. Do not combine the two Compose files or run both services on the same host port.

To stop CPU mode:

```bash
docker compose -f compose.cpu.yaml down
```

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

## GPU memory

Plan for **4 GiB of free VRAM per API worker** as an initial deployment budget, then validate under your intended workload. This is planning guidance, not a tested minimum.

A Windows RTX 3090 measurement showed approximately **1,301 MiB additional GPU memory** during model loading and five sequential verifications. It sampled total GPU usage while other workloads were present, so it is not an isolated process peak. Linux container usage, including the PyTorch runtime import, has not been measured. More workers create additional model sessions and may increase memory use.

The FP32 recognition checkpoint is approximately 434 MiB on disk; runtime memory is greater than the checkpoint size. A 24 GiB GPU is not required by this model.

## Where files are stored

```text
Face-verification/
  models/
    LVFace-B_Glint360K/
      LVFace-B_Glint360K.onnx
    det_10g.onnx
  Dockerfile
  compose.yaml
  compose.cpu.yaml
  download_models.py
```

- **Models:** approximately 451 MiB combined, downloaded into the clone's `models/` folder. The API mounts this directory at `/app/models` read-only. Weights are not included in Git or embedded in the image.
- **Python environment:** `/opt/venv` inside the Docker image. Docker does not create or use a host `.venv` folder.
- **Uploaded images:** not explicitly saved by the application. Multipart parsing may temporarily spool uploads to disk.

Rebuilding the image does not require downloading valid model files again. Neither image building nor API startup downloads model weights; the separate `download-models` command handles them.

## GPU runtime

The image uses `public.ecr.aws/docker/library/python:3.12-slim-bookworm` and installs PyTorch's CUDA 12.8 build:

```bash
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
```

PyTorch supplies the CUDA/cuDNN runtime libraries. The service imports it before creating ONNX Runtime's CUDA session. LVFace-B recognition still runs through ONNX Runtime using the ONNX checkpoint. The build does not use an NVIDIA CUDA container image or install the full CUDA compiler toolkit. The host still needs an NVIDIA driver and Docker GPU support.

References: [PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/), [ONNX Runtime CUDA integration](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#compatibility-with-pytorch).

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

## Integration with another application

The default host port binds to `127.0.0.1`, so the API is accessible only from the Docker host. Applications on the same Docker network can use `http://face-verification:8000/verify`. If your application is in a separate Compose project, attach both services to a shared network. Keep one API worker per container initially; verification requests are serialized to limit simultaneous model memory use.

The container runs as a non-root user. The one-off downloader runs as root to initialize the host-mounted models directory, then exits. It does not need GPU access.

Authentication, rate limiting, liveness detection, face-quality rejection, and automated threshold calibration are not implemented. Configure access control and request limits in your application's gateway before exposing the service to other networks. Facial similarity does not establish that a live person is present.

## Troubleshooting

### Image or package downloads fail

The build needs access to Amazon ECR Public, Debian repositories, PyPI, and `download.pytorch.org`. Model downloading also needs Hugging Face access. For registry or package errors, check the failing hostname and Docker Desktop's proxy configuration.

Test base-image access separately:

```bash
docker pull public.ecr.aws/docker/library/python:3.12-slim-bookworm
```

If your environment requires an approved Python 3.12 Debian Bookworm image mirror, set `PYTHON_IMAGE` before building. For example, in PowerShell, where Docker Hub is accessible:

```powershell
$env:PYTHON_IMAGE = "python:3.12-slim-bookworm"
docker compose run --rm --build download-models
```

Direct builds accept `docker build --build-arg PYTHON_IMAGE=<image-reference> .`.

### Models are missing

Run the `download-models` command from the quick start and confirm that both ONNX files exist in `models/`. The API does not download missing files automatically.

### CUDA cannot load

Confirm GPU access is enabled for Docker and the host NVIDIA driver supports the selected CUDA runtime. Check `docker compose logs face-verification`. GPU-mode startup fails if the CUDA provider cannot load; use the CPU configuration when GPU execution is unavailable.

### No face or multiple faces detected

Use an image with exactly one clearly visible face. Sharp, well-lit, closer photographs are more reliable than small or blurry distant faces. Check the input guidance above for details.

### Port 8000 is already in use

Stop the process using that port or change the host side of the port mapping, for example to `127.0.0.1:8001:8000`, then access the API on port 8001.

## Validation status

Local CPU/GPU functionality and API checks passed on an RTX 3090. These are functionality checks, not an accuracy benchmark or threshold calibration. The revised Linux Docker image has not yet been built or run in the validation environment; verify it on your deployment host before integration.

## Model licensing

The LVFace model card is tagged MIT, but the official repository README restricts downloaded models to **non-commercial research**. Treat the weights as research-only pending clarification from the authors. InsightFace pretrained SCRFD weights also have non-commercial research restrictions. Source-code licenses do not override model-weight terms; commercial deployment or redistribution requires appropriate permission.
