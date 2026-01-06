# Backend API

## Prerequisites

- Python 3.12 or 3.13 (Python 3.14 is not supported by pydantic-core yet)

## Setup

From the repo root:

```bash
cd backend
conda create -n mdoc_test python=3.12 -y
conda activate mdoc_test
pip install -r requirements.txt
```

If you already have a conda env, activate it instead of creating a new one.



Optional: allow the frontend origin for CORS:

```bash
export ALLOWED_ORIGINS=http://localhost:3000
```

## Run (dev)

```bash
cd backend
python -m uvicorn app:app --reload --port 8000
```

From the repo root, you can also run:

```bash
python -m uvicorn backend.app:app --reload --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
```

## API endpoints

- `GET /health`
- `GET /uploads` (list uploaded PDFs and statuses)
- `POST /uploads` (multipart file upload, PDF only; triggers extract + index/embedding)
- `POST /qa` (requires `upload_id` and `question`)

## Troubleshooting

- **Form data requires "python-multipart"**: make sure you run with the conda env's Python.
  - Use `python -m uvicorn ...` or `conda run -n mdoc_test python -m uvicorn ...` to avoid picking the base environment.
- **faiss-gpu installation**: pip does not provide `faiss-gpu`.
  - Install via conda: `conda install -n mdoc_test -c pytorch -c nvidia -c conda-forge faiss-gpu`
- **CUDA "no kernel image" / unsupported arch**:
  - Ensure nvcc matches your GPU. For RTX 5070 (sm_120), use CUDA 12.8+.
  - Clear extension cache before rebuild:
    - `rm -rf ~/.cache/torch_extensions/py312_cu128/decompress_residuals_cpp ~/.cache/torch_extensions/py312_cu128/packbits_cpp`
  - Compile with correct arch: `export TORCH_CUDA_ARCH_LIST="12.0"`.
