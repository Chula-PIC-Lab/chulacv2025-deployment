# Computer Vision Deployment (Edge to Production)

This lab guides you through the process of taking a local computer vision script and turning it into a production-ready, containerized API for **Edge devices** (like the NVIDIA Jetson Nano) and **Production Linux environments**.

---

## Phase 1: Local Development (`1_objdet.py`)

The first step is to get your detection logic running locally on your hardware.

### 1. Install Mamba (Miniforge)

We use **Miniforge** (which includes `mamba`) because it is optimized for multiple architectures (x86_64, ARM64) and provides a fast, robust environment solver.

```bash
# Download and install Miniforge (Example for ARM64/Jetson)
wget https://github.com/conda-forge/miniforge/releases/download/26.1.1-3/Miniforge3-26.1.1-3-Linux-aarch64.sh
bash Miniforge3-26.1.1-3-Linux-aarch64.sh
# Follow the prompts and restart your terminal
```

### 2. Set Up the Environment

Create an isolated environment using the provided `cv2026.yaml` file:

```bash
mamba env create -f cv2026.yaml
mamba activate cv2026
```

### 3. Run the Detection Script

1. Download the demo model:
   ```bash
   wget https://piclab.ai/classes/cv2023/raccoons.onnx
   ```
2. Run the script:
   ```bash
   python 1_objdet.py
   ```

*Note: This script is configured to use **Hardware Acceleration** (like CUDA) if available, falling back to CPU if not.*

---

## Phase 2: Building the API (`2_servingAPI.py`)

In production, models are served as APIs so other applications (web, mobile, etc.) can access them remotely.

### 1. The Transition

We have converted the detection logic into a **FastAPI** service in `2_servingAPI.py`.

- **Script**: `1_objdet.py` -> Direct hardware/camera access + GUI window.
- **API**: `2_servingAPI.py` -> Accepts image bytes via POST request + Returns JSON results.

### 2. Test the API Locally

1. Download the classification model:
   ```bash
   wget https://piclab.ai/classes/cv2023/chestxray.onnx
   ```
2. Start the server:
   ```bash
   uvicorn 2_servingAPI:app --host 0.0.0.0 --port 8500
   ```
3. Test with `curl` from another terminal:
   ```bash
   curl -X POST -F "image=@your_image.jpg" http://localhost:8500/chestxray
   ```

---

## Phase 3: Containerization (`Dockerfile`)

Finally, we package the entire system (OS, drivers, environment, code) into a **Docker container** to ensure it runs identically across different hardware without manual setup.

### 1. Build and Run the Container

We use **Docker Compose** to manage hardware passthrough (like GPUs) and port mapping:

```bash
# Build and start the container in the background
docker-compose up --build -d
```

### 2. Hardware Monitoring

Check that the container is actually using the specialized hardware (GPU/NPU). On NVIDIA devices, use `jtop` or `nvidia-smi`:

```bash
jtop  # (On Jetson)
nvidia-smi # (On Desktop/Server)
```

---

## Technical Deep Dive: Why Mamba & Docker?

### 1. The "Software vs System" Gap

In **production**, `pip` is often insufficient. `pip` and `venv` isolate your Python packages, but they rely on your host's system libraries. To understand why this leads to failures across different Linux versions, you must understand these three key libraries:

- **`libc`**: The standard C library. The core interface between software and the Linux kernel.
- **`glibc` (GNU C Library)**: The most common implementation of `libc`. Software binaries are specifically "linked" to a version of `glibc`.
- **`libstdc++` (Standard C++ Library)**: The library that provides the core foundations of the C++ language (like `std::vector` and `std::string`) which high-performance vision libraries are built on.

**The "Standard One-Way" Characteristic of GLIBC Compatibility:**

Deployment failures often occur due to how **`glibc` versioning works**:

1. **Backward Compatible (YES)**: Software built on an older `glibc` (e.g., 2.17) will run on a newer system.
2. **Forward Compatible (NO)**: Software built on a newer `glibc` (e.g., 2.39) will **NOT** run on an older system. It will crash immediately with a `GLIBC not found` error.

**Understanding the `manylinux` Standard:**

When you run `pip install`, it often downloads a **"wheel"** with a tag like `manylinux_2_28`. This is a professional standard that maps to a specific `glibc` version:

- **`manylinux2014`**: Built on CentOS 7 (`glibc` 2.17). This was the standard for a decade but is being **phased out** since CentOS 7 reached EOL in June 2024.
- **`manylinux_2_28`**: The **current industry baseline** (built on AlmaLinux 8). Many major libraries (like PyTorch) switched to this as their default in late 2024.
- **`manylinux_2_31`**: Built on Ubuntu 20.04. Very common for ARM and modern Ubuntu users, but will **NOT** run on older JetPack installations (Ubuntu 18.04).
- **`manylinux_2_34` / `manylinux_2_39`**: Upcoming "Alpha" standards for cutting-edge distributions (Ubuntu 22.04+).

This is why `pip` is risky for edge devices: it may pull a `manylinux` version that is too new for your hardware's operating system.

**Comparative GLIBC Versions Across Distributions:**

| OS Version                           | GLIBC Version  | Status                  |
| :----------------------------------- | :------------- | :---------------------- |
| **CentOS 7**                   | 2.17           | Extreme Legacy          |
| **Ubuntu 18.04 (Jetson Nano)** | **2.27** | Common Edge Basis       |
| **Ubuntu 20.04**               | 2.31           | Wide Production Basis   |
| **RHEL 8 / CentOS 8**          | 2.28           | Enterprise Standard     |
| **RHEL 9**                     | 2.34           | Modern Enterprise       |
| **Ubuntu 22.04**               | 2.35           | Current Modern Standard |
| **Ubuntu 24.04**               | 2.39           | Cutting Edge            |

**Why Mamba (Conda) wins on specialized hardware:**

- **Escaping the "Version Trap"**: `pip` wheels are often built on the latest Ubuntu versions. If you try to run them on older, stable hardware (like a Jetson with Ubuntu 18.04), they will fail because they expect a newer `glibc`. **Conda packages include their own compatible runtime libraries**, ensuring your code runs anywhere.
- **System Binaries**: OpenCV and ONNX Runtime require specific versions of `libvision`, `libjpeg`, and `libprotobuf`. `pip` assumes these are already installed on your Jetson. **Conda installs these entire binaries directly into your environment**, rather than assuming they are on the host.
- **Isolating Python Itself**: Unlike `venv`, Conda installs the **Python interpreter binary** itself. This allows you to have a Python 3.9 environment running perfectly on a system that only came with Python 3.6, without touching the OS.

### 2. Detailed Dockerfile Breakdown

- **`FROM`**: We start with a base image that matches our hardware (e.g., `nvcr.io/nvidia/l4t-base` for Jetson or `nvidia/cuda` for desktop).
- **`micromamba`**: A tiny C++ "engine" for Conda that is much faster and lighter for edge devices. It is the professional choice for lean, stable, and reproducible production images.
- **`2_servingAPI.py`**: Our production script that exposes the model as a web service.
- **`MAMBA_DOCKERFILE_ACTIVATE=1`**: An automated hook that ensures our API runs inside the correct environment without manual activation.

---

## Key Deployment Concepts

- **Deterministic Reproducibility**: Guaranteeing the same build every time.
- **Hardware Abstraction**: Using Docker to bridge the gap between code and specialized chips (GPU/NPU).
- **API Serving**: Exposing deep learning models via FastAPI for global access.
