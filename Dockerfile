# This Dockerfile is for Jetson Nano (GPU-enabled, Ubuntu 18.04)
FROM nvcr.io/nvidia/l4t-base:r32.7.1

# 1. Install micromamba binary & system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl bzip2 ca-certificates libgl1-mesa-glx libglib2.0-0 \
    && curl -Ls https://micro.mamba.pm/api/micromamba/linux-aarch64/latest | tar -xvj bin/micromamba \
    && mv bin/micromamba /usr/local/bin/micromamba \
    && rm -rf /var/lib/apt/lists/*

# 2. Set up environment
WORKDIR /app
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH=$MAMBA_ROOT_PREFIX/bin:$PATH

# 3. Use your updated cv2026-docker.yaml
COPY cv2026-docker.yaml /tmp/env.yaml
RUN micromamba install -y -n base -f /tmp/env.yaml && \
    micromamba clean --all --yes

# 4. App files (Note the renamed filename)
COPY 2_servingAPI.py /app/
RUN curl -L https://piclab.ai/classes/cv2023/chestxray.onnx -o /app/chestxray.onnx

# 5. Activate environment
ARG MAMBA_DOCKERFILE_ACTIVATE=1
ENV PATH=$MAMBA_ROOT_PREFIX/envs/base/bin:$PATH

EXPOSE 8500

# Start the application using the correct filename
CMD ["uvicorn", "2_servingAPI:app", "--host", "0.0.0.0", "--port", "8500"]
