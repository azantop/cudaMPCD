FROM nvidia/cuda:12.3.2-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV UV_LINK_MODE=copy

EXPOSE 8888

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libhdf5-dev \
    libhdf5-serial-dev \
    python3 \
    python3-dev \
    python3-pip \
    libgtest-dev \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir uv
RUN echo "alias jl='jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root'" >> /root/.bashrc
# then at runtime: uv pip install --system .
# or: uv sync --no-venv  (depending on uv version)
# or: pip3 install .
