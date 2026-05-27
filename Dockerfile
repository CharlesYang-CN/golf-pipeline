FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3-pip \
    wget curl git build-essential \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libgomp1 libegl1-mesa libgl1 \
    ffmpeg \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu124

RUN pip install --no-cache-dir \
    mediapipe \
    opencv-python \
    numpy==1.26.4 \
    scipy \
    cython \
    lapx \
    wget \
    ultralytics \
    runpod

COPY scripts/ /app/scripts/
COPY handler.py /app/

COPY angle/ /app/angle/
COPY swing/ /app/swing/
COPY romp/ /app/romp/
COPY romp_models/ /root/.romp/

RUN pip install --no-cache-dir --no-build-isolation -e /app/romp/

RUN mkdir -p /data

WORKDIR /app
ENTRYPOINT ["python", "/app/handler.py"]
