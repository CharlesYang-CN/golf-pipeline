FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    wget curl git build-essential \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libgomp1 libegl1-mesa libgl1 \
    ffmpeg \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu124

RUN pip install --no-cache-dir \
    mediapipe opencv-python numpy==1.26.4 \
    scipy cython lapx wget ultralytics runpod

COPY scripts/ /app/scripts/
COPY handler.py /app/
COPY angle/ /app/angle/
COPY swing/ /app/swing/
COPY romp/ /app/romp/
COPY romp_models/ /root/.romp/

RUN pip install --no-cache-dir --no-build-isolation -e /app/romp/

RUN mkdir -p /data && python -c "import runpod; print('runpod OK')"

WORKDIR /app
ENTRYPOINT ["python", "-u", "/app/handler.py"]
