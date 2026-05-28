FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-deps chatterbox-tts

WORKDIR /
COPY requirements.txt /requirements.txt
RUN pip install -r requirements.txt

COPY rp_handler.py /

# Pre-pull model weights so cold start does not pay the HF download
RUN python -c "from chatterbox.tts import ChatterboxTTS; ChatterboxTTS.from_pretrained(device='cuda')"

CMD ["python3", "-u", "rp_handler.py"]
