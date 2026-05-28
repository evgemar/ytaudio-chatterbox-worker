FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-deps chatterbox-tts

WORKDIR /
COPY requirements.txt /requirements.txt
RUN pip install -r requirements.txt

COPY rp_handler.py /

# Pre-pull model weights so cold start does not pay the HF download.
# Build runners on RunPod are CPU-only, so we load on CPU here — at
# request time the handler loads to CUDA from cached weights.
# Prefetch multilingual first (preferred), fall back to base TTS.
RUN python -c "\
try:\n\
    try: from chatterbox.mtl_tts import ChatterboxMultilingualTTS as C\n\
    except ImportError: from chatterbox.tts import ChatterboxMultilingualTTS as C\n\
    print('Prefetching multilingual'); C.from_pretrained(device='cpu')\n\
except ImportError:\n\
    from chatterbox.tts import ChatterboxTTS as C\n\
    print('Prefetching base TTS'); C.from_pretrained(device='cpu')\n\
"

CMD ["python3", "-u", "rp_handler.py"]
