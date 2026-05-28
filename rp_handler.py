"""
RunPod Serverless worker for Chatterbox TTS with flexible reference audio
input. Forked from geronimi73/runpod_chatterbox, extended with:
  - audio_base64 input (per-segment cloning)
  - audio_url input (S3 / presigned URL flow)
  - configurable yt_url duration limit
  - exposes Chatterbox generation params: exaggeration, cfg_weight,
    temperature, seed
  - returns audio_base64 + metadata; keeps API contract our Node adapter
    in src/tts/chatterbox.js already expects.
"""
import base64
import os
import random
import tempfile
import urllib.request
from pathlib import Path

import runpod
import torch
import torchaudio
import yt_dlp
# Prefer the multilingual variant (23 languages, accepts `language` kwarg).
# Fall back to the English-only ChatterboxTTS if the multilingual class is
# not available in the installed package version.
try:
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS as _ChatterboxClass
    MULTILINGUAL = True
except ImportError:
    try:
        from chatterbox.tts import ChatterboxMultilingualTTS as _ChatterboxClass
        MULTILINGUAL = True
    except ImportError:
        from chatterbox.tts import ChatterboxTTS as _ChatterboxClass
        MULTILINGUAL = False

MODEL = None
DEFAULT_REF_DURATION = 60  # seconds — cap reference clip to keep cold start tight


def initialize_model():
    global MODEL
    if MODEL is not None:
        return MODEL
    print(f"Loading {_ChatterboxClass.__name__} to CUDA (multilingual={MULTILINGUAL})...")
    MODEL = _ChatterboxClass.from_pretrained(device="cuda")
    print("Model ready")
    return MODEL


# ---------- audio source resolution ----------

def _save_base64(b64: str, out_path: str) -> str:
    """Decode base64 (with or without data: prefix) to a wav file."""
    if "," in b64 and b64.lstrip().startswith("data:"):
        b64 = b64.split(",", 1)[1]
    with open(out_path, "wb") as f:
        f.write(base64.b64decode(b64))
    return out_path


def _download_url(url: str, out_path: str) -> str:
    with urllib.request.urlopen(url, timeout=120) as r, open(out_path, "wb") as f:
        f.write(r.read())
    return out_path


def _download_youtube(url: str, out_dir: str, duration_limit: int) -> str:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": f"{out_dir}/ref.%(ext)s",
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
            "preferredquality": "192",
        }],
        "postprocessor_args": ["-ar", "44100"],
        "prefer_ffmpeg": True,
    }
    if duration_limit:
        ydl_opts["postprocessor_args"].extend(["-t", str(duration_limit)])
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return f"{out_dir}/ref.wav"


def resolve_reference_audio(inp: dict, work_dir: str) -> str:
    """Resolve one of audio_base64 / audio_url / yt_url to a local wav path."""
    if inp.get("audio_base64") or inp.get("reference_audio_base64"):
        b64 = inp.get("audio_base64") or inp.get("reference_audio_base64")
        return _save_base64(b64, os.path.join(work_dir, "ref.wav"))
    if inp.get("audio_url") or inp.get("voice_url"):
        url = inp.get("audio_url") or inp.get("voice_url")
        return _download_url(url, os.path.join(work_dir, "ref.wav"))
    if inp.get("yt_url"):
        duration = int(inp.get("yt_duration", DEFAULT_REF_DURATION))
        return _download_youtube(inp["yt_url"], work_dir, duration)
    raise ValueError(
        "Missing reference audio: provide one of "
        "'audio_base64' | 'audio_url' | 'yt_url' in input."
    )


# ---------- tensor → base64 ----------

def audio_tensor_to_base64(audio_tensor, sample_rate) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        torchaudio.save(tmp.name, audio_tensor, sample_rate)
        with open(tmp.name, "rb") as f:
            data = f.read()
        os.unlink(tmp.name)
    return base64.b64encode(data).decode("ascii")


# ---------- main handler ----------

def handler(event):
    inp = event.get("input", {}) or {}

    text = inp.get("text") or inp.get("prompt")
    if not text:
        return {"error": "Missing required field 'text' (or legacy 'prompt')."}

    # Reproducibility
    seed = inp.get("seed")
    if isinstance(seed, int):
        torch.manual_seed(seed)
        random.seed(seed)

    work_dir = tempfile.mkdtemp(prefix="cb_")
    ref_path = None
    try:
        ref_path = resolve_reference_audio(inp, work_dir)

        gen_kwargs = {"audio_prompt_path": ref_path}
        for k in ("exaggeration", "cfg_weight", "temperature"):
            if k in inp and inp[k] is not None:
                gen_kwargs[k] = inp[k]
        # `language` is only accepted by the multilingual class — pass
        # through only when we actually loaded it.
        if MULTILINGUAL and inp.get("language"):
            gen_kwargs["language_id"] = inp["language"]

        audio_tensor = MODEL.generate(text, **gen_kwargs)
        audio_b64 = audio_tensor_to_base64(audio_tensor, MODEL.sr)

        return {
            "audio_base64": audio_b64,
            "metadata": {
                "sample_rate": MODEL.sr,
                "audio_shape": list(audio_tensor.shape),
                "ref_source": (
                    "base64" if inp.get("audio_base64") or inp.get("reference_audio_base64")
                    else "url" if inp.get("audio_url") or inp.get("voice_url")
                    else "yt_url"
                ),
            },
        }
    except Exception as e:
        print(f"handler error: {e}")
        return {"error": str(e)}
    finally:
        if ref_path and os.path.exists(ref_path):
            try: os.remove(ref_path)
            except OSError: pass
        try:
            for f in os.listdir(work_dir):
                try: os.remove(os.path.join(work_dir, f))
                except OSError: pass
            os.rmdir(work_dir)
        except OSError:
            pass


if __name__ == "__main__":
    initialize_model()
    runpod.serverless.start({"handler": handler})
