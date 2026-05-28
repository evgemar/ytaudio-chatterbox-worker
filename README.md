# runpod-chatterbox-worker

RunPod Serverless worker for **Chatterbox TTS** with flexible reference
audio input. Forked from
[`geronimi73/runpod_chatterbox`](https://github.com/geronimi73/runpod_chatterbox)
and extended for per-segment voice cloning.

## What's different from the upstream

- Accepts reference audio in three forms — pick whichever fits your client:
  - `audio_base64` (or `reference_audio_base64`) — what `src/tts/chatterbox.js` in this repo sends
  - `audio_url` (or `voice_url`) — fetched server-side, ideal with S3 / MinIO presigned URLs
  - `yt_url` — original behaviour, configurable `yt_duration`
- `text` field replaces (and aliases) `prompt`.
- Passes through Chatterbox knobs: `exaggeration`, `cfg_weight`, `temperature`, `language`, `seed`.
- Bakes the model weights into the image at build time (`Dockerfile` runs a no-op `from_pretrained` so the first job does not pay the HF download).

## Input schema

```json
{
  "input": {
    "text": "Hello world",                    // required (or "prompt")
    "audio_base64": "<base64 wav>",           // one of these three:
    "audio_url":    "https://...wav",
    "yt_url":       "https://youtube.com/...",
    "yt_duration":  60,                       // optional, only with yt_url
    "exaggeration": 1.0,                      // optional
    "cfg_weight":   0.5,
    "temperature":  0.7,
    "language":     "en",
    "seed":         42
  }
}
```

## Output schema

```json
{
  "audio_base64": "<base64 wav>",
  "metadata": {
    "sample_rate": 24000,
    "audio_shape": [1, 95232],
    "ref_source": "base64" | "url" | "yt_url"
  }
}
```

## Deploying to RunPod Serverless

1. Push this directory to a GitHub repo (public or private — RunPod can read both with a connected account).
2. RunPod Console → Serverless → New Endpoint → "From GitHub Repo".
3. Pick GPU class **24 GB** (RTX 4090 / A5000). Chatterbox fits comfortably.
4. Container disk: **20 GB** (model weights are baked in but PyTorch CUDA libs are heavy).
5. Idle timeout: **60 s**. Max workers: tune for your traffic.
6. Once deployed, the endpoint ID lands at `https://api.runpod.ai/v2/<endpoint-id>`.

## Wiring back into ytaudio

In `/Users/evgemar/www/ytaudio/.env`:

```
RUNPOD_API_KEY=...
CHATTERBOX_BASE_URL=https://api.runpod.ai/v2/<your-endpoint-id>
TTS_ROUTES={"*":"chatterbox"}
```

Then `docker compose restart worker media-worker`. The Node adapter at
`src/tts/chatterbox.js` already sends `reference_audio_base64` and
expects `audio_base64` back, which matches this handler.

## Test it

```bash
curl -X POST https://api.runpod.ai/v2/<endpoint-id>/runsync \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "text": "Testing my forked Chatterbox worker.",
      "yt_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
      "yt_duration": 30
    }
  }'
```

Expect a job id; poll `/status/<job-id>` until COMPLETED. Output is a
base64 WAV.
