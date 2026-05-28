"""Pre-download model weights to a cached Docker layer.
Prefer the multilingual class, fall back to base ChatterboxTTS."""
try:
    try:
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS as Cls
    except ImportError:
        from chatterbox.tts import ChatterboxMultilingualTTS as Cls
    print(f"Prefetching {Cls.__name__} (multilingual)")
    Cls.from_pretrained(device="cpu")
except ImportError:
    from chatterbox.tts import ChatterboxTTS as Cls
    print(f"Prefetching {Cls.__name__} (base)")
    Cls.from_pretrained(device="cpu")
print("Prefetch complete")
