"""
Demucs source-separation helpers for the HeartMuLa_Demucs ComfyUI node.

Models are cached in memory after first load.  The cache is keyed on
(model_name, device) so switching devices always reloads cleanly.
"""

import logging
from typing import Dict

import torch
import torchaudio

logger = logging.getLogger("HeartMuLa")

# ── model cache ───────────────────────────────────────────────────────────────
_MODEL_CACHE: Dict[str, object] = {}

_STANDARD_STEMS = ("vocals", "drums", "bass", "other")


def _load_model(model_name: str, device: torch.device):
    """Return a cached (or freshly loaded) Demucs model on *device*."""
    cache_key = f"{model_name}|{device}"
    if cache_key not in _MODEL_CACHE:
        from demucs.pretrained import get_model  # demucs handles HF download
        logger.info("HeartMuLa Demucs: loading model '%s' …", model_name)
        model = get_model(model_name)
        model.eval()
        model.to(device)
        _MODEL_CACHE[cache_key] = model
        logger.info("HeartMuLa Demucs: model ready on %s", device)
    return _MODEL_CACHE[cache_key]


def _unload_model(model_name: str, device: torch.device) -> None:
    key = f"{model_name}|{device}"
    _MODEL_CACHE.pop(key, None)


# ── main separation function ──────────────────────────────────────────────────

def separate_stems(
    audio_input,
    model_name:        str,
    device:            torch.device,
    shifts:            int   = 1,
    overlap:           float = 0.25,
    keep_model_loaded: bool  = True,
) -> Dict[str, dict]:
    """Separate *audio_input* (ComfyUI AUDIO dict) into stems.

    Returns a dict keyed by stem name, each value being a ComfyUI AUDIO dict
    ``{"waveform": (1, C, S), "sample_rate": int}``.
    """
    from demucs.apply import apply_model

    # ── unpack AUDIO / LazyAudioMap / legacy tuple ────────────────────────────
    if isinstance(audio_input, str):
        waveform, sr = torchaudio.load(audio_input)
    elif isinstance(audio_input, (list, tuple)):
        sr, waveform = audio_input
        if isinstance(waveform, str):
            waveform, sr = torchaudio.load(waveform)
    elif hasattr(audio_input, "__getitem__"):
        waveform = audio_input["waveform"]
        sr       = audio_input["sample_rate"]
    else:
        raise ValueError(f"HeartMuLa Demucs: unsupported audio type {type(audio_input)}")

    # waveform may be (B, C, S) or (C, S)
    if isinstance(waveform, torch.Tensor):
        wav = waveform[0] if waveform.ndim == 3 else waveform
    else:
        import numpy as np
        wav = torch.from_numpy(waveform)
        if wav.ndim == 3:
            wav = wav[0]

    wav = wav.float()

    # ── ensure stereo ─────────────────────────────────────────────────────────
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    elif wav.shape[0] > 2:
        wav = wav[:2]

    # ── load model & resample ─────────────────────────────────────────────────
    model    = _load_model(model_name, device)
    model_sr = model.samplerate

    if sr != model_sr:
        logger.info(
            "HeartMuLa Demucs: resampling %d → %d Hz", sr, model_sr
        )
        wav = torchaudio.functional.resample(wav, sr, model_sr)

    # ── separate ──────────────────────────────────────────────────────────────
    logger.info(
        "HeartMuLa Demucs: separating with shifts=%d overlap=%.2f",
        shifts, overlap,
    )
    with torch.no_grad():
        # apply_model expects (batch, channels, samples)
        sources = apply_model(
            model,
            wav.unsqueeze(0).to(device),
            shifts=shifts,
            overlap=overlap,
            progress=True,
            device=device,
        )
    # sources: (1, num_stems, C, S)
    sources = sources[0].cpu()  # (num_stems, C, S)

    # ── build result dict ─────────────────────────────────────────────────────
    result: Dict[str, dict] = {}
    for i, stem_name in enumerate(model.sources):
        result[stem_name] = {
            "waveform":    sources[i].unsqueeze(0),   # (1, C, S)
            "sample_rate": model_sr,
        }

    # Guarantee all four standard stems exist (some models use different names)
    ref_shape = sources[0].unsqueeze(0).shape
    for stem in _STANDARD_STEMS:
        if stem not in result:
            result[stem] = {
                "waveform":    torch.zeros(ref_shape),
                "sample_rate": model_sr,
            }

    if not keep_model_loaded:
        _unload_model(model_name, device)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return result
