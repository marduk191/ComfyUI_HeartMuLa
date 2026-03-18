"""MilkDrop-style audio visualizer — pure numpy, no extra dependencies.

Speed strategy
--------------
* Batch FFT: all per-frame frequency analysis done in one np.fft.rfft call.
* Pre-computed grids: X/Y/R/THETA meshgrids built once.
* Per-frame effect: fully vectorised numpy — no Python inner loops.
* Optional render_scale: compute at 1/N resolution, upsample with PIL.
"""

import numpy as np
from PIL import Image, ImageDraw
from .audio_visualizer import _get_font, _text_width, _draw_shadowed_text, _lyric_index


# ── Audio analysis (batch) ────────────────────────────────────────────────────

def _analyse_audio(audio_np: np.ndarray, sample_rate: int,
                   n_frames: int, fps: int) -> dict:
    """Return per-frame bass/mid/treble/beat arrays, all in [0, 1].

    Uses a single batched rfft call — no Python loop over frames.
    """
    hop = max(1, sample_rate // fps)
    win = 2048
    window = np.hanning(win).astype(np.float32)

    # Build strided view: (n_frames, win)
    padded = np.pad(audio_np.astype(np.float32), win // 2, mode="reflect")
    max_start = len(padded) - win
    starts = np.arange(n_frames) * hop
    starts = np.clip(starts, 0, max_start)

    # Materialise all frames at once
    idx = starts[:, None] + np.arange(win)[None, :]          # (N, win)
    frames_2d = padded[idx] * window[None, :]                 # (N, win)

    # Batch FFT
    mag = np.abs(np.fft.rfft(frames_2d, axis=1)).astype(np.float32)  # (N, win//2+1)

    n_bins = mag.shape[1]
    b1 = max(1, int(n_bins * 0.06))   # bass:   0 – 6 %
    b2 = max(b1 + 1, int(n_bins * 0.35))  # mid: 6 – 35 %

    total = mag.sum(axis=1, keepdims=True) + 1e-8
    bass_arr   = mag[:, :b1].sum(axis=1) / total[:, 0]
    mid_arr    = mag[:, b1:b2].sum(axis=1) / total[:, 0]
    treble_arr = mag[:, b2:].sum(axis=1) / total[:, 0]

    for arr in (bass_arr, mid_arr, treble_arr):
        pk = arr.max()
        if pk > 1e-8:
            arr /= pk

    beat_arr = np.maximum(0.0, np.diff(bass_arr, prepend=bass_arr[0]))
    pk = beat_arr.max()
    if pk > 1e-8:
        beat_arr /= pk

    return {"bass": bass_arr, "mid": mid_arr,
            "treble": treble_arr, "beat": beat_arr}


# ── Colour maps ────────────────────────────────────────────────────────────────

def _hsv_hue_to_rgb(h: np.ndarray) -> np.ndarray:
    """Map hue array [0,1] → RGB float32 (H,W,3).  Vectorised HSV s=v=1."""
    h6 = h * 6.0
    r = np.clip(np.abs(h6 - 3.0) - 1.0, 0.0, 1.0)
    g = np.clip(2.0 - np.abs(h6 - 2.0), 0.0, 1.0)
    b = np.clip(2.0 - np.abs(h6 - 4.0), 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)


def _cmap_rainbow(v, hue_off):
    return _hsv_hue_to_rgb((v + hue_off) % 1.0)


def _cmap_fire(v, _):
    r = np.clip(v * 3.0,        0.0, 1.0)
    g = np.clip(v * 3.0 - 1.0, 0.0, 1.0)
    b = np.clip(v * 3.0 - 2.0, 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)


def _cmap_ice(v, hue_off):
    r = np.clip(v * 2.0 - 1.0, 0.0, 1.0)
    g = np.clip(v * 1.5 - 0.3, 0.0, 1.0) * 0.7
    b = np.clip(v * 0.8 + 0.2, 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)


def _cmap_neon(v, hue_off):
    h = (v * 2.0 + hue_off) % 1.0
    tau = np.pi * 2.0
    r = np.clip(np.sin(h * tau + 0.0) * 0.5 + 0.3, 0.0, 1.0) * v
    g = np.clip(np.sin(h * tau + 2.0) * 0.5 + 0.4, 0.0, 1.0) * v
    b = np.clip(np.sin(h * tau + 4.0) * 0.5 + 0.5, 0.0, 1.0) * v
    return np.stack([r, g, b], axis=-1)


_CMAPS = {
    "rainbow": _cmap_rainbow,
    "fire":    _cmap_fire,
    "ice":     _cmap_ice,
    "neon":    _cmap_neon,
}


# ── Effects ────────────────────────────────────────────────────────────────────

def _plasma(X, Y, R, THETA, t, bass, mid, treble, beat, intensity):
    v = (
        np.sin(X * (3.0 + bass   * 2.0 * intensity) + t * 1.20) * 0.25 +
        np.sin(Y * (4.0 + mid    * 2.0 * intensity) + t * 0.80) * 0.25 +
        np.sin(R * (5.0 + treble * 3.0 * intensity) - t * 1.50) * 0.20 +
        np.sin((X + Y) * (3.5 + bass * intensity)   + t * 0.60) * 0.15 +
        np.sin((X - Y) *  2.5                        + t * 1.10) * 0.15
    )
    return np.clip((v + 1.0) * 0.5 + beat * 0.2, 0.0, 1.0)


def _tunnel(X, Y, R, THETA, t, bass, mid, treble, beat, intensity):
    r_mod = R + bass * 0.15 * intensity * np.sin(THETA * 4.0 + t)
    v = np.sin(8.0 / (r_mod + 0.05) - t * (2.0 + mid * intensity) + THETA * 3.0)
    v = (v + 1.0) * 0.5 * np.clip(1.0 - R * 0.8, 0.0, 1.0)
    return np.clip(v + beat * 0.3, 0.0, 1.0)


def _kaleidoscope(X, Y, R, THETA, t, bass, mid, treble, beat, intensity):
    seg_w   = np.pi / 6.0
    theta_k = THETA % (2.0 * seg_w)
    theta_k = np.where(theta_k > seg_w, 2.0 * seg_w - theta_k, theta_k)
    r_pulse = R * (1.0 + bass * 0.2 * intensity * np.sin(t * 2.0))
    kx = r_pulse * np.cos(theta_k + t * 0.3)
    ky = r_pulse * np.sin(theta_k + t * 0.3)
    v = (
        np.sin(kx * (4.0 + mid    * 2.0 * intensity) + t      ) * 0.35 +
        np.sin(ky * (4.0 + treble * 2.0 * intensity) + t * 1.3) * 0.35 +
        np.sin(r_pulse * 5.0                          - t * 1.5) * 0.30
    )
    return np.clip((v + 1.0) * 0.5 + beat * 0.3, 0.0, 1.0)


def _combined(X, Y, R, THETA, t, bass, mid, treble, beat, intensity):
    p = _plasma(X, Y, R, THETA, t, bass, mid, treble, beat, intensity)
    k = _kaleidoscope(X, Y, R, THETA, t, bass, mid, treble, beat, intensity)
    w = 0.3 + bass * 0.2 * intensity
    return np.clip(p * (1.0 - w) + k * w, 0.0, 1.0)


_EFFECTS = {
    "combined":      _combined,
    "plasma":        _plasma,
    "tunnel":        _tunnel,
    "kaleidoscope":  _kaleidoscope,
}


# ── Main render function ───────────────────────────────────────────────────────

def render_milkdrop_frames(
    audio_np:     np.ndarray,
    sample_rate:  int,
    fps:          int,
    width:        int,
    height:       int,
    effect:       str,
    color_scheme: str,
    intensity:    float,
    render_scale: int = 1,
    lyric_lines:  list = [],
) -> np.ndarray:
    """Return (N, H, W, 3) float32 [0,1] tensor of animation frames."""
    duration = len(audio_np) / max(sample_rate, 1)
    n_frames = max(1, int(duration * fps))

    # Compute at reduced resolution, upsample at the end
    rw = max(32, width  // render_scale)
    rh = max(32, height // render_scale)

    # Pre-compute coordinate grids (float32 saves memory and speeds math)
    xs = ((np.arange(rw, dtype=np.float32) - rw / 2) / (rw  / 2))
    ys = ((np.arange(rh, dtype=np.float32) - rh / 2) / (rh / 2))
    X, Y  = np.meshgrid(xs, ys)
    R     = np.sqrt(X * X + Y * Y)
    THETA = np.arctan2(Y, X)

    # Batch audio analysis
    analysis   = _analyse_audio(audio_np, sample_rate, n_frames, fps)
    bass_arr   = analysis["bass"]
    mid_arr    = analysis["mid"]
    treble_arr = analysis["treble"]
    beat_arr   = analysis["beat"]

    effect_fn = _EFFECTS.get(effect, _combined)
    cmap      = _CMAPS.get(color_scheme, _cmap_rainbow)

    need_upsample = (render_scale > 1)
    has_lyrics    = bool(lyric_lines)
    duration      = len(audio_np) / max(sample_rate, 1)

    # Pre-load fonts once if lyrics are present
    if has_lyrics:
        main_sz = max(18, int(height * 0.054))
        dim_sz  = max(13, int(height * 0.034))
        font_m  = _get_font(main_sz, bold=True)
        font_d  = _get_font(dim_sz,  bold=False)

    frames_out = []

    for fi in range(n_frames):
        t = fi / fps

        v = effect_fn(
            X, Y, R, THETA, t,
            float(bass_arr[fi]),
            float(mid_arr[fi]),
            float(treble_arr[fi]),
            float(beat_arr[fi]),
            intensity,
        )

        hue_off = (t * 0.07 + float(bass_arr[fi]) * 0.1) % 1.0
        rgb = cmap(v, hue_off)   # (rh, rw, 3) float32

        if need_upsample:
            img = Image.fromarray((rgb * 255.0).astype(np.uint8))
            img = img.resize((width, height), Image.BILINEAR)
            rgb = np.asarray(img, dtype=np.float32) * (1.0 / 255.0)

        if has_lyrics:
            img  = Image.fromarray((rgb * 255.0).astype(np.uint8))
            draw = ImageDraw.Draw(img)
            n    = len(lyric_lines)
            ci   = _lyric_index(t, duration, n)
            cy   = height * 4 // 5   # lyric band near bottom

            for offset, is_main in ((-1, False), (0, True), (1, False)):
                idx = ci + offset
                if not (0 <= idx < n):
                    continue
                text  = lyric_lines[idx]
                fnt   = font_m if is_main else font_d
                color = (255, 255, 255) if is_main else (180, 180, 180)
                tw    = _text_width(fnt, text)
                tx    = (width - tw) // 2
                sz    = main_sz if is_main else dim_sz
                if offset == -1:
                    ty = cy - main_sz - dim_sz
                elif offset == 0:
                    ty = cy - main_sz // 2
                else:
                    ty = cy + main_sz // 2 + 4
                _draw_shadowed_text(draw, (tx, ty), text, fnt, color, (0, 0, 0))

            rgb = np.asarray(img, dtype=np.float32) * (1.0 / 255.0)

        frames_out.append(rgb)

    return np.stack(frames_out, axis=0)   # (N, H, W, 3)
