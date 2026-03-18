"""Audio visualization utilities for HeartMuLa nodes.

Renders animated frames showing waveform and/or spectrogram with an optional
lyric subtitle overlay.  Uses only numpy and Pillow — no extra dependencies.

Performance strategy
--------------------
* Spectrogram: the frequency axis is resized and the entire result is colorized
  *once* during pre-computation.  Per-frame work is a single 1-D time-axis
  interpolation instead of a full 2-D bilinear resample.
* Waveform: polygon fill is done with a numpy broadcast mask instead of a
  Python loop over every pixel column.
* The numpy canvas is constructed directly; PIL is only used for text drawing.
"""

import os
import logging
import numpy as np
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger("HeartMuLa")

# ── Color schemes ──────────────────────────────────────────────────────────────

_COLOR_SCHEMES = {
    "dark": {
        "bg":           np.array([10,  12,  24],  dtype=np.uint8),
        "wave_fill":    np.array([20,  60, 110],  dtype=np.uint8),
        "wave_line":    np.array([80, 180, 255],  dtype=np.uint8),
        "playhead":     np.array([255,  80,  80], dtype=np.uint8),
        "lyric_main":   (255, 255, 255),
        "lyric_dim":    (130, 130, 150),
        "lyric_shadow": (0,   0,   0),
        "spec_lo":      np.array([10,  12,  24],  dtype=np.float32),
        "spec_mid":     np.array([30, 100, 200],  dtype=np.float32),
        "spec_hi":      np.array([200, 230, 255], dtype=np.float32),
        "bar":          (80, 180, 255),
        "separator":    (40,  60, 100),
    },
    "neon": {
        "bg":           np.array([0,   0,   0],   dtype=np.uint8),
        "wave_fill":    np.array([0,   55,  28],  dtype=np.uint8),
        "wave_line":    np.array([0,  255, 128],  dtype=np.uint8),
        "playhead":     np.array([255,  0, 255],  dtype=np.uint8),
        "lyric_main":   (255, 255,  0),
        "lyric_dim":    (140, 140,  0),
        "lyric_shadow": (80,   0,  80),
        "spec_lo":      np.array([0,   0,   0],   dtype=np.float32),
        "spec_mid":     np.array([0,  170,  70],  dtype=np.float32),
        "spec_hi":      np.array([220, 255,  0],  dtype=np.float32),
        "bar":          (0,  255, 128),
        "separator":    (0,   80,  40),
    },
    "light": {
        "bg":           np.array([245, 245, 252], dtype=np.uint8),
        "wave_fill":    np.array([180, 205, 245], dtype=np.uint8),
        "wave_line":    np.array([40,   90, 200], dtype=np.uint8),
        "playhead":     np.array([200,  50,  50], dtype=np.uint8),
        "lyric_main":   (20,  20,  50),
        "lyric_dim":    (130, 130, 160),
        "lyric_shadow": (210, 210, 225),
        "spec_lo":      np.array([245, 245, 252], dtype=np.float32),
        "spec_mid":     np.array([100, 150, 220], dtype=np.float32),
        "spec_hi":      np.array([20,   60, 180], dtype=np.float32),
        "bar":          (40,  90, 200),
        "separator":    (180, 190, 220),
    },
}


# ── Font helpers ───────────────────────────────────────────────────────────────

def _get_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates: list[str] = []
    if os.name == "nt":
        windir = os.environ.get("WINDIR", "C:/Windows")
        fonts  = os.path.join(windir, "Fonts")
        candidates = [
            os.path.join(fonts, "arialbd.ttf"  if bold else "arial.ttf"),
            os.path.join(fonts, "seguisb.ttf"  if bold else "segoeui.ttf"),
            os.path.join(fonts, "calibrib.ttf" if bold else "calibri.ttf"),
            os.path.join(fonts, "verdanab.ttf" if bold else "verdana.ttf"),
        ]
    else:
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
                else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold
                else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _text_width(font: ImageFont.ImageFont, text: str) -> int:
    try:
        b = font.getbbox(text)
        return b[2] - b[0]
    except AttributeError:
        return font.getsize(text)[0]


def _draw_shadowed_text(draw, xy, text, font, color, shadow_color, offset=2):
    draw.text((xy[0] + offset, xy[1] + offset), text, font=font, fill=shadow_color)
    draw.text(xy, text, font=font, fill=color)


# ── Spectrogram helpers ────────────────────────────────────────────────────────

def _compute_log_spectrogram(mono: np.ndarray, n_fft: int = 1024,
                              hop: int = 256) -> np.ndarray:
    """Return log-magnitude spectrogram (n_frames, n_bins) via numpy FFT."""
    window  = np.hanning(n_fft).astype(np.float32)
    padded  = np.pad(mono.astype(np.float32), n_fft // 2, mode="reflect")
    n_fr    = 1 + (len(padded) - n_fft) // hop
    shape   = (n_fr, n_fft)
    strides = (padded.strides[0] * hop, padded.strides[0])
    frames  = np.lib.stride_tricks.as_strided(padded, shape=shape,
                                               strides=strides).copy()
    frames *= window
    mag     = np.abs(np.fft.rfft(frames, axis=1))   # (n_fr, n_fft//2+1)
    log_mag = np.log1p(mag)
    peak    = log_mag.max()
    if peak > 1e-8:
        log_mag /= peak
    return log_mag.astype(np.float32)


def _precompute_spectrogram(log_mag: np.ndarray, target_h: int,
                             cs: dict) -> np.ndarray:
    """Resize freq axis and colorise the spectrogram once.

    Returns an array of shape (target_h, T, 3) uint8 with the frequency axis
    already scaled so that low frequencies are at the bottom (high row index).
    The time axis is left at its original resolution; per-frame code only
    needs to slice and interpolate in the T dimension.
    """
    T, F    = log_mag.shape
    # Resize freq axis F → target_h  (flip so low freq = bottom)
    f_idx   = np.linspace(F - 1, 0, target_h)   # reversed = low freq at row 0 bottom
    f0      = np.floor(f_idx).astype(np.int32).clip(0, F - 1)
    f1      = np.ceil(f_idx).astype(np.int32).clip(0, F - 1)
    ff      = (f_idx - f0).astype(np.float32)   # (target_h,)
    # freq-resized: (T, target_h)
    spec_fh = log_mag[:, f0] * (1.0 - ff)[None, :] + log_mag[:, f1] * ff[None, :]
    # Transpose to (target_h, T)
    spec_ht = spec_fh.T                          # (target_h, T)
    # Colorise — vectorised 3-stop gradient
    lo  = cs["spec_lo"]
    mid = cs["spec_mid"]
    hi  = cs["spec_hi"]
    v   = spec_ht[..., None]                     # (target_h, T, 1)
    rgb = np.where(
        v < 0.5,
        lo  + (mid - lo)  * (v * 2.0),
        mid + (hi  - mid) * ((v - 0.5) * 2.0),
    ).clip(0, 255).astype(np.uint8)              # (target_h, T, 3)
    return rgb


def _slice_spectrogram(spec_ht3: np.ndarray, f0: int, f1: int,
                        target_w: int) -> np.ndarray:
    """Extract a time-window slice and resize the T axis to target_w.

    spec_ht3: (spec_h, T, 3) uint8 — pre-computed by _precompute_spectrogram.
    Returns (spec_h, target_w, 3) uint8.
    """
    slice_ = spec_ht3[:, f0:f1, :]    # (spec_h, window_T, 3)
    wT     = slice_.shape[1]
    if wT < 2:
        result = np.zeros((spec_ht3.shape[0], target_w, 3), dtype=np.uint8)
        result[:] = spec_ht3[:, max(0, f0), :][:, None, :]
        return result
    t_idx  = np.linspace(0, wT - 1, target_w)
    ti0    = np.floor(t_idx).astype(np.int32).clip(0, wT - 1)
    ti1    = np.ceil(t_idx).astype(np.int32).clip(0, wT - 1)
    tf     = (t_idx - ti0).astype(np.float32)[None, :, None]   # (1, W, 1)
    result = (
        slice_[:, ti0, :].astype(np.float32) * (1.0 - tf) +
        slice_[:, ti1, :].astype(np.float32) * tf
    ).clip(0, 255).astype(np.uint8)              # (spec_h, W, 3)
    return result


# ── Waveform helper ────────────────────────────────────────────────────────────

def _render_waveform(canvas: np.ndarray, audio_np: np.ndarray,
                     sample_rate: int, t: float, window_sec: float,
                     wave_y0: int, wave_h: int, cs: dict) -> None:
    """Draw a scrolling waveform into canvas (H, W, 3) in-place using numpy."""
    width    = canvas.shape[1]
    duration = len(audio_np) / sample_rate
    t_start  = t - window_sec / 2
    t_end    = t + window_sec / 2

    s_start  = max(0, int(t_start * sample_rate))
    s_end    = min(len(audio_np), int(t_end * sample_rate))
    chunk    = audio_np[s_start:s_end]
    if len(chunk) < 2:
        return

    # Map chunk to pixel columns
    col_start = max(0, int((max(0.0, t_start) - t_start) / window_sec * width))
    col_end   = min(width, int((min(duration, t_end) - t_start) / window_sec * width))
    n_cols    = col_end - col_start
    if n_cols < 2:
        return

    # Resample chunk to exactly n_cols via linear interpolation
    src_idx  = np.linspace(0, len(chunk) - 1, n_cols)
    chunk_rs = np.interp(src_idx, np.arange(len(chunk)), chunk)

    amps     = np.clip(chunk_rs, -1.0, 1.0)
    mid_y    = wave_h // 2
    halves   = (np.abs(amps) * (mid_y - 2) * 0.9).astype(np.int32)
    top_ys   = np.clip(mid_y - halves, 0, wave_h - 1)   # (n_cols,)
    bot_ys   = np.clip(mid_y + halves, 0, wave_h - 1)   # (n_cols,)

    # Vectorised fill: build (wave_h, n_cols) boolean mask
    rows     = np.arange(wave_h)[:, None]                # (wave_h, 1)
    mask     = (rows >= top_ys[None, :]) & (rows <= bot_ys[None, :])

    section  = canvas[wave_y0:wave_y0 + wave_h, col_start:col_end]
    section[mask]                  = cs["wave_fill"]
    section[top_ys, np.arange(n_cols)] = cs["wave_line"]
    section[bot_ys, np.arange(n_cols)] = cs["wave_line"]

    # Playhead at centre of window
    cx = width // 2
    canvas[wave_y0:wave_y0 + wave_h, cx] = cs["playhead"]


# ── Lyric helpers ──────────────────────────────────────────────────────────────

def parse_lyrics(lyrics_str: str) -> list:
    if not lyrics_str or not lyrics_str.strip():
        return []
    return [ln.strip() for ln in lyrics_str.splitlines() if ln.strip()]


def _lyric_index(t: float, duration: float, n: int) -> int:
    if n == 0 or duration <= 0:
        return 0
    return min(int(t / (duration / n)), n - 1)


# ── Main render function ───────────────────────────────────────────────────────

def render_frames(
    audio_np:      np.ndarray,
    sample_rate:   int,
    fps:           int,
    width:         int,
    height:        int,
    visualization: str,
    color_scheme:  str,
    lyric_lines:   list,
) -> np.ndarray:
    """Return (N, H, W, 3) float32 0-1 tensor of animation frames."""
    cs        = _COLOR_SCHEMES[color_scheme]
    n_samples = len(audio_np)
    duration  = n_samples / max(sample_rate, 1)
    n_frames  = max(1, int(duration * fps))

    has_wave  = "waveform"    in visualization
    has_spec  = "spectrogram" in visualization
    has_lyric = bool(lyric_lines)

    # ── Layout ────────────────────────────────────────────────────────────────
    progress_h = max(3, int(height * 0.009))
    lyric_h    = int(height * 0.22) if has_lyric else 0
    viz_h      = height - lyric_h - progress_h

    if has_wave and has_spec:
        wave_h, spec_h = viz_h // 2, viz_h - viz_h // 2
    elif has_wave:
        wave_h, spec_h = viz_h, 0
    else:
        wave_h, spec_h = 0, viz_h

    wave_y0  = 0
    spec_y0  = wave_h
    lyric_y0 = wave_h + spec_h
    bar_y0   = lyric_y0 + lyric_h

    # ── Fonts ─────────────────────────────────────────────────────────────────
    main_sz = max(18, int(height * 0.054))
    dim_sz  = max(13, int(height * 0.034))
    font_m  = _get_font(main_sz, bold=True)
    font_d  = _get_font(dim_sz,  bold=False)

    # ── Pre-compute spectrogram (freq resize + colorise) once ─────────────────
    spec_precomp = None
    spec_T = 0
    hop    = 256
    if has_spec and spec_h > 0:
        log_mag = _compute_log_spectrogram(audio_np, n_fft=1024, hop=hop)
        # drop top third of bins (mostly noise above ~Nyquist/3)
        log_mag = log_mag[:, : int(log_mag.shape[1] * 0.67)]
        spec_T  = log_mag.shape[0]
        spec_precomp = _precompute_spectrogram(log_mag, spec_h, cs)
        # spec_precomp: (spec_h, spec_T, 3) uint8

    window_sec    = 8.0
    half_win_fr   = int(window_sec * sample_rate / hop / 2)

    # ── Background tile (reused for every frame) ───────────────────────────────
    bg_tile = np.empty((height, width, 3), dtype=np.uint8)
    bg_tile[:] = cs["bg"]

    # ── Frame loop ────────────────────────────────────────────────────────────
    frames_out = []
    for fi in range(n_frames):
        t        = fi / fps
        progress = t / max(duration, 1e-6)

        canvas = bg_tile.copy()

        # ── Waveform ──────────────────────────────────────────────────────────
        if has_wave and wave_h > 0:
            _render_waveform(canvas, audio_np, sample_rate, t, window_sec,
                             wave_y0, wave_h, cs)

        # ── Spectrogram ───────────────────────────────────────────────────────
        if has_spec and spec_h > 0 and spec_precomp is not None:
            ctr_fr = int(t * sample_rate / hop)
            f0     = max(0, ctr_fr - half_win_fr)
            f1     = min(spec_T, ctr_fr + half_win_fr)
            frame_spec = _slice_spectrogram(spec_precomp, f0, f1, width)
            canvas[spec_y0:spec_y0 + spec_h] = frame_spec
            # Playhead
            cx_s = width // 2
            canvas[spec_y0:spec_y0 + spec_h, cx_s] = cs["playhead"]

        # ── Separator ─────────────────────────────────────────────────────────
        if has_lyric and lyric_h > 0:
            sep = cs["separator"]
            canvas[lyric_y0, :] = sep

        # ── Progress bar ──────────────────────────────────────────────────────
        bar_w = max(1, int(width * progress))
        canvas[bar_y0:bar_y0 + progress_h, :bar_w] = cs["bar"]

        # ── Lyrics (PIL — text only) ───────────────────────────────────────────
        if has_lyric and lyric_h > 0:
            img  = Image.fromarray(canvas)
            draw = ImageDraw.Draw(img)
            n    = len(lyric_lines)
            ci   = _lyric_index(t, duration, n)
            cy   = lyric_y0 + lyric_h // 2

            for offset, ck, fnt in (
                (-1, "lyric_dim",  font_d),
                ( 0, "lyric_main", font_m),
                (+1, "lyric_dim",  font_d),
            ):
                idx = ci + offset
                if not (0 <= idx < n):
                    continue
                text = lyric_lines[idx]
                tw   = _text_width(fnt, text)
                tx   = (width - tw) // 2
                if offset == -1:
                    ty = cy - main_sz - dim_sz
                elif offset == 0:
                    ty = cy - main_sz // 2
                else:
                    ty = cy + main_sz // 2 + 4
                _draw_shadowed_text(draw, (tx, ty), text, fnt,
                                    cs[ck], cs["lyric_shadow"])
            canvas = np.array(img)

        frames_out.append(canvas.astype(np.float32) * (1.0 / 255.0))

    return np.stack(frames_out, axis=0)   # (N, H, W, 3)
