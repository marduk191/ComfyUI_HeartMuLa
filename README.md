# ComfyUI_HeartMuLa

ComfyUI custom nodes for AI music generation, lyrics transcription, source separation, and audio visualization — powered by the [HeartMuLa](https://huggingface.co/HeartMuLa) model family.

There are visualizer demo videos at https://huggingface.co/marduk191/ComfyUI_HeartMuLa/tree/main/assets/visualizer_examples

---

## Nodes

### 🎵 HeartMuLa Music Generator
Generates music from lyrics and style tags using the HeartMuLa autoregressive language model (Llama 3.2 backbone + HeartCodec audio codec).

**Inputs**
| Input | Type | Description |
|---|---|---|
| `lyrics` | STRING | Song lyrics to condition generation on |
| `tags` | STRING | Style/genre tags (e.g. `pop, female vocal, piano`) |
| `version` | DROPDOWN | Model version (`RL-oss-3B-20260123`, `3B`, etc.) |
| `codec_version` | DROPDOWN | Codec checkpoint version |
| `seed` | INT | Random seed |
| `max_audio_length_seconds` | FLOAT | Maximum output duration |
| `topk` | INT | Top-k sampling |
| `temperature` | FLOAT | Sampling temperature |
| `cfg_scale` | FLOAT | Classifier-free guidance scale |
| `keep_model_loaded` | BOOLEAN | Keep model in VRAM between runs |
| `offload_mode` | DROPDOWN | `auto` or `aggressive` VRAM offloading |
| `quantize_4bit` | BOOLEAN | NF4/FP4 4-bit quantization (reduces VRAM) |
| `use_compile` | BOOLEAN | `torch.compile` for faster inference |
| `tf32_matmul` | BOOLEAN | TF32 matmul precision (speeds up HeartCodec) |
| `cudnn_benchmark` | BOOLEAN | cuDNN autotuner |
| `flash_attention` | BOOLEAN | Flash Attention via SDPA |

**Outputs:** `AUDIO`, `STRING` (file path)

---

### 🎤 HeartMuLa Lyrics Transcriber
Transcribes lyrics from audio using a Whisper-based model fine-tuned for music.

**Inputs**
| Input | Type | Description |
|---|---|---|
| `audio_input` | AUDIO | Audio to transcribe (also accepts VHS_AUDIO) |
| `num_beams` | INT | Beam search width |
| `compression_ratio_threshold` | FLOAT | Filter low-quality segments |
| `no_speech_threshold` | FLOAT | Silence detection threshold |
| `logprob_threshold` | FLOAT | Low-confidence segment filter |
| `temperature_tuple` | STRING | Comma-separated fallback temperatures |

**Outputs:** `STRING` (transcribed lyrics)

---

### 📊 HeartMuLa Audio Visualizer
Pre-renders a video frame sequence with a scrolling waveform, spectrogram, and synchronized lyric overlay.

**Inputs**
| Input | Type | Description |
|---|---|---|
| `audio` | AUDIO | Audio to visualize |
| `lyrics` | STRING | (optional) One lyric line per row |
| `fps` | INT | Output frame rate |
| `width` / `height` | INT | Frame dimensions |
| `window_sec` | FLOAT | Waveform scroll window width |
| `color_scheme` | DROPDOWN | `dark`, `neon`, `light` |
| `spec_scale` | FLOAT | Spectrogram height ratio |

**Outputs:** `IMAGE` (batch of frames)

---

### 🌀 HeartMuLa MilkDrop Visualizer
Pre-renders psychedelic audio-reactive frames using plasma, tunnel, and kaleidoscope effects driven by FFT analysis of the audio.

**Inputs**
| Input | Type | Description |
|---|---|---|
| `audio` | AUDIO | Audio to visualize |
| `lyrics` | STRING | (optional) One lyric line per row |
| `fps` | INT | Output frame rate |
| `width` / `height` | INT | Frame dimensions |
| `effect` | DROPDOWN | `combined`, `plasma`, `tunnel`, `kaleidoscope` |
| `color_scheme` | DROPDOWN | `rainbow`, `fire`, `ice`, `neon` |
| `intensity` | FLOAT | Effect reactivity multiplier |
| `render_scale` | INT | Downsample factor for faster rendering (1=full quality) |

**Outputs:** `IMAGE` (batch of frames)

---

### 🎛️ HeartMuLa Demucs
Separates audio into 4 stems using [Demucs](https://github.com/facebookresearch/demucs). Models are downloaded automatically on first use.

**Inputs**
| Input | Type | Description |
|---|---|---|
| `audio` | AUDIO | Audio to separate |
| `model` | DROPDOWN | `htdemucs_ft`, `htdemucs`, `mdx_extra`, `mdx_extra_q` |
| `shifts` | INT | Number of random shifts for better quality (higher = slower) |
| `device` | DROPDOWN | `cuda` or `cpu` |

**Outputs:** `AUDIO` × 4 — `vocals`, `drums`, `bass`, `other`

> **Model guide:** `htdemucs_ft` is best for vocal separation. `mdx_extra` is a strong alternative. `mdx_extra_q` is fastest but lower quality.

---

### 🖼️ HeartMuLa Frame Picker
Extracts a single frame from any IMAGE batch by index. Useful for grabbing a thumbnail from visualizer output.

**Inputs**
| Input | Type | Description |
|---|---|---|
| `frames` | IMAGE | Any IMAGE batch |
| `frame_index` | INT | Frame to extract (0 = first, clamped to valid range) |

**Outputs:** `IMAGE` (single frame)

---

## Installation

### Via ComfyUI Manager (recommended)
Search for `ComfyUI_HeartMuLa` in the Manager and install.

### Manual
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/marduk191/ComfyUI_HeartMuLa
```

Dependencies install automatically via `install.py` on first launch. The following packages are added to your ComfyUI environment:

| Package | Purpose |
|---|---|
| `torchtune >= 0.4.0` | HeartMuLa transformer backbone |
| `torchao` | Required by torchtune |
| `vector-quantize-pytorch` | HeartCodec residual quantizer |
| `soundfile` | Audio I/O |
| `bitsandbytes` | 4-bit quantization |
| `demucs >= 4.0.0` | Source separation |
| `imageio-ffmpeg` | FFmpeg backend |

> **Note:** If your environment uses `fsspec >= 2026`, `install.py` automatically protects it by installing torchtune with `--no-deps`.

---

## Model Downloads

Models are downloaded automatically from HuggingFace on first use and cached to `ComfyUI/models/HeartMuLa/`.

| Model | HuggingFace Repo |
|---|---|
| HeartMuLa RL-oss-3B | `HeartMuLa/HeartMuLa-RL-oss-3B-20260123` |
| HeartMuLa oss-3B | `HeartMuLa/HeartMuLa-oss-3B-happy-new-year` |
| HeartMuLa 3B | `HeartMuLa/HeartMuLa-3B` |
| HeartTranscriptor | `HeartMuLa/HeartTranscriptor` |
| HeartCodec | bundled with music generation models |

---

## Requirements

- Python 3.10+
- PyTorch 2.1+ with CUDA (CPU supported but slow)
- ComfyUI (recent build)

**RTX 5090 / Blackwell:** Flash Attention 3 is active automatically via PyTorch SDPA on CUDA 13. FP4 quantization is used instead of NF4 when `quantize_4bit` is enabled on CC ≥ 10 GPUs.

---

## Example Workflows

| Workflow | Description |
|---|---|
| `Generate Music.json` | Basic music generation → preview |
| `Generate Music - Visualizers.json` | Generation with Audio Visualizer and MilkDrop output |
| `Lyrics Transcriber.json` | Transcribe lyrics from any audio file |

---

## License

See [LICENSE](LICENSE).
