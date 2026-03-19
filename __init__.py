import sys
import types
from importlib.machinery import ModuleSpec

# Inject a mock torchcodec module if the real one is unavailable or broken.
# torchcodec has native DLL dependencies that can conflict with other extensions.
if "torchcodec" not in sys.modules:
    try:
        _m = types.ModuleType("torchcodec")
        _m.__spec__ = ModuleSpec("torchcodec", None, origin="built-in")
        _m.__version__ = "0.2.0"
        _d = types.ModuleType("torchcodec.decoders")

        class MockDecoder:
            pass

        _d.AudioDecoder = MockDecoder
        _d.VideoDecoder = MockDecoder
        _m.decoders = _d
        sys.modules["torchcodec"] = _m
        sys.modules["torchcodec.decoders"] = _d
    except Exception:
        pass

import gc
import logging
import os
import uuid
import warnings

import numpy as np
import soundfile as sf
import torch
import torchaudio
from transformers import BitsAndBytesConfig

import comfy.model_management as mm
import folder_paths

# Allow large KV-cache allocations without OOM on fragmented VRAM.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Ensure ffmpeg is available for transformers audio pipelines.
# imageio-ffmpeg ships a self-contained ffmpeg binary; add it to PATH so that
# subprocess-based decoders (transformers audio_utils) can find it.
try:
    import imageio_ffmpeg
    _ffmpeg_dir = os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe())
    if _ffmpeg_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
except Exception:
    pass  # ffmpeg may already be on PATH, or will fail later with a clear message

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torchtune").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")
logger = logging.getLogger("HeartMuLa")


current_dir = os.path.dirname(os.path.abspath(__file__))
util_dir = os.path.join(current_dir, "util")
if util_dir not in sys.path:
    sys.path.insert(0, util_dir)

folder_paths.add_model_folder_path(
    "HeartMuLa", os.path.join(folder_paths.models_dir, "HeartMuLa")
)
folder_paths.add_model_folder_path(
    "HeartMuLa", os.path.join(current_dir, "util", "heartlib", "ckpt")
)


def get_model_base_dir():
    paths = folder_paths.get_folder_paths("HeartMuLa")
    for p in paths:
        if os.path.exists(p):
            return p
    return paths[0]


MODEL_BASE_DIR = get_model_base_dir()


def _get_device() -> torch.device:
    return mm.get_torch_device()


def _get_dtype(device: torch.device) -> torch.dtype:
    if mm.should_use_bf16(device):
        return torch.bfloat16
    if mm.should_use_fp16(device):
        return torch.float16
    return torch.float32


class HeartMuLaModelManager:
    """Singleton that owns all loaded HeartMuLa pipelines."""

    _instance = None
    _gen_pipes: dict = {}
    _transcribe_pipe = None
    _device: torch.device = _get_device()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_gen_pipeline(
        self, version: str = "3B", codec_version: str = "oss", quantize_4bit: bool = False,
        use_compile: bool = True,
    ):
        # Normalise empty / None codec_version
        if not codec_version or str(codec_version).lower() == "none":
            codec_version = "oss"

        key = (version, codec_version, quantize_4bit, use_compile)
        if key not in self._gen_pipes:
            from heartlib import HeartMuLaGenPipeline

            model_dtype = _get_dtype(self._device)
            bnb_config = None

            if quantize_4bit:
                if self._device.type != "cuda":
                    logger.warning("HeartMuLa: 4-bit quantization requires a CUDA device — skipping.")
                else:
                    quant_type = "nf4"
                    try:
                        major, _ = torch.cuda.get_device_capability()
                        # Blackwell (compute capability 10+) supports native FP4
                        if major >= 10:
                            quant_type = "fp4"
                    except Exception:
                        pass
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type=quant_type,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_use_double_quant=True,
                    )

            self._gen_pipes[key] = HeartMuLaGenPipeline.from_pretrained(
                MODEL_BASE_DIR,
                device=self._device,
                torch_dtype=model_dtype,
                version=version,
                codec_version=codec_version,
                lazy_load=True,
                bnb_config=bnb_config,
                use_compile=use_compile,
            )
            mm.soft_empty_cache()
            gc.collect()

        return self._gen_pipes[key]

    def get_transcribe_pipeline(self):
        if self._transcribe_pipe is None:
            from heartlib import HeartTranscriptorPipeline

            # ~1.5 GB for Whisper large-v2
            mm.free_memory(1_500 * 1024 * 1024, self._device)
            self._transcribe_pipe = HeartTranscriptorPipeline.from_pretrained(
                MODEL_BASE_DIR,
                device=self._device,
                dtype=torch.float16,
            )
            mm.soft_empty_cache()
            gc.collect()
        return self._transcribe_pipe


class HeartMuLa_Generate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lyrics": (
                    "STRING",
                    {"multiline": True, "placeholder": "[Verse]\n..."},
                ),
                "tags": (
                    "STRING",
                    {
                        "multiline": True,
                        "placeholder": "upbeat pop, electric guitar, 120bpm",
                    },
                ),
                "version": (
                    ["3B", "RL-oss-3B-20260123", "3B-happy-new-year", "oss-3B-happy-new-year"],
                    {"default": "3B"},
                ),
                "codec_version": (
                    ["oss", "oss-20260123", "none"],
                    {"default": "oss"},
                ),
                "seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF},
                ),
                "max_audio_length_seconds": (
                    "INT",
                    {"default": 240, "min": 10, "max": 600, "step": 1},
                ),
                "topk": (
                    "INT",
                    {"default": 50, "min": 1, "max": 250, "step": 1},
                ),
                "temperature": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.01},
                ),
                "cfg_scale": (
                    "FLOAT",
                    {"default": 1.5, "min": 1.0, "max": 10.0, "step": 0.1},
                ),
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
                "quantize_4bit": ("BOOLEAN", {"default": False}),
                "use_compile": ("BOOLEAN", {"default": False}),
                "tf32_matmul": ("BOOLEAN", {"default": True}),
                "cudnn_benchmark": ("BOOLEAN", {"default": True}),
                "flash_attention": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio_output", "filepath")
    FUNCTION = "generate"
    CATEGORY = "HeartMuLa"

    def generate(
        self,
        lyrics,
        tags,
        version,
        codec_version,
        seed,
        max_audio_length_seconds,
        topk,
        temperature,
        cfg_scale,
        keep_model_loaded,
        quantize_4bit=False,
        use_compile=False,
        tf32_matmul=True,
        cudnn_benchmark=True,
        flash_attention=True,
    ):
        if torch.cuda.is_available():
            torch.set_float32_matmul_precision("high" if tf32_matmul else "highest")
            torch.backends.cudnn.benchmark = cudnn_benchmark
            torch.backends.cuda.enable_flash_sdp(flash_attention)

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed & 0xFFFFFFFF)

        max_audio_length_ms = int(max_audio_length_seconds * 1000)
        manager = HeartMuLaModelManager()
        pipe = manager.get_gen_pipeline(
            version, codec_version=codec_version, quantize_4bit=quantize_4bit,
            use_compile=use_compile,
        )

        output_dir = folder_paths.get_temp_directory()
        os.makedirs(output_dir, exist_ok=True)
        filename = f"heartmula_gen_{uuid.uuid4().hex}.wav"
        out_path = os.path.join(output_dir, filename)

        try:
            with torch.inference_mode():
                pipe(
                    {"lyrics": lyrics, "tags": tags},
                    max_audio_length_ms=max_audio_length_ms,
                    save_path=out_path,
                    topk=topk,
                    temperature=temperature,
                    cfg_scale=cfg_scale,
                    keep_model_loaded=keep_model_loaded,
                )
        except Exception as e:
            print(f"HeartMuLa: generation failed — {e}")
            raise
        finally:
            mm.soft_empty_cache()
            gc.collect()

        # Load the saved WAV back as a ComfyUI AUDIO tensor
        try:
            waveform, sample_rate = torchaudio.load(out_path)
            waveform = waveform.float()
        except Exception:
            waveform_np, sample_rate = sf.read(out_path)
            if waveform_np.ndim == 1:
                waveform_np = waveform_np[np.newaxis, :]
            else:
                waveform_np = waveform_np.T
            waveform = torch.from_numpy(waveform_np).float()

        # ComfyUI AUDIO format: (batch, channels, samples)
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(0)

        audio_output = {"waveform": waveform, "sample_rate": sample_rate}
        return (audio_output, out_path)


class HeartMuLa_Transcribe:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_input": ("AUDIO",),
                "num_beams": (
                    "INT",
                    {"default": 2, "min": 1, "max": 8, "step": 1},
                ),
                "temperature_tuple": (
                    "STRING",
                    {"default": "0.0,0.1,0.2,0.4"},
                ),
                "no_speech_threshold": (
                    "FLOAT",
                    {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "logprob_threshold": (
                    "FLOAT",
                    {"default": -1.0, "min": -5.0, "max": 5.0, "step": 0.1},
                ),
                "compression_ratio_threshold": (
                    "FLOAT",
                    {"default": 1.8, "min": 0.0, "max": 5.0, "step": 0.1},
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("lyrics_text",)
    FUNCTION = "transcribe"
    CATEGORY = "HeartMuLa"

    def transcribe(
        self,
        audio_input,
        num_beams,
        temperature_tuple,
        no_speech_threshold,
        logprob_threshold,
        compression_ratio_threshold,
    ):
        try:
            torchaudio.set_audio_backend("soundfile")
        except Exception:
            pass

        if isinstance(audio_input, str):
            # Bare file path string
            waveform, sr = torchaudio.load(audio_input)
        elif isinstance(audio_input, (list, tuple)):
            # Legacy (sr, waveform) tuple — waveform may itself be a path string
            sr, waveform = audio_input
            if isinstance(waveform, str):
                waveform, sr = torchaudio.load(waveform)
            elif isinstance(waveform, np.ndarray):
                waveform = torch.from_numpy(waveform)
        elif hasattr(audio_input, "__getitem__"):
            # Real dict or dict-like object (e.g. VHS LazyAudioMap)
            waveform, sr = audio_input["waveform"], audio_input["sample_rate"]
        else:
            raise ValueError(
                f"HeartMuLa Transcribe: unsupported audio input type {type(audio_input)}. "
                "Connect an AUDIO or VHS_AUDIO output."
            )

        if waveform.ndim == 3:
            waveform = waveform.squeeze(0)
        elif waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        waveform = waveform.to(torch.float32).cpu()

        # Warn if audio appears to be a stereo mix — the model was trained on
        # isolated vocal recordings and will produce poor results on full mixes.
        # Use a source separator (e.g. demucs) to extract vocals first.
        if waveform.shape[0] > 1:
            logger.warning(
                "[HeartMuLa] Transcribe received multi-channel (mixed) audio. "
                "HeartTranscriptor was trained on isolated vocals. For best results, "
                "run a source separator (e.g. demucs) to extract the vocal stem first."
            )

        # Convert to mono float32 numpy array at the original sample rate.
        # Passing {"array": ..., "sampling_rate": ...} to the pipeline skips
        # ffmpeg entirely — the feature extractor handles resampling internally.
        wav_np = waveform.numpy()
        if wav_np.ndim == 2:
            wav_np = wav_np.mean(axis=0)  # (channels, samples) → mono
        wav_np = wav_np.astype("float32")

        try:
            temp_tuple = tuple(float(x.strip()) for x in temperature_tuple.split(","))
        except Exception:
            temp_tuple = (0.0, 0.1, 0.2, 0.4)

        manager = HeartMuLaModelManager()
        pipe = manager.get_transcribe_pipeline()

        try:
            with torch.inference_mode():
                result = pipe(
                    {"array": wav_np, "sampling_rate": sr},
                    max_new_tokens=256,
                    num_beams=num_beams,
                    task="transcribe",
                    condition_on_prev_tokens=False,
                    compression_ratio_threshold=compression_ratio_threshold,
                    temperature=temp_tuple,
                    logprob_threshold=logprob_threshold,
                    no_speech_threshold=no_speech_threshold,
                )
        finally:
            mm.soft_empty_cache()
            gc.collect()

        text = result if isinstance(result, str) else result.get("text", str(result))
        return (text,)


class HeartMuLa_AudioViz:
    """Render animated visualization frames from audio with optional lyric overlay."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "fps": ("INT", {"default": 24, "min": 8, "max": 60, "step": 1}),
                "width": (
                    "INT",
                    {"default": 768, "min": 256, "max": 1920, "step": 64},
                ),
                "height": (
                    "INT",
                    {"default": 432, "min": 256, "max": 1080, "step": 64},
                ),
                "visualization": (
                    ["waveform+spectrogram", "waveform", "spectrogram"],
                ),
                "color_scheme": (["dark", "neon", "light"],),
            },
            "optional": {
                "lyrics": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    FUNCTION = "visualize"
    CATEGORY = "HeartMuLa"

    def visualize(
        self,
        audio,
        fps,
        width,
        height,
        visualization,
        color_scheme,
        lyrics="",
    ):
        from .util.audio_visualizer import render_frames, parse_lyrics

        waveform = audio["waveform"]   # (B, C, S)
        sr = audio["sample_rate"]

        # collapse to mono float32 numpy
        wav = waveform[0]              # (C, S)
        mono = wav.mean(dim=0) if wav.shape[0] > 1 else wav[0]
        mono_np = mono.float().cpu().numpy()

        # normalise to -1..1
        peak = float(np.abs(mono_np).max())
        if peak > 1e-6:
            mono_np = mono_np / peak

        lyric_lines = parse_lyrics(lyrics)

        logger.info(
            "HeartMuLa: rendering %d frames at %dx%d (%s, %s)",
            max(1, int(len(mono_np) / sr * fps)),
            width,
            height,
            visualization,
            color_scheme,
        )

        frames = render_frames(
            audio_np=mono_np,
            sample_rate=sr,
            fps=fps,
            width=width,
            height=height,
            visualization=visualization,
            color_scheme=color_scheme,
            lyric_lines=lyric_lines,
        )

        return (torch.from_numpy(frames),)


class HeartMuLa_MilkDrop:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio":        ("AUDIO",),
                "fps":          ("INT",   {"default": 24,  "min": 12,  "max": 60,   "step": 1}),
                "width":        ("INT",   {"default": 512, "min": 64,  "max": 1920, "step": 8}),
                "height":       ("INT",   {"default": 512, "min": 64,  "max": 1080, "step": 8}),
                "effect":       (["combined", "plasma", "tunnel", "kaleidoscope"],),
                "color_scheme": (["rainbow", "fire", "ice", "neon"],),
                "intensity":    ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0,  "step": 0.1}),
                "render_scale": ("INT",   {"default": 1,   "min": 1,   "max": 4,    "step": 1,
                                           "tooltip": "Compute at 1/N resolution then upsample. "
                                                      "Higher = faster but lower quality."}),
            },
            "optional": {
                "lyrics": ("STRING", {"default": "", "multiline": True,
                                      "tooltip": "One lyric line per row, timed evenly across the audio."}),
            },
        }

    RETURN_TYPES  = ("IMAGE",)
    RETURN_NAMES  = ("frames",)
    FUNCTION      = "render"
    CATEGORY      = "HeartMuLa"

    def render(self, audio, fps, width, height, effect,
               color_scheme, intensity, render_scale, lyrics=""):
        from .util.milkdrop import render_milkdrop_frames
        from .util.audio_visualizer import parse_lyrics

        waveform = audio["waveform"]   # (B, C, S)
        sr       = audio["sample_rate"]
        wav      = waveform[0]
        mono     = wav.mean(dim=0) if wav.shape[0] > 1 else wav[0]
        mono_np  = mono.float().cpu().numpy()
        peak     = float(np.abs(mono_np).max())
        if peak > 1e-6:
            mono_np = mono_np / peak

        lyric_lines = parse_lyrics(lyrics)

        n_frames = max(1, int(len(mono_np) / sr * fps))
        logger.info(
            "HeartMuLa MilkDrop: rendering %d frames at %dx%d (%s, %s, scale=%d)",
            n_frames, width, height, effect, color_scheme, render_scale,
        )

        frames = render_milkdrop_frames(
            audio_np=mono_np,
            sample_rate=sr,
            fps=fps,
            width=width,
            height=height,
            effect=effect,
            color_scheme=color_scheme,
            intensity=intensity,
            render_scale=render_scale,
            lyric_lines=lyric_lines,
        )

        return (torch.from_numpy(frames),)


class HeartMuLa_FramePicker:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "frame_index": ("INT", {"default": 0, "min": 0, "max": 99999, "step": 1,
                                        "tooltip": "Index of the frame to extract (0 = first frame)."}),
            }
        }

    RETURN_TYPES  = ("IMAGE",)
    RETURN_NAMES  = ("image",)
    FUNCTION      = "pick"
    CATEGORY      = "HeartMuLa"

    def pick(self, frames, frame_index):
        # frames: (N, H, W, C)
        n = frames.shape[0]
        idx = max(0, min(frame_index, n - 1))
        return (frames[idx].unsqueeze(0),)


class HeartMuLa_Demucs:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio":             ("AUDIO",),
                "model":             (["htdemucs_ft", "htdemucs", "mdx_extra", "mdx_extra_q"],),
                "shifts":            ("INT",   {"default": 1, "min": 1, "max": 10, "step": 1,
                                                "tooltip": "Random shifts for TTA. Higher = better quality, slower."}),
                "overlap":           ("FLOAT", {"default": 0.25, "min": 0.0, "max": 0.5, "step": 0.05}),
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES  = ("AUDIO", "AUDIO", "AUDIO", "AUDIO")
    RETURN_NAMES  = ("vocals", "drums", "bass", "other")
    FUNCTION      = "separate"
    CATEGORY      = "HeartMuLa"

    def separate(self, audio, model, shifts, overlap, keep_model_loaded):
        from .util.demucs_node import separate_stems
        import comfy.model_management as mm

        device = mm.get_torch_device()
        stems  = separate_stems(
            audio_input=audio,
            model_name=model,
            device=device,
            shifts=shifts,
            overlap=overlap,
            keep_model_loaded=keep_model_loaded,
        )
        return (stems["vocals"], stems["drums"], stems["bass"], stems["other"])


_ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")


class HeartMuLa_StaticFrames:
    """Repeats a static image from the assets folder to match audio duration."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "fps":   ("INT", {"default": 24, "min": 1, "max": 60, "step": 1}),
            }
        }

    RETURN_TYPES  = ("IMAGE",)
    RETURN_NAMES  = ("frames",)
    FUNCTION      = "generate"
    CATEGORY      = "HeartMuLa"

    def generate(self, audio, fps):
        from PIL import Image as PILImage

        img_path = os.path.join(_ASSETS_DIR, "noviz.jpg")
        img      = PILImage.open(img_path).convert("RGB")
        frame    = torch.from_numpy(
            np.array(img, dtype=np.float32) / 255.0
        ).unsqueeze(0)   # (1, H, W, 3)

        waveform    = audio["waveform"]          # (B, C, S)
        sr          = audio["sample_rate"]
        n_samples   = waveform.shape[-1]
        duration    = n_samples / max(sr, 1)
        n_frames    = max(1, int(duration * fps))

        frames = frame.expand(n_frames, -1, -1, -1).contiguous()
        return (frames,)


NODE_CLASS_MAPPINGS = {
    "HeartMuLa_Generate":      HeartMuLa_Generate,
    "HeartMuLa_Transcribe":    HeartMuLa_Transcribe,
    "HeartMuLa_AudioViz":      HeartMuLa_AudioViz,
    "HeartMuLa_MilkDrop":      HeartMuLa_MilkDrop,
    "HeartMuLa_FramePicker":   HeartMuLa_FramePicker,
    "HeartMuLa_Demucs":        HeartMuLa_Demucs,
    "HeartMuLa_StaticFrames":  HeartMuLa_StaticFrames,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HeartMuLa_Generate":      "HeartMuLa Music Generator",
    "HeartMuLa_Transcribe":    "HeartMuLa Lyrics Transcriber",
    "HeartMuLa_AudioViz":      "HeartMuLa Audio Visualizer",
    "HeartMuLa_MilkDrop":      "HeartMuLa MilkDrop Visualizer",
    "HeartMuLa_FramePicker":   "HeartMuLa Frame Picker",
    "HeartMuLa_Demucs":        "HeartMuLa Demucs Separator",
    "HeartMuLa_StaticFrames":  "HeartMuLa Static Frames",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
