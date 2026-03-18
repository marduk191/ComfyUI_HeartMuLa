import gc
import json
import os
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Dict, Optional

import soundfile as sf
import torch
import torchaudio
from huggingface_hub import snapshot_download
from tokenizers import Tokenizer
from transformers import BitsAndBytesConfig

import comfy.model_management as mm
import comfy.utils

from ..heartcodec.modeling_heartcodec import HeartCodec
from ..heartmula.modeling_heartmula import HeartMuLa

# ---------------------------------------------------------------------------
# HuggingFace repo IDs for each local model subfolder.
# None means the model variant is not publicly available.
# ---------------------------------------------------------------------------
_HF_REPO_MAP: Dict[str, Optional[str]] = {
    # Base files (tokenizer + gen_config)
    "__base__": "HeartMuLa/HeartMuLaGen",
    # Codecs
    "HeartCodec-oss": "HeartMuLa/HeartCodec-oss-20260123",
    "HeartCodec-oss-20260123": "HeartMuLa/HeartCodec-oss-20260123",
    # Generation models
    "HeartMuLa-oss-3B": "HeartMuLa/HeartMuLa-oss-3B",
    "HeartMuLa-RL-oss-3B-20260123": "HeartMuLa/HeartMuLa-RL-oss-3B-20260123",
    # Both folder names for the happy-new-year model — "3B-happy-new-year" is
    # the legacy name from earlier workflows; "oss-3B-happy-new-year" is current.
    "HeartMuLa-3B-happy-new-year": "HeartMuLa/HeartMuLa-oss-3B-happy-new-year",
    "HeartMuLa-oss-3B-happy-new-year": "HeartMuLa/HeartMuLa-oss-3B-happy-new-year",
}


def _ensure_downloaded(local_dir: str, repo_id: Optional[str]) -> None:
    """Download *repo_id* into *local_dir* if the directory is absent or empty."""
    if os.path.isdir(local_dir) and os.listdir(local_dir):
        return
    if repo_id is None:
        raise FileNotFoundError(
            f"Model folder {local_dir!r} does not exist and no public "
            f"HuggingFace repository is known for it. "
            f"Please download the weights manually."
        )
    print(f"[HeartMuLa] Downloading {repo_id} → {local_dir}")
    os.makedirs(local_dir, exist_ok=True)
    snapshot_download(repo_id=repo_id, local_dir=local_dir)


def _ensure_base_files(pretrained_path: str) -> None:
    """Download tokenizer.json and gen_config.json if missing."""
    tokenizer_path = os.path.join(pretrained_path, "tokenizer.json")
    gen_config_path = os.path.join(pretrained_path, "gen_config.json")
    if not os.path.exists(tokenizer_path) or not os.path.exists(gen_config_path):
        repo_id = _HF_REPO_MAP["__base__"]
        print(f"[HeartMuLa] Downloading base files ({repo_id}) → {pretrained_path}")
        os.makedirs(pretrained_path, exist_ok=True)
        snapshot_download(
            repo_id=repo_id,
            local_dir=pretrained_path,
            ignore_patterns=["*.md", ".gitattributes"],
        )


@dataclass
class HeartMuLaGenConfig:
    text_bos_id: int = 128000
    text_eos_id: int = 128001
    audio_eos_id: int = 8193
    empty_id: int = 0

    @classmethod
    def from_file(cls, path: str) -> "HeartMuLaGenConfig":
        with open(path, encoding="utf-8") as fp:
            data = json.load(fp)
        return cls(**data)


class HeartMuLaGenPipeline:
    def __init__(
        self,
        model: Optional[HeartMuLa],
        audio_codec: Optional[HeartCodec],
        muq_mulan: Optional[Any],
        text_tokenizer: Tokenizer,
        config: HeartMuLaGenConfig,
        device: torch.device,
        dtype: torch.dtype,
        heartmula_path: Optional[str] = None,
        heartcodec_path: Optional[str] = None,
        bnb_config: Optional[BitsAndBytesConfig] = None,
        num_quantizers: Optional[int] = None,
        use_compile: bool = True,
    ):
        self.model = model
        self.audio_codec = audio_codec
        self.muq_mulan = muq_mulan
        self.text_tokenizer = text_tokenizer
        self.config = config
        self.device = device
        self.dtype = dtype
        self.heartmula_path = heartmula_path
        self.heartcodec_path = heartcodec_path
        self.bnb_config = bnb_config
        self.use_compile = use_compile
        # 8 codebooks + 1 text stream
        self._parallel_number = (num_quantizers + 1) if num_quantizers else 9
        self._muq_dim = model.config.muq_dim if model else None

    # ------------------------------------------------------------------
    # Lazy model loaders
    # ------------------------------------------------------------------

    # Approximate VRAM footprints used for mm.free_memory() requests.
    # These are intentionally conservative upper bounds so ComfyUI has room
    # to evict its own managed models (SD, VAE, CLIP, etc.) when needed.
    _HEARTMULA_BYTES = 7 * 1024 * 1024 * 1024   # ~7 GB for 3B BF16
    _HEARTCODEC_BYTES = 512 * 1024 * 1024        # ~512 MB

    def load_heartmula(self) -> None:
        if self.model is None:
            # Tell ComfyUI how much VRAM we need so it can evict other models.
            mm.free_memory(self._HEARTMULA_BYTES, self.device)
            # Auto-download the generation model weights if not present.
            mula_folder = os.path.basename(self.heartmula_path)
            _ensure_downloaded(self.heartmula_path, _HF_REPO_MAP.get(mula_folder))
            if self.bnb_config is not None:
                # bitsandbytes quantized models must be placed on the target
                # device at load time via device_map — .to() after the fact
                # raises an error because the weights are already quantized.
                self.model = HeartMuLa.from_pretrained(
                    self.heartmula_path,
                    torch_dtype=self.dtype,
                    quantization_config=self.bnb_config,
                    device_map=self.device,
                )
            else:
                self.model = HeartMuLa.from_pretrained(
                    self.heartmula_path,
                    torch_dtype=self.dtype,
                )
                if str(next(self.model.parameters()).device) != str(self.device):
                    self.model.to(self.device)
        self.model.eval()
        self._muq_dim = self.model.config.muq_dim

        # torch.compile wraps the backbone and decoder to eliminate Python dispatch
        # overhead in the autoregressive loop.  dynamic=True handles variable prompt
        # lengths; fullgraph=False allows safe fallbacks for unsupported ops.
        # _orig_mod is set by torch.compile on the returned OptimizedModule, so we
        # use its presence to detect whether compilation already happened.
        if self.use_compile and self.device.type == "cuda" and hasattr(torch, "compile"):
            try:
                if not hasattr(self.model.backbone, "_orig_mod"):
                    self.model.backbone = torch.compile(
                        self.model.backbone, dynamic=True, fullgraph=False
                    )
                if not hasattr(self.model.decoder, "_orig_mod"):
                    self.model.decoder = torch.compile(
                        self.model.decoder, dynamic=True, fullgraph=False
                    )
                print("[HeartMuLa] torch.compile enabled for backbone and decoder.")
            except Exception as e:
                print(f"[HeartMuLa] torch.compile skipped: {e}")

    def load_heartcodec(self) -> None:
        if self.audio_codec is None:
            # Tell ComfyUI how much VRAM we need so it can evict other models.
            mm.free_memory(self._HEARTCODEC_BYTES, self.device)
            # ignore_mismatched_sizes=True is required for the oss-20260123 checkpoint
            # whose weight shapes differ slightly from the config defaults.
            # HeartCodec must run in float32: flow_matching.py creates float32 tensors
            # internally (t_span, timestep scalars) that flow directly into model layers,
            # so a BF16 codec would cause input/bias dtype mismatches at runtime.
            self.audio_codec = HeartCodec.from_pretrained(
                self.heartcodec_path,
                ignore_mismatched_sizes=True,
            )
        if str(next(self.audio_codec.parameters()).device) != str(self.device):
            self.audio_codec.to(self.device)
        self.audio_codec.eval()

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------

    def preprocess(self, inputs: Dict[str, Any], cfg_scale: float) -> Dict[str, Any]:
        self.load_heartmula()

        # Wrap tags with the expected XML delimiters
        tags = inputs["tags"].lower()
        if not tags.startswith("<tag>"):
            tags = f"<tag>{tags}"
        if not tags.endswith("</tag>"):
            tags = f"{tags}</tag>"
        tags_ids = self.text_tokenizer.encode(tags).ids
        if tags_ids[0] != self.config.text_bos_id:
            tags_ids = [self.config.text_bos_id] + tags_ids
        if tags_ids[-1] != self.config.text_eos_id:
            tags_ids = tags_ids + [self.config.text_eos_id]

        muq_embed = torch.zeros([self._muq_dim], dtype=self.dtype, device=self.device)
        muq_idx = len(tags_ids)

        lyrics = inputs["lyrics"].lower()
        lyrics_ids = self.text_tokenizer.encode(lyrics).ids
        if lyrics_ids[0] != self.config.text_bos_id:
            lyrics_ids = [self.config.text_bos_id] + lyrics_ids
        if lyrics_ids[-1] != self.config.text_eos_id:
            lyrics_ids = lyrics_ids + [self.config.text_eos_id]

        prompt_len = len(tags_ids) + 1 + len(lyrics_ids)
        tokens = torch.zeros(
            [prompt_len, self._parallel_number], dtype=torch.long, device=self.device
        )
        tokens[: len(tags_ids), -1] = torch.tensor(tags_ids, device=self.device)
        tokens[len(tags_ids) + 1 :, -1] = torch.tensor(lyrics_ids, device=self.device)
        tokens_mask = torch.zeros_like(tokens, dtype=torch.bool)
        tokens_mask[:, -1] = True

        bs_size = 2 if cfg_scale != 1.0 else 1

        def _cfg_cat(t: torch.Tensor) -> torch.Tensor:
            t = t.unsqueeze(0)
            return torch.cat([t, t], dim=0) if cfg_scale != 1.0 else t

        return {
            "tokens": _cfg_cat(tokens),
            "tokens_mask": _cfg_cat(tokens_mask),
            "muq_embed": _cfg_cat(muq_embed),
            "muq_idx": [muq_idx] * bs_size,
            "pos": _cfg_cat(
                torch.arange(prompt_len, dtype=torch.long, device=self.device)
            ),
        }

    def _get_autocast_ctx(self):
        """Return an appropriate autocast context for the current device."""
        if self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=self.dtype)
        # MPS / XPU / CPU — autocast not needed; caller already holds
        # torch.inference_mode(), so just return a no-op context.
        return nullcontext()

    def _forward(
        self,
        model_inputs: Dict[str, Any],
        max_audio_length_ms: int,
        temperature: float,
        topk: int,
        cfg_scale: float,
    ) -> torch.Tensor:
        self.load_heartmula()
        batch_size = 2 if cfg_scale != 1.0 else 1
        self.model.setup_caches(batch_size)

        frames = []
        with self._get_autocast_ctx():
            curr_token = self.model.generate_frame(
                tokens=model_inputs["tokens"],
                tokens_mask=model_inputs["tokens_mask"],
                input_pos=model_inputs["pos"],
                temperature=temperature,
                topk=topk,
                cfg_scale=cfg_scale,
                continuous_segments=model_inputs["muq_embed"],
                starts=model_inputs["muq_idx"],
            )
        frames.append(curr_token[0:1])

        max_frames = max_audio_length_ms // 80
        pbar = comfy.utils.ProgressBar(max_frames)
        # Update the progress bar every N frames instead of every frame to
        # avoid interrupting the CUDA stream with Python-side queue calls on
        # each step of the autoregressive loop.
        _PBAR_INTERVAL = 10
        _pbar_accum = 0
        for i in range(max_frames):
            padded = torch.ones(
                (curr_token.shape[0], self._parallel_number),
                device=self.device,
                dtype=torch.long,
            ) * self.config.empty_id
            padded[:, :-1] = curr_token
            padded = padded.unsqueeze(1)
            padded_mask = torch.ones_like(padded, dtype=torch.bool)
            padded_mask[..., -1] = False

            with self._get_autocast_ctx():
                curr_token = self.model.generate_frame(
                    tokens=padded,
                    tokens_mask=padded_mask,
                    input_pos=model_inputs["pos"][..., -1:] + i + 1,
                    temperature=temperature,
                    topk=topk,
                    cfg_scale=cfg_scale,
                )
            _pbar_accum += 1
            if _pbar_accum >= _PBAR_INTERVAL:
                pbar.update(_pbar_accum)
                _pbar_accum = 0
            if torch.any(curr_token[0:1] >= self.config.audio_eos_id):
                break
            frames.append(curr_token[0:1])
        if _pbar_accum:
            pbar.update(_pbar_accum)

        return torch.stack(frames).permute(1, 2, 0).squeeze(0).cpu()

    def _empty_cache(self) -> None:
        mm.soft_empty_cache()

    def _synchronize(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elif self.device.type == "mps":
            torch.mps.synchronize()
        elif self.device.type == "xpu":
            torch.xpu.synchronize()

    def _needs_offload(self) -> bool:
        """Return True when VRAM is too constrained to hold both models at once."""
        vstate = mm.vram_state
        # Build the constrained set defensively — future ComfyUI versions may
        # add or rename enum members (e.g. CPU was added after SHARED).
        _constrained = {
            s for name in ("NO_VRAM", "LOW_VRAM", "SHARED", "DISABLED", "CPU")
            if (s := getattr(mm.VRAMState, name, None)) is not None
        }
        if vstate in _constrained:
            return True
        # On NORMAL_VRAM check whether there is headroom for the codec.
        if vstate == mm.VRAMState.NORMAL_VRAM and self.device.type == "cuda":
            free = mm.get_free_memory(self.device)
            return free < self._HEARTCODEC_BYTES * 2
        # HIGH_VRAM — plenty of room, no offload needed.
        return False

    def postprocess(
        self,
        frames: torch.Tensor,
        save_path: str,
        keep_model_loaded: bool,
    ) -> None:
        # Move the generator off the GPU when VRAM is tight so the codec
        # can load cleanly.  ComfyUI's free_memory() (called in
        # load_heartcodec) will also evict its own managed models as needed.
        generator_offloaded = False
        if self._needs_offload() and self.model is not None:
            self.model.to("cpu")
            generator_offloaded = True
            self._empty_cache()
            gc.collect()
            self._synchronize()

        try:
            self.load_heartcodec()
            with torch.inference_mode():
                wav = self.audio_codec.detokenize(
                    frames.to(self.device), device=self.device
                )
                wav = wav.detach().cpu().float()

            try:
                torchaudio.save(save_path, wav, 48000)
            except Exception:
                wav_np = wav.numpy()
                if wav_np.ndim == 2:
                    wav_np = wav_np.T
                sf.write(save_path, wav_np, 48000)
        finally:
            # Codec is always unloaded — it is not needed between generations.
            if self.audio_codec is not None:
                del self.audio_codec
                self.audio_codec = None
            self._empty_cache()
            gc.collect()

            if keep_model_loaded:
                # Restore generator to the compute device if it was offloaded.
                if generator_offloaded and self.model is not None:
                    mm.free_memory(self._HEARTMULA_BYTES, self.device)
                    self.model.to(self.device)
            else:
                if self.model is not None:
                    del self.model
                    self.model = None
                self._empty_cache()

    def __call__(self, inputs: Dict[str, Any], **kwargs) -> None:
        keep_model_loaded = kwargs.get("keep_model_loaded", True)
        cfg_scale = kwargs.get("cfg_scale", 1.5)

        model_inputs = self.preprocess(inputs, cfg_scale=cfg_scale)
        frames = self._forward(
            model_inputs,
            max_audio_length_ms=kwargs.get("max_audio_length_ms", 120000),
            temperature=kwargs.get("temperature", 1.0),
            topk=kwargs.get("topk", 50),
            cfg_scale=cfg_scale,
        )
        self.postprocess(
            frames,
            kwargs.get("save_path", "out.wav"),
            keep_model_loaded,
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_path: str,
        device: torch.device,
        torch_dtype: torch.dtype,
        version: str,
        codec_version: str = "oss",
        bnb_config: Optional[BitsAndBytesConfig] = None,
        lazy_load: bool = True,
        use_compile: bool = True,
    ) -> "HeartMuLaGenPipeline":
        heartcodec_path = os.path.join(pretrained_path, f"HeartCodec-{codec_version}")

        # Build the model directory path based on the version string convention
        if "RL" in version or "2026" in version or "happy" in version:
            heartmula_path = os.path.join(pretrained_path, f"HeartMuLa-{version}")
        else:
            heartmula_path = os.path.join(pretrained_path, f"HeartMuLa-oss-{version}")

        # Auto-download base files and codec if not already present.
        # The generation model itself is downloaded lazily in load_heartmula().
        _ensure_base_files(pretrained_path)
        codec_folder = os.path.basename(heartcodec_path)
        _ensure_downloaded(heartcodec_path, _HF_REPO_MAP.get(codec_folder))

        tokenizer = Tokenizer.from_file(
            os.path.join(pretrained_path, "tokenizer.json")
        )
        gen_config = HeartMuLaGenConfig.from_file(
            os.path.join(pretrained_path, "gen_config.json")
        )

        return cls(
            None,
            None,
            None,
            tokenizer,
            gen_config,
            device,
            torch_dtype,
            heartmula_path,
            heartcodec_path,
            bnb_config,
            use_compile=use_compile,
        )
