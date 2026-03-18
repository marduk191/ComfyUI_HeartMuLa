from typing import Optional

import torch
import torch.nn as nn
from transformers.modeling_utils import PreTrainedModel

import torchtune
from torchtune.models import llama3_2

from .configuration_heartmula import HeartMuLaConfig


# ---------------------------------------------------------------------------
# Backbone / decoder flavour constructors
# ---------------------------------------------------------------------------

def llama3_2_3B() -> torchtune.modules.transformer.TransformerDecoder:
    return llama3_2.llama3_2(
        vocab_size=128_256,
        num_layers=28,
        num_heads=24,
        num_kv_heads=8,
        embed_dim=3072,
        max_seq_len=8192,
        intermediate_dim=8192,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500_000,
        scale_factor=32,
    )


def llama3_2_300M() -> torchtune.modules.transformer.TransformerDecoder:
    return llama3_2.llama3_2(
        vocab_size=128_256,
        num_layers=3,
        num_heads=8,
        num_kv_heads=4,
        embed_dim=3072,
        max_seq_len=2048,
        intermediate_dim=8192,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500_000,
        scale_factor=32,
    )


def llama3_2_7B() -> torchtune.modules.transformer.TransformerDecoder:
    return llama3_2.llama3_2(
        vocab_size=128_256,
        num_layers=32,
        num_heads=32,
        num_kv_heads=8,
        embed_dim=4096,
        max_seq_len=8192,
        intermediate_dim=14336,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500_000,
        scale_factor=32,
    )


def llama3_2_400M() -> torchtune.modules.transformer.TransformerDecoder:
    return llama3_2.llama3_2(
        vocab_size=128_256,
        num_layers=4,
        num_heads=8,
        num_kv_heads=4,
        embed_dim=3072,
        max_seq_len=2048,
        intermediate_dim=8192,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500_000,
        scale_factor=32,
    )


FLAVORS = {
    "llama-3B": llama3_2_3B,
    "llama-300M": llama3_2_300M,
    "llama-7B": llama3_2_7B,
    "llama-400M": llama3_2_400M,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prepare_transformer(
    model: torchtune.modules.transformer.TransformerDecoder,
):
    """Strip the token-embedding and output projection so we can inject our own."""
    embed_dim: int = model.tok_embeddings.embedding_dim
    model.tok_embeddings = nn.Identity()
    model.output = nn.Identity()
    return model, embed_dim


def _create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))


def _index_causal_mask(mask: torch.Tensor, input_pos: torch.Tensor) -> torch.Tensor:
    return mask[input_pos, :]


def _multinomial_sample_one_no_sync(probs: torch.Tensor) -> torch.Tensor:
    q = torch.empty_like(probs).exponential_(1)
    return torch.argmax(probs / q, dim=-1, keepdim=True).to(dtype=torch.int)


def sample_topk(logits: torch.Tensor, topk: int, temperature: float) -> torch.Tensor:
    logits = logits / temperature
    filter_value: float = -float("Inf")
    indices_to_remove = logits < torch.topk(logits, topk)[0][..., -1, None]
    scores_processed = logits.masked_fill(indices_to_remove, filter_value)
    scores_processed = torch.nn.functional.log_softmax(scores_processed, dim=-1)
    probs = torch.nn.functional.softmax(scores_processed, dim=-1)
    return _multinomial_sample_one_no_sync(probs)


def _rope_init_all(model: nn.Module) -> None:
    """
    Call rope_init() on every sub-module that exposes it.

    torchtune >= 0.5 requires explicit rope_init() after setup_caches().
    Earlier versions silently ignore the attribute lookup, so this is safe
    across the full supported version range.
    """
    for module in model.modules():
        if hasattr(module, "rope_init") and callable(module.rope_init):
            module.rope_init()


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class HeartMuLa(PreTrainedModel):
    config_class = HeartMuLaConfig

    def __init__(self, config: HeartMuLaConfig):
        super().__init__(config)
        self.config = config

        self.backbone, backbone_dim = _prepare_transformer(
            FLAVORS[config.backbone_flavor]()
        )
        self.decoder, decoder_dim = _prepare_transformer(
            FLAVORS[config.decoder_flavor]()
        )

        self.text_embeddings = nn.Embedding(config.text_vocab_size, backbone_dim)
        self.audio_embeddings = nn.Embedding(
            config.audio_vocab_size * config.audio_num_codebooks, backbone_dim
        )
        self.unconditional_text_embedding = nn.Embedding(1, backbone_dim)
        self.projection = nn.Linear(backbone_dim, decoder_dim, bias=False)
        self.codebook0_head = nn.Linear(
            backbone_dim, config.audio_vocab_size, bias=False
        )
        self.audio_head = nn.Parameter(
            torch.empty(
                config.audio_num_codebooks - 1, decoder_dim, config.audio_vocab_size
            )
        )
        self.muq_linear = nn.Linear(config.muq_dim, backbone_dim)

        self.post_init()

    def setup_caches(self, max_batch_size: int) -> None:
        """
        Initialise KV-caches for backbone and decoder.

        Fixes applied vs. the original:
        - Explicitly call rope_init() on all attention layers after setup_caches()
          (required by torchtune >= 0.5; safe to call on older versions too).
        - Move backbone/decoder to the correct device after cache allocation so that
          newly registered cache buffers land on CUDA, not CPU.
        - Register causal-mask buffers as non-persistent (no VRAM footprint in state
          dict) and always on the same device as the model parameters.
        """
        dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device

        # Reset any existing caches (ignore if not yet initialised)
        try:
            self.reset_caches()
        except (RuntimeError, AttributeError):
            pass

        # Allocate KV-caches
        self.backbone.setup_caches(max_batch_size, dtype)
        self.decoder.setup_caches(
            max_batch_size,
            dtype,
            decoder_max_seq_len=self.config.audio_num_codebooks,
        )

        # torchtune >= 0.5: RoPE must be explicitly initialised after setup_caches
        _rope_init_all(self.backbone)
        _rope_init_all(self.decoder)

        # Ensure all newly created cache buffers are on the intended device
        self.backbone.to(device)
        self.decoder.to(device)

        # Causal masks — non-persistent so they don't bloat checkpoint saves
        self.register_buffer(
            "backbone_causal_mask",
            _create_causal_mask(self.backbone.max_seq_len, device),
            persistent=False,
        )
        self.register_buffer(
            "decoder_causal_mask",
            _create_causal_mask(self.config.audio_num_codebooks, device),
            persistent=False,
        )

    def reset_caches(self) -> None:
        self.backbone.reset_caches()
        self.decoder.reset_caches()

    # ------------------------------------------------------------------
    # Forward / generation
    # ------------------------------------------------------------------

    def generate_frame(
        self,
        tokens: torch.Tensor,
        tokens_mask: torch.Tensor,
        input_pos: torch.Tensor,
        temperature: float,
        topk: int,
        cfg_scale: float,
        continuous_segments: Optional[torch.Tensor] = None,
        starts: Optional[list] = None,
    ) -> torch.Tensor:
        b, s, _ = tokens.size()

        curr_backbone_mask = _index_causal_mask(self.backbone_causal_mask, input_pos)

        uncond_mask = None
        if cfg_scale > 1.0 and b > 1:
            actual_B = b // 2
            uncond_mask = torch.cat(
                [
                    torch.zeros(actual_B, dtype=torch.bool, device=tokens.device),
                    torch.ones(actual_B, dtype=torch.bool, device=tokens.device),
                ]
            )

        embeds = self._embed_tokens(tokens, uncond_mask=uncond_mask)
        masked_embeds = embeds * tokens_mask.unsqueeze(-1)
        h = masked_embeds.sum(dim=2, dtype=embeds.dtype)

        if continuous_segments is not None:
            continuous_segments = self.muq_linear(continuous_segments)
            if uncond_mask is not None:
                uncond_embed = self.unconditional_text_embedding(
                    torch.zeros(1, device=tokens.device, dtype=torch.long)
                )
                mask_expanded = uncond_mask.view(b, 1).expand_as(continuous_segments)
                continuous_segments = torch.where(
                    mask_expanded, uncond_embed, continuous_segments
                )
            batch_indices = torch.arange(h.shape[0], device=h.device)
            h[batch_indices, starts] = continuous_segments

        h = self.backbone(h, input_pos=input_pos, mask=curr_backbone_mask)
        last_h = h[:, -1, :]

        c0_logits = self.codebook0_head(last_h)
        if cfg_scale > 1.0 and b > 1 and b % 2 == 0:
            actual_B = b // 2
            guided = c0_logits[actual_B:] + (
                c0_logits[:actual_B] - c0_logits[actual_B:]
            ) * cfg_scale
            c0_sample = sample_topk(guided, topk, temperature).repeat(2, 1)
        else:
            c0_sample = sample_topk(c0_logits, topk, temperature)

        c0_embed = self._embed_audio(0, c0_sample)
        self.decoder.reset_caches()

        curr_h = torch.cat([last_h.unsqueeze(1), c0_embed], dim=1).to(embeds.dtype)
        curr_sample = c0_sample.clone()
        curr_pos = (
            torch.arange(0, curr_h.size(1), device=curr_h.device)
            .unsqueeze(0)
            .repeat(curr_h.size(0), 1)
        )

        for i in range(1, self.config.audio_num_codebooks):
            curr_decoder_mask = _index_causal_mask(self.decoder_causal_mask, curr_pos)
            decoder_h = self.decoder(
                self.projection(curr_h), input_pos=curr_pos, mask=curr_decoder_mask
            )
            ci_logits = torch.mm(decoder_h[:, -1, :], self.audio_head[i - 1])

            if cfg_scale > 1.0 and b > 1 and b % 2 == 0:
                actual_B = b // 2
                guided_ci = ci_logits[actual_B:] + (
                    ci_logits[:actual_B] - ci_logits[actual_B:]
                ) * cfg_scale
                ci_sample = sample_topk(guided_ci, topk, temperature).repeat(2, 1)
            else:
                ci_sample = sample_topk(ci_logits, topk, temperature)

            ci_embed = self._embed_audio(i, ci_sample)
            curr_h = ci_embed
            curr_sample = torch.cat([curr_sample, ci_sample], dim=1)
            curr_pos = curr_pos[:, -1:] + 1

        return curr_sample

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    def _embed_audio(self, codebook: int, tokens: torch.Tensor) -> torch.Tensor:
        return self.audio_embeddings(tokens + codebook * self.config.audio_vocab_size)

    def _embed_tokens(
        self, tokens: torch.Tensor, uncond_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        B, S, _ = tokens.size()
        text_embeds = self.text_embeddings(tokens[:, :, -1])

        if uncond_mask is not None:
            uncond_text_embed = self.unconditional_text_embedding(
                torch.zeros(1, device=tokens.device, dtype=torch.long)
            )
            text_embeds = torch.where(
                uncond_mask.view(B, 1, 1).expand_as(text_embeds),
                uncond_text_embed,
                text_embeds,
            )

        audio_tokens = tokens[:, :, :-1] + (
            self.config.audio_vocab_size
            * torch.arange(self.config.audio_num_codebooks, device=tokens.device)
        )
        audio_embeds = self.audio_embeddings(audio_tokens.view(-1)).reshape(
            B, S, self.config.audio_num_codebooks, -1
        )

        return torch.cat([audio_embeds, text_embeds.unsqueeze(-2)], dim=-2)


