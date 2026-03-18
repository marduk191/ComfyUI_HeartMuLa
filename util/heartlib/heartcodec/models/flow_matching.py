from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from vector_quantize_pytorch import ResidualVQ

from .transformer import LlamaTransformer


def _rvq_get_output_from_indices(
    vq_embed: ResidualVQ, indices: torch.Tensor
) -> torch.Tensor:
    """
    Reconstruct quantised vectors from codebook indices.

    The ``get_output_from_indices`` method was introduced in
    vector_quantize_pytorch >= 1.9 and is the preferred API.  Some
    environments end up with an older or differently-patched version that
    lacks the method (issue #70), so we fall back to a manual reconstruction
    using the individual quantiser codebooks.

    Args:
        vq_embed: A fitted ``ResidualVQ`` instance.
        indices:  Shape ``(B, T, num_quantizers)``.

    Returns:
        Quantised representation of shape ``(B, T, dim)``.
    """
    if hasattr(vq_embed, "get_output_from_indices"):
        return vq_embed.get_output_from_indices(indices)

    # ------------------------------------------------------------------
    # Fallback: reconstruct by summing per-quantiser codebook look-ups.
    # ``vq_embed.layers`` is a ModuleList of VectorQuantize objects; each
    # exposes a ``._codebook.embed`` tensor of shape ``(1, codebook_size, dim)``.
    # ------------------------------------------------------------------
    output: Optional[torch.Tensor] = None
    for i, layer in enumerate(vq_embed.layers):
        idx = indices[..., i]  # (B, T)
        # embed: (1, codebook_size, dim) → (codebook_size, dim)
        codebook = layer._codebook.embed[0]
        emb = F.embedding(idx, codebook)  # (B, T, dim)
        output = emb if output is None else output + emb

    if output is None:
        raise RuntimeError(
            "ResidualVQ has no layers — cannot reconstruct output from indices."
        )
    return output


class FlowMatching(nn.Module):
    def __init__(
        self,
        # RVQ parameters
        dim: int = 512,
        codebook_size: int = 8192,
        decay: float = 0.9,
        commitment_weight: float = 1.0,
        threshold_ema_dead_code: int = 2,
        use_cosine_sim: bool = False,
        codebook_dim: int = 32,
        num_quantizers: int = 8,
        # Diffusion-transformer backbone parameters
        attention_head_dim: int = 64,
        in_channels: int = 1024,
        norm_type: str = "ada_norm_single",
        num_attention_heads: int = 24,
        num_layers: int = 24,
        num_layers_2: int = 6,
        out_channels: int = 256,
    ):
        super().__init__()

        self.vq_embed = ResidualVQ(
            dim=dim,
            codebook_size=codebook_size,
            decay=decay,
            commitment_weight=commitment_weight,
            threshold_ema_dead_code=threshold_ema_dead_code,
            use_cosine_sim=use_cosine_sim,
            codebook_dim=codebook_dim,
            num_quantizers=num_quantizers,
        )
        self.cond_feature_emb = nn.Linear(dim, dim)
        self.zero_cond_embedding1 = nn.Parameter(torch.randn(dim))
        self.estimator = LlamaTransformer(
            attention_head_dim=attention_head_dim,
            in_channels=in_channels,
            norm_type=norm_type,
            num_attention_heads=num_attention_heads,
            num_layers=num_layers,
            num_layers_2=num_layers_2,
            out_channels=out_channels,
        )
        self.latent_dim = out_channels

    @torch.no_grad()
    def inference_codes(
        self,
        codes: list,
        true_latents: torch.Tensor,
        latent_length: int,
        incontext_length: int,
        guidance_scale: float = 2.0,
        num_steps: int = 20,
        disable_progress: bool = True,
        scenario: str = "start_seg",
    ) -> torch.Tensor:
        device = true_latents.device
        dtype = true_latents.dtype

        codes_bestrq_emb = codes[0]
        batch_size = codes_bestrq_emb.shape[0]

        self.vq_embed.eval()
        # Compatibility wrapper handles both old and new vector_quantize_pytorch APIs
        quantized_feature_emb = _rvq_get_output_from_indices(
            self.vq_embed, codes_bestrq_emb.transpose(1, 2)
        )
        quantized_feature_emb = self.cond_feature_emb(quantized_feature_emb)  # (B, T, 512)
        quantized_feature_emb = F.interpolate(
            quantized_feature_emb.permute(0, 2, 1), scale_factor=2, mode="nearest"
        ).permute(0, 2, 1)

        num_frames = quantized_feature_emb.shape[1]
        latents = torch.randn(
            (batch_size, num_frames, self.latent_dim), device=device, dtype=dtype
        )
        latent_masks = torch.zeros(
            latents.shape[0], latents.shape[1], dtype=torch.int64, device=device
        )
        latent_masks[:, 0:latent_length] = 2
        if scenario == "other_seg":
            latent_masks[:, 0:incontext_length] = 1

        mask_active = (latent_masks > 0.5).unsqueeze(-1)
        quantized_feature_emb = (
            mask_active * quantized_feature_emb
            + (~mask_active) * self.zero_cond_embedding1.unsqueeze(0)
        )

        incontext_latents = (
            true_latents
            * ((latent_masks > 0.5) & (latent_masks < 1.5)).unsqueeze(-1).float()
        )
        incontext_length = int(
            ((latent_masks > 0.5) & (latent_masks < 1.5)).sum(-1)[0].item()
        )

        additional_model_input = quantized_feature_emb
        temperature = 1.0
        t_span = torch.linspace(0, 1, num_steps + 1, device=device)

        latents = self.solve_euler(
            latents * temperature,
            incontext_latents,
            incontext_length,
            t_span,
            additional_model_input,
            guidance_scale,
        )

        latents[:, 0:incontext_length, :] = incontext_latents[:, 0:incontext_length, :]
        return latents

    def solve_euler(
        self,
        x: torch.Tensor,
        incontext_x: torch.Tensor,
        incontext_length: int,
        t_span: torch.Tensor,
        mu: torch.Tensor,
        guidance_scale: float,
    ) -> torch.Tensor:
        """
        Fixed-step Euler ODE solver for flow-matching decoding.

        Args:
            x:                Noisy latent ``(B, T, dim)``.
            incontext_x:      In-context latent for conditioning ``(B, T, dim)``.
            incontext_length: Number of in-context frames.
            t_span:           Time steps ``(num_steps + 1,)``.
            mu:               Conditioning tensor from the codec encoder.
            guidance_scale:   Classifier-free guidance weight.
        """
        t = t_span[0]
        dt = t_span[1] - t_span[0]
        noise = x.clone()
        sol = []

        for step in tqdm(range(1, len(t_span)), disable=True):
            # Blend noise and in-context latent at the current timestep
            x[:, 0:incontext_length, :] = (
                (1 - (1 - 1e-6) * t) * noise[:, 0:incontext_length, :]
                + t * incontext_x[:, 0:incontext_length, :]
            )

            if guidance_scale > 1.0:
                # Classifier-free guidance: run conditioned and unconditioned in one batch
                dphi_dt = self.estimator(
                    torch.cat(
                        [
                            torch.cat([x, x], 0),
                            torch.cat([incontext_x, incontext_x], 0),
                            torch.cat([torch.zeros_like(mu), mu], 0),
                        ],
                        dim=2,
                    ),
                    timestep=t.unsqueeze(-1).repeat(2),
                )
                dphi_dt_uncond, dphi_dt_cond = dphi_dt.chunk(2, 0)
                dphi_dt = dphi_dt_uncond + guidance_scale * (
                    dphi_dt_cond - dphi_dt_uncond
                )
            else:
                dphi_dt = self.estimator(
                    torch.cat([x, incontext_x, mu], dim=2),
                    timestep=t.unsqueeze(-1),
                )

            x = x + dt * dphi_dt
            t = t + dt
            sol.append(x)
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

        return sol[-1]
