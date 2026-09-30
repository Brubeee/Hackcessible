"""Causal text-to-pose decoder for a future iSign sequence-model experiment.

The caller supplies frozen, token-level MiniLM hidden states and a padding mask.
Pose targets must already use the fit-only canonicalization and normalization
from the experiment manifest. The model returns normalized poses with shape
``(batch, frames, 75, 3)``. Training and inference are deliberately separate:
``forward_teacher`` consumes right-shifted ground-truth poses, while ``generate``
accepts no pose targets and autoregressively feeds its own previous predictions.

This module does not compute MiniLM features. The existing V5 cache contains
pooled 384D sentence embeddings, which cannot be passed off as token states.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class ProgressivePoseDecoder(nn.Module):
    """Two-layer causal pose decoder cross-attending to MiniLM token states.

    Parameters:
        text_dim: Width of frozen MiniLM token hidden states (384 for MiniLM-L6).
        model_dim: Transformer width; defaults to 256.
        num_heads: Attention heads; defaults to 8.
        num_layers: Causal decoder layers; defaults to 2.
        feedforward_dim: Inner width of each decoder layer.
        dropout: Dropout used during teacher-forced training.
        max_frames: Maximum generated sequence length.
        points: Number of body/hand points, in the caller's exported topology.
        coordinates: Coordinates per point (3 for xyz).

    `forward_teacher(text_hidden, text_mask, target_pose)` expects normalized
    target poses shaped ``(B,T,points*coordinates)`` or ``(B,T,points,coordinates)``.
    It returns ``(B,T,points,coordinates)``. At output frame t, the decoder can
    access ground-truth pose only through frame t-1.

    `generate(text_hidden, text_mask, frames)` expects the same token states and
    mask but no target poses. It returns normalized autoregressive predictions
    shaped ``(B,frames,points,coordinates)`` and requires the module in eval mode.
    """

    def __init__(
        self,
        text_dim: int = 384,
        model_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        feedforward_dim: int = 1024,
        dropout: float = 0.1,
        max_frames: int = 48,
        points: int = 75,
        coordinates: int = 3,
    ) -> None:
        super().__init__()
        if model_dim % num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        if min(text_dim, model_dim, num_heads, num_layers, feedforward_dim, max_frames, points, coordinates) < 1:
            raise ValueError("All model dimensions and frame/point counts must be positive")

        self.text_dim = int(text_dim)
        self.model_dim = int(model_dim)
        self.max_frames = int(max_frames)
        self.points = int(points)
        self.coordinates = int(coordinates)
        self.pose_dim = self.points * self.coordinates

        self.text_projection = nn.Linear(self.text_dim, self.model_dim)
        self.text_norm = nn.LayerNorm(self.model_dim)
        self.pose_projection = nn.Linear(self.pose_dim, self.model_dim)
        self.progress_projection = nn.Sequential(
            nn.Linear(1, self.model_dim),
            nn.GELU(),
            nn.Linear(self.model_dim, self.model_dim),
        )
        self.start_pose = nn.Parameter(torch.zeros(1, 1, self.pose_dim))
        self.input_dropout = nn.Dropout(dropout)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.model_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(self.model_dim),
        )
        self.pose_head = nn.Linear(self.model_dim, self.pose_dim)

        nn.init.normal_(self.start_pose, mean=0.0, std=0.02)

    def _validate_text(
        self, text_hidden: torch.Tensor, text_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if text_hidden.ndim != 3:
            raise ValueError("text_hidden must have shape (batch, tokens, text_dim)")
        if text_hidden.shape[-1] != self.text_dim:
            raise ValueError(f"Expected token width {self.text_dim}, got {text_hidden.shape[-1]}")
        if text_mask.shape != text_hidden.shape[:2]:
            raise ValueError("text_mask must have shape (batch, tokens)")
        if text_hidden.shape[0] < 1 or text_hidden.shape[1] < 1:
            raise ValueError("Text batch and token dimensions must be nonempty")
        if text_hidden.device != self.start_pose.device or text_mask.device != self.start_pose.device:
            raise ValueError("Move token states and mask to the same device as the model")
        mask = text_mask.to(dtype=torch.bool)
        if not mask.any(dim=1).all():
            raise ValueError("Every text example must contain at least one unmasked token")
        hidden = text_hidden.to(dtype=self.text_projection.weight.dtype)
        memory = self.text_norm(self.text_projection(hidden))
        return memory, ~mask

    def _progress(self, batch: int, frames: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if not 1 <= frames <= self.max_frames:
            raise ValueError(f"frames must be in [1, {self.max_frames}]")
        if frames == 1:
            values = torch.zeros(1, device=device, dtype=dtype)
        else:
            values = torch.linspace(0.0, 1.0, frames, device=device, dtype=dtype)
        return values.view(1, frames, 1).expand(batch, -1, -1)

    def _decode_prefix(
        self,
        memory: torch.Tensor,
        memory_padding_mask: torch.Tensor,
        pose_inputs: torch.Tensor,
    ) -> torch.Tensor:
        batch, frames, pose_dim = pose_inputs.shape
        if pose_dim != self.pose_dim:
            raise ValueError(f"Expected pose input width {self.pose_dim}, got {pose_dim}")
        if frames > self.max_frames:
            raise ValueError(f"Decoder prefix exceeds max_frames={self.max_frames}")
        progress = self._progress(batch, frames, pose_inputs.device, pose_inputs.dtype)
        decoder_inputs = self.pose_projection(pose_inputs) + self.progress_projection(progress)
        decoder_inputs = self.input_dropout(decoder_inputs)

        # Boolean True entries are blocked by PyTorch's Transformer mask API.
        causal_mask = torch.triu(
            torch.ones((frames, frames), dtype=torch.bool, device=pose_inputs.device),
            diagonal=1,
        )
        decoded = self.decoder(
            tgt=decoder_inputs,
            memory=memory,
            tgt_mask=causal_mask,
            memory_key_padding_mask=memory_padding_mask,
        )
        return self.pose_head(decoded)

    def _flatten_pose(self, target_pose: torch.Tensor) -> torch.Tensor:
        if target_pose.ndim == 4:
            if tuple(target_pose.shape[-2:]) != (self.points, self.coordinates):
                raise ValueError(
                    f"Structured target must end in ({self.points}, {self.coordinates})"
                )
            return target_pose.reshape(target_pose.shape[0], target_pose.shape[1], self.pose_dim)
        if target_pose.ndim == 3 and target_pose.shape[-1] == self.pose_dim:
            return target_pose
        raise ValueError(
            "target_pose must have shape (B,T,pose_dim) or (B,T,points,coordinates)"
        )

    def forward_teacher(
        self,
        text_hidden: torch.Tensor,
        text_mask: torch.Tensor,
        target_pose: torch.Tensor,
    ) -> torch.Tensor:
        """Decode with ground-truth history shifted one frame to the right."""
        target = self._flatten_pose(target_pose)
        batch, frames, _ = target.shape
        if batch != text_hidden.shape[0]:
            raise ValueError("Text and pose batch sizes do not match")
        if not 1 <= frames <= self.max_frames:
            raise ValueError(f"Target frame count must be in [1, {self.max_frames}]")
        target = target.to(dtype=self.start_pose.dtype)
        bos = self.start_pose.expand(batch, -1, -1)
        pose_inputs = torch.cat((bos, target[:, :-1]), dim=1)
        memory, memory_padding_mask = self._validate_text(text_hidden, text_mask)
        flat_prediction = self._decode_prefix(memory, memory_padding_mask, pose_inputs)
        return flat_prediction.reshape(batch, frames, self.points, self.coordinates)

    @torch.no_grad()
    def generate(
        self,
        text_hidden: torch.Tensor,
        text_mask: torch.Tensor,
        frames: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate a full sequence using only BOS and prior model outputs.

        The explicit eval-mode requirement ensures repeatability and prevents
        dropout from changing each autoregressive step. This method accepts no
        target pose argument and cannot teacher-force ground-truth frames.
        """
        if self.training:
            raise RuntimeError("Call model.eval() before autoregressive generation")
        frame_count = self.max_frames if frames is None else int(frames)
        if not 1 <= frame_count <= self.max_frames:
            raise ValueError(f"frames must be in [1, {self.max_frames}]")
        batch = text_hidden.shape[0]
        memory, memory_padding_mask = self._validate_text(text_hidden, text_mask)
        pose_inputs = self.start_pose.expand(batch, -1, -1)
        generated = []
        for _ in range(frame_count):
            flat_prefix = self._decode_prefix(memory, memory_padding_mask, pose_inputs)
            next_pose = flat_prefix[:, -1:, :]
            generated.append(next_pose[:, 0])
            if len(generated) < frame_count:
                pose_inputs = torch.cat((pose_inputs, next_pose), dim=1)
        flat_prediction = torch.stack(generated, dim=1)
        return flat_prediction.reshape(batch, frame_count, self.points, self.coordinates)


def run_cpu_smoke_checks() -> dict[str, float | tuple[int, ...]]:
    """Run deterministic shape, gradient, causality, padding, and free-run checks."""
    torch.manual_seed(2026)
    model = ProgressivePoseDecoder(dropout=0.0).cpu()
    batch, tokens = 2, 7
    text_hidden = torch.randn(batch, tokens, model.text_dim)
    text_mask = torch.tensor(
        [[1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 0, 0]], dtype=torch.bool
    )
    target = torch.randn(batch, model.max_frames, model.pose_dim)

    model.train()
    teacher = model.forward_teacher(text_hidden, text_mask, target)
    expected_shape = (batch, model.max_frames, model.points, model.coordinates)
    assert tuple(teacher.shape) == expected_shape
    assert torch.isfinite(teacher).all()
    (teacher.square().mean()).backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e6)
    assert torch.isfinite(gradient_norm) and float(gradient_norm) > 0

    # Changing target frames from cutoff onward cannot affect outputs through cutoff.
    model.eval()
    with torch.no_grad():
        reference_teacher = model.forward_teacher(text_hidden, text_mask, target)
        cutoff = 19
        changed_future = target.clone()
        changed_future[:, cutoff:] = torch.randn_like(changed_future[:, cutoff:]) * 100.0
        changed_teacher = model.forward_teacher(text_hidden, text_mask, changed_future)
        assert torch.allclose(
            reference_teacher[:, : cutoff + 1],
            changed_teacher[:, : cutoff + 1],
            atol=1e-6,
            rtol=1e-5,
        ), "Future target poses leaked into earlier teacher-forced outputs."

        # Padded token values are irrelevant when their mask positions are false.
        changed_padding = text_hidden.clone()
        changed_padding[~text_mask] = torch.randn_like(changed_padding[~text_mask]) * 1000.0
        padding_teacher = model.forward_teacher(changed_padding, text_mask, target)
        assert torch.allclose(reference_teacher, padding_teacher, atol=1e-6, rtol=1e-5), (
            "Masked text padding affected decoder outputs."
        )

        free_run = model.generate(text_hidden, text_mask, frames=model.max_frames)
        repeated_free_run = model.generate(text_hidden, text_mask, frames=model.max_frames)
        assert tuple(free_run.shape) == expected_shape
        assert torch.isfinite(free_run).all()
        assert torch.allclose(free_run, repeated_free_run, atol=0.0, rtol=0.0), (
            "Eval-mode free-run generation is not repeatable."
        )

    return {
        "gradient_norm": float(gradient_norm),
        "teacher_shape": tuple(teacher.shape),
        "free_run_shape": tuple(free_run.shape),
        "future_target_max_delta": float(
            (reference_teacher[:, : cutoff + 1] - changed_teacher[:, : cutoff + 1])
            .abs()
            .max()
        ),
        "padding_max_delta": float((reference_teacher - padding_teacher).abs().max()),
        "repeatability_max_delta": float((free_run - repeated_free_run).abs().max()),
    }


if __name__ == "__main__":
    print("ProgressivePoseDecoder CPU smoke checks:", run_cpu_smoke_checks())
