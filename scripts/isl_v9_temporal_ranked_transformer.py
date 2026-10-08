"""Non-autoregressive ISL text-to-pose decoder with a calibrated, text-ranked V9 objective.

The decoder architecture is retained from V7 for a controlled same-role loss experiment.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn


def sinusoidal_pe(length: int, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Generate sinusoidal positional encodings directly on target device."""
    position = torch.arange(0, length, dtype=dtype, device=device).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, dtype=dtype, device=device) * (-math.log(10000.0) / dim)
    )
    pe = torch.zeros(1, length, dim, dtype=dtype, device=device)
    pe[0, :, 0::2] = torch.sin(position * div_term)
    pe[0, :, 1::2] = torch.cos(position * div_term)
    return pe


class MotionTransformerDecoder(nn.Module):
    """Non-autoregressive pose generator conditioned on MiniLM token embeddings.

    Parameters:
        text_dim: Width of frozen MiniLM token hidden states (384 for MiniLM-L6).
        model_dim: Transformer latent width (defaults to 256).
        num_heads: Attention heads (defaults to 8).
        num_layers: Decoder layers (defaults to 4).
        feedforward_dim: Feedforward hidden width (defaults to 1024).
        dropout: Dropout rate (defaults to 0.1).
        max_frames: Target sequence length (48 frames).
        points: Topology joint count (75 points: 33 body, 21 left hand, 21 right hand).
        coordinates: Dimensions per joint (3 for XYZ).
    """

    def __init__(
        self,
        text_dim: int = 384,
        model_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        feedforward_dim: int = 1024,
        dropout: float = 0.1,
        max_frames: int = 48,
        points: int = 75,
        coordinates: int = 3,
    ) -> None:
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError(f"model_dim ({model_dim}) must be divisible by num_heads ({num_heads})")
        if min(text_dim, model_dim, num_heads, num_layers, feedforward_dim, max_frames, points, coordinates) < 1:
            raise ValueError("All dimensions and counts must be positive integers")

        self.text_dim = int(text_dim)
        self.model_dim = int(model_dim)
        self.max_frames = int(max_frames)
        self.points = int(points)
        self.coordinates = int(coordinates)
        self.pose_dim = self.points * self.coordinates

        # Text projection & normalization
        self.text_projection = nn.Linear(self.text_dim, self.model_dim)
        self.text_norm = nn.LayerNorm(self.model_dim)

        # Learned frame queries & progress projection
        self.frame_queries = nn.Parameter(torch.zeros(1, self.max_frames, self.model_dim))
        self.progress_projection = nn.Sequential(
            nn.Linear(1, self.model_dim),
            nn.GELU(),
            nn.Linear(self.model_dim, self.model_dim),
        )

        self.query_dropout = nn.Dropout(dropout)

        # Non-autoregressive Transformer Decoder with bidirectional self-attention
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

        # Output projection head: latent -> pose coordinates
        self.pose_head = nn.Sequential(
            nn.Linear(self.model_dim, self.model_dim),
            nn.GELU(),
            nn.Linear(self.model_dim, self.pose_dim),
        )

        nn.init.normal_(self.frame_queries, mean=0.0, std=0.02)

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
        mask = text_mask.to(dtype=torch.bool)
        if not mask.any(dim=1).all():
            raise ValueError("Every text example must contain at least one unmasked token")

        hidden = text_hidden.to(dtype=self.text_projection.weight.dtype)
        memory = self.text_norm(self.text_projection(hidden))

        # Additive FP32 attention bias: 0 allows, -inf blocks.
        memory_padding_bias = torch.where(
            mask,
            torch.zeros((), dtype=memory.dtype, device=memory.device),
            torch.full((), float("-inf"), dtype=memory.dtype, device=memory.device),
        ).contiguous()
        return memory, memory_padding_bias

    def forward(
        self,
        text_hidden: torch.Tensor,
        text_mask: torch.Tensor,
        frames: Optional[int] = None,
    ) -> torch.Tensor:
        """Decode all frames non-autoregressively in a single forward pass.

        Args:
            text_hidden: MiniLM token hidden states (batch, tokens, text_dim).
            text_mask: Boolean attention mask (batch, tokens).
            frames: Optional target frame count (defaults to max_frames=48).

        Returns:
            Normalized pose sequence (batch, frames, points, coordinates).
        """
        frame_count = self.max_frames if frames is None else int(frames)
        if not 1 <= frame_count <= self.max_frames:
            raise ValueError(f"frames must be in [1, {self.max_frames}]")

        batch = text_hidden.shape[0]
        memory, memory_padding_mask = self._validate_text(text_hidden, text_mask)

        # Prepare temporal queries for all frames
        queries = self.frame_queries[:, :frame_count].expand(batch, -1, -1)
        pe = sinusoidal_pe(frame_count, self.model_dim, queries.device, queries.dtype)
        queries = queries + pe

        # Add normalized time progress in [0, 1]
        progress_values = (
            torch.linspace(0.0, 1.0, frame_count, device=queries.device, dtype=queries.dtype)
            .view(1, frame_count, 1)
            .expand(batch, -1, -1)
        )
        queries = queries + self.progress_projection(progress_values)
        queries = self.query_dropout(queries)

        # Non-autoregressive decoding with explicit bidirectional self-attention
        tgt_mask = torch.zeros((frame_count, frame_count), dtype=queries.dtype, device=queries.device)
        decoded = self.decoder(
            tgt=queries,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_padding_mask,
        )

        flat_pose = self.pose_head(decoded)
        return flat_pose.reshape(batch, frame_count, self.points, self.coordinates)


def _grouped_sample_scores(
    prediction: torch.Tensor,
    target: torch.Tensor,
    confidence: torch.Tensor,
    groups: dict[str, tuple[int, int, float]],
    velocity_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Return coordinate-mean position/velocity scores, balanced by body/hand group."""
    conf = torch.nan_to_num(confidence, nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)
    position_by_group: dict[str, torch.Tensor] = {}
    velocity_by_group: dict[str, torch.Tensor] = {}
    batch = prediction.shape[0]
    for name, (start, end, group_weight) in groups.items():
        pos_error = (prediction[:, :, start:end] - target[:, :, start:end]).square()
        pos_weight = conf[:, :, start:end, None].expand_as(pos_error)
        pos_num = (pos_error * pos_weight).sum(dim=(1, 2, 3))
        pos_den = pos_weight.sum(dim=(1, 2, 3)).clamp_min(1.0)
        position_by_group[name] = group_weight * pos_num / pos_den

        pred_velocity = prediction[:, 1:, start:end] - prediction[:, :-1, start:end]
        true_velocity = target[:, 1:, start:end] - target[:, :-1, start:end]
        vel_error = (pred_velocity - true_velocity).square()
        pair_conf = torch.minimum(conf[:, 1:, start:end], conf[:, :-1, start:end])
        vel_weight = pair_conf[..., None].expand_as(vel_error)
        vel_num = (vel_error * vel_weight).sum(dim=(1, 2, 3))
        vel_den = vel_weight.sum(dim=(1, 2, 3)).clamp_min(1.0)
        velocity_by_group[name] = group_weight * vel_num / vel_den

    pos_per_sample = torch.stack(list(position_by_group.values())).sum(dim=0)
    vel_per_sample = torch.stack(list(velocity_by_group.values())).sum(dim=0)
    score_per_sample = pos_per_sample + velocity_weight * vel_per_sample
    return pos_per_sample, vel_per_sample, score_per_sample, {
        'position_by_group': position_by_group,
        'velocity_by_group': velocity_by_group,
    }


def _grouped_temporal_scores(
    prediction: torch.Tensor,
    target: torch.Tensor,
    confidence: torch.Tensor,
    groups: dict[str, tuple[int, int, float]],
    phase_weight: float = 0.25,
    velocity_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Return signed-XYZ velocity MSE and a scale-normalized speed-profile loss.

    The phase term compares aligned temporal profiles after normalizing the
    predicted and reference group-speed curves independently. It therefore
    penalizes timing/profile mismatch without rewarding larger speed.
    """
    conf = torch.nan_to_num(confidence, nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)
    velocity_by_group: dict[str, torch.Tensor] = {}
    phase_by_group: dict[str, torch.Tensor] = {}
    for name, (start, end, group_weight) in groups.items():
        pred_velocity = prediction[:, 1:, start:end] - prediction[:, :-1, start:end]
        true_velocity = target[:, 1:, start:end] - target[:, :-1, start:end]
        pair_conf = torch.minimum(conf[:, 1:, start:end], conf[:, :-1, start:end])

        velocity_error = (pred_velocity - true_velocity).square()
        velocity_weights = pair_conf[..., None].expand_as(velocity_error)
        velocity_by_group[name] = group_weight * (
            (velocity_error * velocity_weights).sum(dim=(1, 2, 3))
            / velocity_weights.sum(dim=(1, 2, 3)).clamp_min(1.0)
        )

        pred_speed = torch.linalg.vector_norm(pred_velocity, dim=-1)
        true_speed = torch.linalg.vector_norm(true_velocity, dim=-1)
        frame_support = pair_conf.sum(dim=2)
        frame_weights = frame_support.to(pred_speed.dtype)
        pred_curve = (pred_speed * pair_conf).sum(dim=2) / frame_support.clamp_min(1e-6)
        true_curve = (true_speed * pair_conf).sum(dim=2) / frame_support.clamp_min(1e-6)
        valid_frames = frame_weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        pred_mean_speed = (pred_curve * frame_weights).sum(dim=1, keepdim=True) / valid_frames
        true_mean_speed = (true_curve * frame_weights).sum(dim=1, keepdim=True) / valid_frames
        pred_profile = pred_curve / (pred_mean_speed + 1e-6)
        true_profile = true_curve / (true_mean_speed + 1e-6)
        phase_error = (pred_profile - true_profile).abs()
        phase_by_group[name] = group_weight * (
            (phase_error * frame_weights).sum(dim=1) / frame_weights.sum(dim=1).clamp_min(1e-6)
        )

    velocity_per_sample = torch.stack(list(velocity_by_group.values())).sum(dim=0)
    phase_per_sample = torch.stack(list(phase_by_group.values())).sum(dim=0)
    dynamics_per_sample = velocity_weight * velocity_per_sample + phase_weight * phase_per_sample
    return velocity_per_sample, phase_per_sample, dynamics_per_sample, {
        'velocity_by_group': velocity_by_group,
        'phase_by_group': phase_by_group,
    }


def compute_motion_loss_components(
    prediction: torch.Tensor,
    target: torch.Tensor,
    confidence: torch.Tensor,
    groups: dict[str, tuple[int, int, float]],
    negative_prediction: Optional[torch.Tensor] = None,
    ranking_mask: Optional[torch.Tensor] = None,
    velocity_weight: float = 1.0,
    phase_weight: float = 0.25,
    rank_margin: float = 0.01,
) -> dict[str, torch.Tensor]:
    """Build the V9 matched pose, temporal, and paired-ranking components."""
    position_per_sample, _, _, _ = _grouped_sample_scores(
        prediction, target, confidence, groups, velocity_weight=0.0
    )
    velocity_per_sample, phase_per_sample, dynamics_per_sample, _ = _grouped_temporal_scores(
        prediction, target, confidence, groups, phase_weight=phase_weight,
        velocity_weight=velocity_weight
    )
    rank_per_sample = dynamics_per_sample.new_zeros(dynamics_per_sample.shape)
    rank_valid_mask = torch.zeros_like(dynamics_per_sample, dtype=torch.bool)
    if negative_prediction is not None:
        _, _, negative_dynamics_per_sample, _ = _grouped_temporal_scores(
            negative_prediction, target, confidence, groups, phase_weight=phase_weight,
            velocity_weight=velocity_weight
        )
        rank_per_sample = torch.relu(
            rank_margin + dynamics_per_sample - negative_dynamics_per_sample
        )
        if ranking_mask is not None:
            rank_valid_mask = ranking_mask.to(device=rank_per_sample.device, dtype=torch.bool)
            if rank_valid_mask.shape != rank_per_sample.shape:
                raise ValueError('V9 ranking mask must have one boolean per paired target.')
        else:
            rank_valid_mask = torch.ones_like(rank_per_sample, dtype=torch.bool)
    return {
        'pose': position_per_sample.mean(),
        'velocity': velocity_per_sample.mean(),
        'phase_profile': phase_per_sample.mean(),
        'dynamics': dynamics_per_sample.mean(),
        'rank': (rank_per_sample * rank_valid_mask.to(rank_per_sample.dtype)).sum()
                / rank_valid_mask.sum().clamp_min(1),
        'rank_win_fraction': (
            ((rank_per_sample < rank_margin) & rank_valid_mask).to(rank_per_sample.dtype).sum()
            / rank_valid_mask.sum().clamp_min(1)
        ),
        'valid_rank_pairs': rank_valid_mask.sum(),
    }


def compute_motion_losses(
    prediction: torch.Tensor,
    target: torch.Tensor,
    confidence: torch.Tensor,
    groups: dict[str, tuple[int, int, float]],
    negative_prediction: Optional[torch.Tensor] = None,
    ranking_mask: Optional[torch.Tensor] = None,
    velocity_weight: float = 1.0,
    rank_weight: float = 0.20,
    rank_margin: float = 0.01,
    phase_weight: float = 0.25,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Use one matched pose-plus-motion score for fit, stop, and selection.

    V9 minimizes confidence-weighted coordinate-mean pose error plus signed
    XYZ velocity error and a scale-normalized temporal-profile auxiliary. Its
    caption hinge compares only the temporal error for two captions on the
    same target, preventing the static pose prior from dominating ranking.
    """
    components = compute_motion_loss_components(
        prediction, target, confidence, groups, negative_prediction, ranking_mask,
        velocity_weight=velocity_weight, phase_weight=phase_weight, rank_margin=rank_margin,
    )
    if negative_prediction is not None and int(components['valid_rank_pairs'].detach()) <= 0:
        raise ValueError('V9 temporal ranking requires at least one valid true/negative caption pair.')
    rank_loss = components['rank']
    total_loss = components['pose'] + components['dynamics'] + rank_weight * rank_loss
    metrics = {
        'loss': float(total_loss.detach().item()),
        'position_mse': float(components['pose'].detach().item()),
        'velocity_mse': float(components['velocity'].detach().item()),
        'phase_profile_error': float(components['phase_profile'].detach().item()),
        'dynamic_error': float(components['dynamics'].detach().item()),
        'selection_score': float((components['pose'] + components['dynamics'] + rank_weight * rank_loss).detach().item()),
        'ranking_loss': float(rank_loss.detach().item()),
        'rank_true_win_fraction': float(components['rank_win_fraction'].detach().item()),
        'valid_rank_pairs': int(components['valid_rank_pairs'].detach().item()),
    }
    _, _, _, by_group = _grouped_sample_scores(prediction, target, confidence, groups, 0.0)
    _, _, _, temporal_by_group = _grouped_temporal_scores(prediction, target, confidence, groups, phase_weight)
    for name, value in by_group['position_by_group'].items():
        metrics[f'position_{name}'] = float(value.mean().detach().item())
    for name, value in temporal_by_group['velocity_by_group'].items():
        metrics[f'velocity_{name}'] = float(value.mean().detach().item())
    for name, value in temporal_by_group['phase_by_group'].items():
        metrics[f'phase_{name}'] = float(value.mean().detach().item())
    return total_loss, metrics


def run_cpu_smoke_checks() -> dict[str, float | tuple[int, ...]]:
    """Run deterministic CPU contracts for V9 decoding, losses, and controls."""
    torch.manual_seed(2026)
    model = MotionTransformerDecoder(dropout=0.0).cpu()
    assert isinstance(model.pose_head, nn.Sequential)
    final_pose_projection = next(
        (layer for layer in reversed(model.pose_head) if isinstance(layer, nn.Linear)),
        None,
    )
    assert final_pose_projection is not None, 'V9 pose head must end with a Linear projection.'
    assert isinstance(model.pose_head[-1], nn.Linear), 'Final V9 pose-head module must be Linear.'
    assert tuple(final_pose_projection.weight.shape) == (model.pose_dim, model.model_dim), (
        'V9 final pose projection must map model_dim to pose_dim.'
    )
    batch, tokens = 4, 7
    text_hidden = torch.randn(batch, tokens, model.text_dim)
    text_mask = torch.tensor(
        [[1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 0, 0],
         [1, 1, 1, 1, 1, 1, 0], [1, 1, 1, 0, 0, 0, 0]], dtype=torch.bool
    )
    target = torch.randn(batch, model.max_frames, model.points, model.coordinates)
    confidence = torch.rand(batch, model.max_frames, model.points)
    groups = {'body': (0, 33, 0.20), 'left_hand': (33, 54, 0.40), 'right_hand': (54, 75, 0.40)}

    model.train()
    output = model(text_hidden, text_mask)
    negative = model(torch.roll(text_hidden, 1, 0), torch.roll(text_mask, 1, 0))
    expected_shape = (batch, model.max_frames, model.points, model.coordinates)
    assert tuple(output.shape) == expected_shape and torch.isfinite(output).all()
    loss, metrics = compute_motion_losses(output, target, confidence, groups, negative_prediction=negative)
    assert torch.isfinite(loss) and metrics['ranking_loss'] >= 0.0
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    assert torch.isfinite(grad_norm) and float(grad_norm) > 0

    model.eval()
    with torch.no_grad():
        ref_out = model(text_hidden, text_mask)
        corrupted_hidden = text_hidden.clone()
        corrupted_hidden[~text_mask] = torch.randn_like(corrupted_hidden[~text_mask]) * 1000.0
        padding_out = model(corrupted_hidden, text_mask)
        padding_delta = float((ref_out - padding_out).abs().max())
        assert padding_delta < 1e-6
        repeated_out = model(text_hidden, text_mask)
        repeat_delta = float((ref_out - repeated_out).abs().max())
        assert repeat_delta == 0.0
        null_hidden = torch.zeros(batch, 1, model.text_dim)
        null_mask = torch.ones(batch, 1, dtype=torch.bool)
        null_out = model(null_hidden, null_mask)
        assert tuple(null_out.shape) == expected_shape and torch.isfinite(null_out).all()

    # Verify the objective is a coordinate mean: repeating XYZ channels leaves it unchanged.
    tiny_pred = torch.zeros(2, 3, 2, 3)
    tiny_true = torch.ones_like(tiny_pred)
    tiny_conf = torch.ones(2, 3, 2)
    tiny_groups = {'all': (0, 2, 1.0)}
    pos, _, score, _ = _grouped_sample_scores(tiny_pred, tiny_true, tiny_conf, tiny_groups, 0.0)
    expected = torch.ones(2)
    assert torch.allclose(pos, expected) and torch.allclose(score, expected)

    # Signed velocity and same-target temporal ranking contracts.
    toy_target = torch.zeros(2, 4, 2, 3)
    toy_target[:, :, :, 0] = torch.arange(4, dtype=torch.float32)[None, :, None]
    toy_conf = torch.ones(2, 4, 2)
    toy_groups = {'all': (0, 2, 1.0)}
    exact = toy_target.clone()
    stationary = torch.zeros_like(toy_target)
    exact_loss, exact_metrics = compute_motion_losses(
        exact, toy_target, toy_conf, toy_groups, negative_prediction=stationary,
        velocity_weight=1.0, phase_weight=0.25, rank_weight=0.20, rank_margin=0.01,
    )
    reversed_motion = -toy_target
    reversed_components = compute_motion_loss_components(
        reversed_motion, toy_target, toy_conf, toy_groups,
        negative_prediction=exact, velocity_weight=1.0, phase_weight=0.25,
        rank_margin=0.01,
    )
    assert exact_metrics['velocity_mse'] == 0.0 and exact_metrics['ranking_loss'] == 0.0
    assert float(reversed_components['velocity'].detach()) > 0.0
    assert float(reversed_components['rank'].detach()) > exact_metrics['ranking_loss']
    assert torch.isfinite(exact_loss)

    phase_target = torch.zeros(1, 4, 1, 3)
    phase_target[0, :, 0, 0] = torch.tensor([0.0, 1.0, 2.0, 3.0])
    phase_prediction = torch.zeros_like(phase_target)
    phase_prediction[0, :, 0, 0] = torch.tensor([0.0, 1.0, 3.0, 4.0])
    full_support = torch.ones(1, 4, 1)
    sparse_support = full_support.clone(); sparse_support[:, 1] = 0.01
    phase_groups = {'all': (0, 1, 1.0)}
    _, full_phase, _, _ = _grouped_temporal_scores(
        phase_prediction, phase_target, full_support, phase_groups
    )
    _, sparse_phase, _, _ = _grouped_temporal_scores(
        phase_prediction, phase_target, sparse_support, phase_groups
    )
    assert float(sparse_phase.mean()) < float(full_phase.mean()), \
        'Sparse frames must contribute in proportion to their confidence support.'
    return {
        'output_shape': tuple(output.shape), 'loss': float(loss.detach().item()),
        'grad_norm': float(grad_norm), 'padding_delta': padding_delta,
        'repeat_delta': repeat_delta, 'null_output_finite': 1.0,
        'coordinate_mean_fixture': float(pos.mean().item()),
        'ranking_loss': metrics['ranking_loss'],
        'signed_velocity_fixture':float(reversed_components['velocity'].detach().item()),
        'same_target_temporal_hinge_fixture':float(reversed_components['rank'].detach().item()),
        'full_support_phase_fixture':float(full_phase.mean().item()),
        'sparse_support_phase_fixture':float(sparse_phase.mean().item()),
    }


if __name__ == '__main__':
    results = run_cpu_smoke_checks()
    print('V9 Motion Transformer CPU preflight passed:', results)
