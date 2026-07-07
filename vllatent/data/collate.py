"""Collate function for batched sports training (TORCH tier) — B1.14.

Converts numpy ``SportsSample`` instances to batched GPU tensors via a
``TrainingBatch`` NamedTuple. Used as ``collate_fn`` for
``torch.utils.data.DataLoader``.

torch is imported LAZILY (inside functions) so this module imports on a
torch-free box.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np

if TYPE_CHECKING:
    import torch

    from vllatent.data.sports_loader import SportsSample


class TrainingBatch(NamedTuple):
    """Batched GPU tensors for one training step."""

    z_t: torch.Tensor              # (B, P, D) fp16
    history_latents: torch.Tensor  # (B, H, P, D) fp16
    history_mask: torch.Tensor     # (B, H) bool
    target_latents: torch.Tensor   # (B, T, P, D) fp16
    history_person_bbox: torch.Tensor     # (B, H, 4) f32
    history_person_visible: torch.Tensor  # (B, H) bool
    history_person_state_valid: torch.Tensor  # (B, H) bool
    history_person_conf: torch.Tensor     # (B, H) f32
    target_person_bbox: torch.Tensor      # (B, T, 4) f32
    target_person_visible: torch.Tensor   # (B, T) bool
    target_person_state_valid: torch.Tensor  # (B, T) bool
    target_person_conf: torch.Tensor      # (B, T) f32
    person_state_target: torch.Tensor     # (B, T, 4) f32
    target_deltas: torch.Tensor    # (B, T, 4) f32
    last_action: torch.Tensor      # (B, 4) f32 — most recent known action (FiLM conditioning)
    planned_actions: torch.Tensor  # (B, T, 6) f32 — B3 candidate/teacher-forced plan input
    planned_actions_valid_mask: torch.Tensor  # (B, T) bool
    vo_confidence: torch.Tensor    # (B, T) f32
    frame_quality: torch.Tensor    # (B,) f32
    dt_seconds: torch.Tensor       # (B, T) f32
    sample_weight: torch.Tensor    # (B,) f32


class ActionPolicyBatch(NamedTuple):
    """Batched tensors for the B2 direct scale-free action policy."""

    z_t: torch.Tensor                         # (B, P, D) fp16
    history_latents: torch.Tensor             # (B, H, P, D) fp16
    history_mask: torch.Tensor                # (B, H) bool
    target_latents: torch.Tensor              # (B, T, P, D) fp16, optional WAM auxiliary target
    target_actions_scale_free: torch.Tensor   # (B, T, 4) f32
    target_actions_moving_mask: torch.Tensor  # (B, T) bool
    target_actions_speed_mask: torch.Tensor   # (B, T) bool
    last_action_scale_free: torch.Tensor      # (B, 4) f32
    action_history_scale_free: torch.Tensor   # (B, H, 4) f32
    action_history_mask: torch.Tensor         # (B, H) bool
    camera_history_path_scale_free: torch.Tensor  # (B, H, 3) f32
    dt_seconds: torch.Tensor                  # (B, T) f32
    odom_reference_speed: torch.Tensor        # (B,) f32, diagnostic/reference only
    vo_confidence: torch.Tensor               # (B, T) f32
    frame_quality: torch.Tensor               # (B,) f32
    sample_weight: torch.Tensor               # (B,) f32


def collate_sports_batch(samples: list[SportsSample]) -> TrainingBatch:
    """Collate a list of SportsSample into a TrainingBatch of tensors."""
    import torch

    z_t = torch.from_numpy(np.stack([s.z_t for s in samples]))
    history_latents = torch.from_numpy(np.stack([s.history_latents for s in samples]))
    history_mask = torch.from_numpy(np.stack([s.history_mask for s in samples]))
    target_latents = torch.from_numpy(np.stack([s.target_latents for s in samples]))
    history_person_bbox = torch.from_numpy(np.stack([s.history_person_bbox for s in samples]))
    history_person_visible = torch.from_numpy(np.stack([s.history_person_visible for s in samples]))
    history_person_state_valid = torch.from_numpy(np.stack([s.history_person_state_valid for s in samples]))
    history_person_conf = torch.from_numpy(np.stack([s.history_person_conf for s in samples]))
    target_person_bbox = torch.from_numpy(np.stack([s.target_person_bbox for s in samples]))
    target_person_visible = torch.from_numpy(np.stack([s.target_person_visible for s in samples]))
    target_person_state_valid = torch.from_numpy(np.stack([s.target_person_state_valid for s in samples]))
    target_person_conf = torch.from_numpy(np.stack([s.target_person_conf for s in samples]))
    person_state_target = torch.from_numpy(np.stack([s.person_state_target for s in samples]))
    target_deltas = torch.from_numpy(np.stack([s.target_deltas for s in samples]))
    last_action = torch.from_numpy(np.stack([s.last_action for s in samples]))
    planned_actions = torch.from_numpy(np.stack([s.planned_actions for s in samples]))
    planned_actions_valid_mask = torch.from_numpy(np.stack([s.planned_actions_valid_mask for s in samples]))
    vo_conf = torch.from_numpy(np.stack([s.vo_confidence for s in samples]))
    dt_sec = torch.from_numpy(np.stack([s.dt_seconds for s in samples]))

    fq = torch.tensor([s.frame_quality for s in samples], dtype=torch.float32)

    weight = fq.clamp(min=0.1) * vo_conf.mean(dim=1).clamp(min=0.05)

    return TrainingBatch(
        z_t=z_t,
        history_latents=history_latents,
        history_mask=history_mask,
        target_latents=target_latents,
        history_person_bbox=history_person_bbox,
        history_person_visible=history_person_visible,
        history_person_state_valid=history_person_state_valid,
        history_person_conf=history_person_conf,
        target_person_bbox=target_person_bbox,
        target_person_visible=target_person_visible,
        target_person_state_valid=target_person_state_valid,
        target_person_conf=target_person_conf,
        person_state_target=person_state_target,
        target_deltas=target_deltas,
        last_action=last_action,
        planned_actions=planned_actions,
        planned_actions_valid_mask=planned_actions_valid_mask,
        vo_confidence=vo_conf,
        frame_quality=fq,
        dt_seconds=dt_sec,
        sample_weight=weight,
    )


def collate_action_policy_batch(samples: list[SportsSample]) -> ActionPolicyBatch:
    """Collate SportsSample objects for B2 action-policy training."""
    import torch

    z_t = torch.from_numpy(np.stack([s.z_t for s in samples]))
    history_latents = torch.from_numpy(np.stack([s.history_latents for s in samples]))
    history_mask = torch.from_numpy(np.stack([s.history_mask for s in samples]))
    target_latents = torch.from_numpy(np.stack([s.target_latents for s in samples]))
    target_actions = torch.from_numpy(np.stack([s.target_actions_scale_free for s in samples]))
    target_mask = torch.from_numpy(np.stack([s.target_actions_moving_mask for s in samples]))
    target_speed_mask = torch.from_numpy(np.stack([s.target_actions_speed_mask for s in samples]))
    last_action = torch.from_numpy(np.stack([s.last_action_scale_free for s in samples]))
    action_history = torch.from_numpy(np.stack([s.action_history_scale_free for s in samples]))
    action_history_mask = torch.from_numpy(np.stack([s.action_history_mask for s in samples]))
    camera_history_path = torch.from_numpy(np.stack([s.camera_history_path_scale_free for s in samples]))
    dt_sec = torch.from_numpy(np.stack([s.dt_seconds for s in samples]))
    vo_conf = torch.from_numpy(np.stack([s.vo_confidence for s in samples]))

    fq = torch.tensor([s.frame_quality for s in samples], dtype=torch.float32)
    odom_ref = torch.tensor([s.odom_reference_speed for s in samples], dtype=torch.float32)
    weight = fq.clamp(min=0.1) * vo_conf.mean(dim=1).clamp(min=0.05)

    return ActionPolicyBatch(
        z_t=z_t,
        history_latents=history_latents,
        history_mask=history_mask,
        target_latents=target_latents,
        target_actions_scale_free=target_actions,
        target_actions_moving_mask=target_mask,
        target_actions_speed_mask=target_speed_mask,
        last_action_scale_free=last_action,
        action_history_scale_free=action_history,
        action_history_mask=action_history_mask,
        camera_history_path_scale_free=camera_history_path,
        dt_seconds=dt_sec,
        odom_reference_speed=odom_ref,
        vo_confidence=vo_conf,
        frame_quality=fq,
        sample_weight=weight,
    )
