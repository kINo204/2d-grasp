from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ContactFeature:
    contact: bool
    point: np.ndarray
    force_dir: np.ndarray
    inward_alignment: float
    normal_force: float
    slip: float


@dataclass
class RewardBreakdown:
    total: float
    contact: float
    inward: float
    wrench: float
    still: float
    lift: float
    reg: float
    terminal: float
    stability_score: float


def _safe_norm(vec: np.ndarray, eps: float = 1e-6) -> float:
    return float(np.sqrt(np.dot(vec, vec) + eps))


def _wrench_proxy(
    contacts: list[ContactFeature],
    object_pos: np.ndarray,
) -> float:
    active = [c for c in contacts if c.contact]
    if len(active) < 2:
        return 0.0
    torques: list[float] = []
    dirs: list[np.ndarray] = []
    for c in active:
        r = c.point - object_pos
        f = c.force_dir
        torques.append(float(r[0] * f[1] - r[1] * f[0]))
        dirs.append(f / _safe_norm(f))
    torque_span = max(torques) - min(torques)
    torque_score = float(np.tanh(3.0 * torque_span))
    pairwise: list[float] = []
    for i in range(len(dirs)):
        for j in range(i + 1, len(dirs)):
            pairwise.append(float((1.0 - np.dot(dirs[i], dirs[j])) * 0.5))
    diversity = float(np.mean(pairwise)) if pairwise else 0.0
    return float(np.clip(0.6 * torque_score + 0.4 * diversity, 0.0, 1.0))


def compute_reward(
    contacts: list[ContactFeature],
    object_pos: np.ndarray,
    object_lin_vel: np.ndarray,
    object_ang_vel: float,
    lift_progress: float,
    action: np.ndarray,
    prev_action: np.ndarray,
    reward_weights: dict[str, float],
    success: bool,
    dropped: bool,
    expected_contacts: int | None = None,
) -> RewardBreakdown:
    if expected_contacts is None:
        expected_contacts = max(len(contacts), 1)
    contact_count = sum(int(c.contact) for c in contacts)
    r_contact = float(np.clip(contact_count / float(max(expected_contacts, 1)), 0.0, 1.0))

    inward_scores: list[float] = []
    slips: list[float] = []
    for c in contacts:
        if not c.contact:
            continue
        inward_scores.append(
            float(max(c.inward_alignment, 0.0) * np.tanh(0.1 * c.normal_force))
        )
        slips.append(c.slip)
    r_inward = float(np.mean(inward_scores)) if inward_scores else 0.0
    r_wrench = _wrench_proxy(contacts, object_pos)

    speed_penalty = float(np.linalg.norm(object_lin_vel) + 0.25 * abs(object_ang_vel))
    if contact_count >= 2:
        slip_penalty = float(np.mean(slips)) if slips else 0.0
        r_still = float(np.exp(-3.0 * speed_penalty)) * float(np.exp(-2.0 * slip_penalty))
    else:
        r_still = 0.0

    r_lift = float(np.clip(lift_progress, 0.0, 1.0))
    action_mag = float(np.mean(np.square(action)))
    action_delta = float(np.mean(np.square(action - prev_action)))
    r_reg = action_mag + 0.5 * action_delta

    terminal = 0.0
    if success:
        terminal += 1.0
    if dropped:
        terminal -= 0.5

    total = (
        reward_weights["w_contact"] * r_contact
        + reward_weights["w_inward"] * r_inward
        + reward_weights["w_wrench"] * r_wrench
        + reward_weights["w_still"] * r_still
        + reward_weights["w_lift"] * r_lift
        - reward_weights["w_reg"] * r_reg
        + terminal
    )
    stability_score = float(np.clip(0.4 * r_inward + 0.6 * r_wrench, 0.0, 1.0))
    return RewardBreakdown(
        total=float(total),
        contact=r_contact,
        inward=r_inward,
        wrench=r_wrench,
        still=r_still,
        lift=r_lift,
        reg=r_reg,
        terminal=terminal,
        stability_score=stability_score,
    )
