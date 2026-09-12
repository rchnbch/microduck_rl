"""Trajectory statistics for a LEARNED behaviour space, and a gate with no names in it.

v4 named its modes in advance (``walk / crawl / hop / roll / other``), and then
measured that the search contributed zero of them: mode count equalled seed
count, and ~10 % of otherwise-viable candidates died to label instability. v5
stops naming. Two things replace the classifier:

1. **A feature vector per rollout that describes what the body did without
   saying what it was** (:class:`BehaviourStats`). An encoder is trained on
   these (:mod:`qd.behaviour_space`) and the archive lives in its latent
   space, so the modes are whatever the data separates.

2. **A viability predicate with no label clause** (:func:`evaluate_viability_v5`).
   P2' kept "one mode label across windows" to close the end-of-episode hole
   (walk 6.5 s, bank the last window's 5 cm early, fall). The same hole is
   closed here by asking that the **support signature be stationary** —
   consecutive windows may not change their body-contact or airborne
   fractions by more than ``delta_max`` — which is what "still the same thing"
   meant physically, minus the names.

Why statistics and not the raw time series. AURORA embeds full sensory
trajectories, and on a deterministic simulator that is fine. This simulator is
measured chaotic: identical genomes spread sd 0.605 m in displacement, and a
walker's step timing decorrelates across replicas within a second. A raw
trajectory embedding would put that replica noise *into* the descriptor —
exactly what v3's Stage A found kills a grid. Order-invariant statistics over
the scored part of the episode (means, standard deviations, RMS, contact
fractions, one spectral peak) describe the *steady behaviour* rather than one
realisation of it, and they are the same quantities every hand-built axis
here was ever computed from. The learned part is the *combination*: which of
these ~80 numbers matter, and how they fold into a few latent coordinates, is
decided by the reconstruction loss and not by anyone's taste.

Nothing in this module imports mjlab. The accumulator takes tensors the
harness already reads every step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import torch

from qd.modes import ModeFeatures, WindowCfg, dominant_frequency

# --------------------------------------------------------------------------- #
# Feature accumulator
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BehaviourCfg:
    exempt_seconds: float = 1.0
    """Leading transition time excluded from every statistic.

    Same value and same reason as the gate's ``exempt_seconds``: every
    candidate spawns standing at HOME, and the first second is the
    stand-to-whatever transition, which is not the behaviour."""

    contact_force_n: float = 0.5
    """De-chatter threshold on per-geom contact, as in :class:`qd.modes.ModeStats`."""


@dataclass
class BehaviourFeatures:
    """``(N, D)`` feature matrix with its column names."""

    vector: np.ndarray
    names: tuple[str, ...]

    KEY: ClassVar[str] = "behaviour/vector"

    def to_info(self) -> dict[str, np.ndarray]:
        return {self.KEY: self.vector}

    @classmethod
    def from_info(cls, info: dict[str, np.ndarray], names: tuple[str, ...]) -> BehaviourFeatures:
        return cls(np.asarray(info[cls.KEY]), names)


def feature_names(num_joints: int, contact_geom_names: tuple[str, ...]) -> tuple[str, ...]:
    """Column order of :meth:`BehaviourStats.finalize`, for inspection and tests."""
    names: list[str] = ["z_mean", "z_std"]
    names += [f"grav_{a}_mean" for a in "xyz"] + [f"grav_{a}_std" for a in "xyz"]
    names += ["vx_mean", "vx_std", "vy_mean", "vy_std", "vz_std"]
    names += [f"w{a}_mean" for a in "xyz"] + [f"w{a}_std" for a in "xyz"]
    names += [f"q{j}_mean" for j in range(num_joints)]
    names += [f"q{j}_std" for j in range(num_joints)]
    names += [f"dq{j}_rms" for j in range(num_joints)]
    names += [f"contact_{g}" for g in contact_geom_names]
    names += ["f_air", "n_contacts_mean", "az_abs_mean", "z_dom_freq", "z_peak_ratio"]
    return tuple(names)


class BehaviourStats:
    """Running sums over the scored steps of one batched rollout.

    Every statistic is a mean, a standard deviation or an RMS over time — so
    the feature vector of a periodic gait does not depend on its phase at
    ``t = exempt_seconds``, which is the property that keeps replica noise out
    of the descriptor. The one trace kept is trunk ``z``, for the spectral
    pair (dominant frequency, peak-to-total power ratio): it is the only
    feature that says how a behaviour *repeats* rather than what it averages.
    """

    def __init__(
        self,
        num_envs: int,
        device: str | torch.device,
        windows: WindowCfg,
        num_joints: int,
        contact_geom_names: tuple[str, ...],
        cfg: BehaviourCfg | None = None,
    ):
        self.cfg = cfg or BehaviourCfg()
        self.num_envs = num_envs
        self.device = device
        self.windows = windows
        self.num_joints = num_joints
        self.geom_names = tuple(contact_geom_names)
        self.names = feature_names(num_joints, self.geom_names)
        self.exempt_steps = round(self.cfg.exempt_seconds / windows.control_dt)
        self.scored_steps = windows.episode_steps - self.exempt_steps
        if self.scored_steps <= 1:
            raise ValueError("exempt_seconds leaves no scored steps")

        f = lambda *shape: torch.zeros(*shape, dtype=torch.float32, device=device)
        G = len(self.geom_names)
        self.count = f(num_envs)
        self.s_z, self.ss_z = f(num_envs), f(num_envs)
        self.s_grav, self.ss_grav = f(num_envs, 3), f(num_envs, 3)
        self.s_v, self.ss_v = f(num_envs, 3), f(num_envs, 3)
        self.s_w, self.ss_w = f(num_envs, 3), f(num_envs, 3)
        self.s_q, self.ss_q = f(num_envs, num_joints), f(num_envs, num_joints)
        self.ss_dq = f(num_envs, num_joints)
        self.s_contact = f(num_envs, G)
        self.s_air = f(num_envs)
        self.s_ncontact = f(num_envs)
        self.s_az = f(num_envs)
        self.z_trace = f(self.scored_steps, num_envs)
        self._step = 0

    def begin(self) -> None:
        for t in (
            self.count, self.s_z, self.ss_z, self.s_grav, self.ss_grav, self.s_v,
            self.ss_v, self.s_w, self.ss_w, self.s_q, self.ss_q, self.ss_dq,
            self.s_contact, self.s_air, self.s_ncontact, self.s_az, self.z_trace,
        ):
            t.zero_()
        self._step = 0

    def update(
        self,
        base_pos: torch.Tensor,
        projected_gravity_b: torch.Tensor,
        lin_vel_w: torch.Tensor,
        ang_vel_w: torch.Tensor,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        contact_found: torch.Tensor,
        contact_force: torch.Tensor | None,
        trunk_az: torch.Tensor,
    ) -> None:
        step = self._step
        self._step += 1
        if step < self.exempt_steps or step >= self.windows.episode_steps:
            return
        k = step - self.exempt_steps
        nn = lambda t: torch.nan_to_num(t.to(torch.float32))

        z = nn(base_pos[:, 2])
        grav = nn(projected_gravity_b)
        v = nn(lin_vel_w)
        w = nn(ang_vel_w)
        q = nn(joint_pos)
        dq = nn(joint_vel)

        touching = contact_found.to(torch.float32) > 0
        if contact_force is not None and self.cfg.contact_force_n > 0:
            force = nn(contact_force)
            if force.dim() == 3:
                force = torch.linalg.vector_norm(force, dim=-1)
            touching = touching & (force >= self.cfg.contact_force_n)
        touching = touching.float()

        self.count += 1.0
        self.s_z += z
        self.ss_z += z * z
        self.s_grav += grav
        self.ss_grav += grav * grav
        self.s_v += v
        self.ss_v += v * v
        self.s_w += w
        self.ss_w += w * w
        self.s_q += q
        self.ss_q += q * q
        self.ss_dq += dq * dq
        self.s_contact += touching
        n_touch = touching.sum(dim=-1)
        self.s_air += (n_touch == 0).float()
        self.s_ncontact += n_touch
        self.s_az += nn(trunk_az).abs()
        self.z_trace[k] = z

    def finalize(self) -> BehaviourFeatures:
        n = self.count.clamp_min(1.0)
        mean = lambda s: s / n.reshape(-1, *([1] * (s.dim() - 1)))
        std = lambda s, ss: (mean(ss) - mean(s) ** 2).clamp_min(0.0).sqrt()
        rms = lambda ss: mean(ss).clamp_min(0.0).sqrt()

        cols = [
            mean(self.s_z)[:, None],
            std(self.s_z, self.ss_z)[:, None],
            mean(self.s_grav),
            std(self.s_grav, self.ss_grav),
            mean(self.s_v)[:, 0:1],
            std(self.s_v, self.ss_v)[:, 0:1],
            mean(self.s_v)[:, 1:2],
            std(self.s_v, self.ss_v)[:, 1:2],
            std(self.s_v, self.ss_v)[:, 2:3],
            mean(self.s_w),
            std(self.s_w, self.ss_w),
            mean(self.s_q),
            std(self.s_q, self.ss_q),
            rms(self.ss_dq),
            mean(self.s_contact),
            mean(self.s_air)[:, None],
            mean(self.s_ncontact)[:, None],
            mean(self.s_az)[:, None],
        ]
        vec = torch.cat(cols, dim=1).detach().cpu().numpy().astype(np.float32)

        used = int(min(max(self._step - self.exempt_steps, 2), self.scored_steps))
        trace = self.z_trace[:used].detach().cpu().numpy().astype(np.float64)
        dom = dominant_frequency(trace, self.windows.control_dt).astype(np.float32)
        centred = trace - trace.mean(axis=0, keepdims=True)
        spectrum = np.abs(np.fft.rfft(centred, axis=0)) ** 2
        spectrum[0] = 0.0
        total = spectrum.sum(axis=0)
        ratio = np.where(total > 0, spectrum.max(axis=0) / np.maximum(total, 1e-12), 0.0)
        vec = np.concatenate(
            [vec, dom[:, None], ratio.astype(np.float32)[:, None]], axis=1
        )
        assert vec.shape[1] == len(self.names), (vec.shape, len(self.names))
        return BehaviourFeatures(vec, self.names)


# --------------------------------------------------------------------------- #
# Viability without labels
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ViabilityV5Cfg:
    """P2'' — P2' with the label clause replaced by a stationarity clause.

    ::

        viable  <=>  finite state throughout
                 AND +x displacement >= d_min in every window from the second on
                 AND |Δ f_body|, |Δ f_air| <= delta_max between consecutive scored windows
                 AND p95 |a_z| <= impact_cap

    ``d_min``, ``impact_cap`` and ``exempt_seconds`` are v4's calibrated
    values, unchanged. ``delta_max`` is the one new number: see its docstring.
    """

    windows: WindowCfg = field(default_factory=WindowCfg)
    d_min: float = 0.05
    exempt_seconds: float = 1.0
    impact_cap: float | None = 17.0

    delta_max: float = 0.15
    """Largest allowed change in ``f_body`` or ``f_air`` between consecutive
    scored windows.

    What clause 3 of P2' actually did, physically: a late fall moves
    ``f_body`` from ~0 to ``(7 - t_fall) / 2`` in the last window, and the
    label flips when that crosses ``walk_body_max = 0.1``. So P2' rejected any
    walker that fell before ~6.8 s. This clause does the same arithmetic
    without a mode name in it — and, unlike the label rule, it does not kill a
    behaviour for sitting *between* two named clusters, which is where ~10 %
    of v4's viable candidates died.

    0.15 is the default pending measurement: ``qd.collect_behaviour`` reports
    the distribution of consecutive-window deltas over v4's verified elites,
    and the checkpoint report states the value used and its margin."""

    def exempt_windows(self) -> int:
        w = self.windows
        return min(w.num_windows, max(0, round(self.exempt_seconds / w.stride_seconds)))


@dataclass
class ViabilityV5Verdict:
    viable: np.ndarray
    finite: np.ndarray
    progress: np.ndarray
    stationary: np.ndarray
    impact: np.ndarray
    max_delta: np.ndarray
    """Largest consecutive-window support change actually seen — for calibration."""

    def rates(self) -> dict[str, float]:
        return {
            "viable": float(np.mean(self.viable)),
            "finite": float(np.mean(self.finite)),
            "progress": float(np.mean(self.progress)),
            "stationary": float(np.mean(self.stationary)),
            "impact": float(np.mean(self.impact)),
        }


def support_deltas(features: ModeFeatures, first_window: int) -> np.ndarray:
    """``(N,)`` max over consecutive scored windows of ``max(|Δf_body|, |Δf_air|)``."""
    body = features.window_f_body[first_window:]
    air = features.window_f_air[first_window:]
    if body.shape[0] < 2:
        return np.zeros(body.shape[1], dtype=np.float32)
    d = np.maximum(np.abs(np.diff(body, axis=0)), np.abs(np.diff(air, axis=0)))
    return d.max(axis=0)


def evaluate_viability_v5(
    features: ModeFeatures, cfg: ViabilityV5Cfg | None = None
) -> ViabilityV5Verdict:
    """Apply P2'' to one rollout's :class:`qd.modes.ModeFeatures`.

    Reads the same window arrays P2' read (``ModeStats`` is label-free; only
    ``ModeFeatures.window_labels`` ever touched the classifier, and it is not
    called here).
    """
    cfg = cfg or ViabilityV5Cfg()
    k = cfg.exempt_windows()
    if features.window_dx.shape[0] != cfg.windows.num_windows:
        raise ValueError(
            f"features carry {features.window_dx.shape[0]} windows but the cfg "
            f"describes {cfg.windows.num_windows}"
        )
    progress = np.all(features.window_dx[k:] >= cfg.d_min, axis=0)
    max_delta = support_deltas(features, k)
    stationary = max_delta <= cfg.delta_max
    if cfg.impact_cap is None:
        impact = np.ones(features.f_body.shape, dtype=bool)
    else:
        impact = features.p95_az <= cfg.impact_cap
    finite = features.finite.astype(bool)
    return ViabilityV5Verdict(
        viable=finite & progress & stationary & impact,
        finite=finite,
        progress=progress,
        stationary=stationary,
        impact=impact,
        max_delta=max_delta,
    )


@dataclass
class ReplicaVerdict:
    """One candidate block folded over world-permuted replicas."""

    viable: np.ndarray
    """k-of-N gate outcome."""
    viable_count: np.ndarray
    fitness: np.ndarray
    """Median displacement over replicas — never the max."""
    features: np.ndarray
    """``(N, D)`` median feature vector over replicas."""
    clause_rates: dict[str, float]


def fold_replicas_v5(
    per_replica: list[tuple[ModeFeatures, np.ndarray]],
    cfg: ViabilityV5Cfg,
    viable_min: int,
) -> ReplicaVerdict:
    """k-of-N viability, median fitness, median features.

    The feature median is taken per coordinate over replicas *before*
    encoding, so one candidate has one descriptor, and it is the descriptor of
    its typical replica rather than of its luckiest."""
    verdicts = [evaluate_viability_v5(f, cfg) for f, _b in per_replica]
    stack = np.stack([v.viable for v in verdicts])
    n = stack.shape[0]
    count = stack.sum(axis=0)
    viable = count >= min(viable_min, n)
    fitness = np.median(np.stack([f.displacement for f, _b in per_replica]), axis=0)
    features = np.median(np.stack([b for _f, b in per_replica]), axis=0)
    rates = {
        key: float(np.mean([v.rates()[key] for v in verdicts]))
        for key in ("finite", "progress", "stationary", "impact")
    }
    rates["viable_replicas_mean"] = float(np.mean(count))
    rates["max_delta_median"] = float(np.median(np.stack([v.max_delta for v in verdicts])))
    return ReplicaVerdict(viable, count, fitness, features, rates)
