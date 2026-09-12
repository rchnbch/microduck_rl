"""The learned behaviour space: standardise -> encode -> whiten, plus the
fixed centroid set and the clustering rule that decides what a "mode" is.

Two encoders, chosen by ``kind``:

* ``pca`` — the linear baseline. Deterministic, no hyper-parameters, and the
  honest control: if an autoencoder does not separate v4's known modes better
  than its own first principal components, the non-linearity is not earning
  its place.
* ``ae`` — a small MLP autoencoder (``D -> 64 -> 32 -> L -> 32 -> 64 -> D``,
  ELU), trained on reconstruction MSE of the standardised features with a
  held-out split and early stopping. This is AURORA's choice, and the reason
  it is the default here is not fashion: the features are contact fractions
  and joint statistics whose *joint* structure (which geoms touch together,
  which joints swing together) is what distinguishes a posture, and a linear
  projection has to spend a component on every pairwise correlation.

Latents are **whitened** after training — first by a plain per-dimension
z-score, then, once replica data exists, **in units of replica noise**
(:meth:`BehaviourSpace.calibrate_to_noise`): one unit along any axis is one
standard deviation of what world-permuted replicas of the *same genome* do.
That is v3's spread-to-noise rule applied to a learned space, and it is what
makes a distance threshold mean the same thing along every axis.

Everything else here is numpy + sklearn so it is unit-testable on CPU:

* :func:`make_centroids` — a CVT over a fixed box, K-means on uniform samples
  (Vassiliades et al.). The **evaluation** centroid set is made once, from the
  frozen evaluation encoder, and never regenerated; every archive compared in
  this job is scored on the same centroids.
* :func:`replica_noise` — the per-genome latent spread across world-permuted
  replicas. It is the unit every distance-based rule below is written in, so
  "distinct" means "further apart than the simulator's own scatter".
* :func:`cluster_modes` — DBSCAN at ``eps = NOISE_MULTIPLIER * noise``,
  ``min_samples = 3``; a *mode* is a cluster of at least ``MIN_MODE_ELITES``
  verified elites. Fixed here, before any v5 result exists.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

NOISE_MULTIPLIER: float = 3.0
"""DBSCAN ``eps`` in units of replica noise. Pre-registered."""

MIN_MODE_ELITES: int = 5
"""Smallest cluster that counts as a mode. Pre-registered."""

DEFAULT_CENTROIDS: int = 1024
"""Size of the fixed evaluation centroid set. Pre-registered."""

AXIS_BINS: int = 10
"""Per-axis bins for the "does this mode span the axis" occupancy table."""


# --------------------------------------------------------------------------- #
# Encoder
# --------------------------------------------------------------------------- #


class _AutoEncoder(nn.Module):
    def __init__(self, dim: int, latent: int, hidden: tuple[int, ...] = (64, 32)):
        super().__init__()
        enc, last = [], dim
        for h in hidden:
            enc += [nn.Linear(last, h), nn.ELU()]
            last = h
        enc.append(nn.Linear(last, latent))
        dec, last = [], latent
        for h in reversed(hidden):
            dec += [nn.Linear(last, h), nn.ELU()]
            last = h
        dec.append(nn.Linear(last, dim))
        self.encoder = nn.Sequential(*enc)
        self.decoder = nn.Sequential(*dec)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


@dataclass
class TrainCfg:
    latent_dim: int = 4
    kind: str = "ae"
    hidden: tuple[int, ...] = (64, 32)
    epochs: int = 400
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-5
    val_fraction: float = 0.15
    patience: int = 40
    seed: int = 0
    std_floor: float = 1e-4
    """Features with less spread than this over the training set (e.g. a geom
    that never touches) are standardised to zero rather than blown up."""


@dataclass
class BehaviourSpace:
    """A fitted encoder with the standardisation it expects and the whitening it emits."""

    names: tuple[str, ...]
    kind: str
    latent_dim: int
    feat_mean: np.ndarray
    feat_std: np.ndarray
    latent_mean: np.ndarray
    latent_std: np.ndarray
    pca_components: np.ndarray | None = None
    """``(L, D)`` for ``kind == "pca"``."""
    ae_state: dict | None = None
    hidden: tuple[int, ...] = (64, 32)
    train_info: dict = field(default_factory=dict)

    # -- inference ----------------------------------------------------------- #

    def standardise(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        return (x - self.feat_mean) / self.feat_std

    def encode(self, x: np.ndarray) -> np.ndarray:
        """``(N, D)`` raw features -> ``(N, L)`` whitened latents."""
        xs = self.standardise(np.atleast_2d(x))
        if self.kind == "pca":
            z = xs @ self.pca_components.T
        else:
            model = self._model()
            with torch.no_grad():
                z = model.encoder(torch.as_tensor(xs, dtype=torch.float32)).numpy()
        return ((z - self.latent_mean) / self.latent_std).astype(np.float32)

    def raw_latent(self, x: np.ndarray) -> np.ndarray:
        """``(N, L)`` latents before whitening."""
        xs = self.standardise(np.atleast_2d(x))
        if self.kind == "pca":
            return (xs @ self.pca_components.T).astype(np.float32)
        model = self._model()
        with torch.no_grad():
            return model.encoder(torch.as_tensor(xs, dtype=torch.float32)).numpy()

    def calibrate_to_noise(self, groups: list[np.ndarray], floor_fraction: float = 0.05) -> dict:
        """Re-whiten so one unit along every latent axis is one replica sd.

        ``groups`` holds, per genome, the RAW features of its viable replicas
        ``(R_i, D)``. Per axis, the noise sd is the RMS over genomes of the
        within-genome sd. Dividing by it is v3's spread-to-noise lesson applied
        to a learned space: distances are then in units of what the simulator
        does to the same genome, and an axis the encoder filled with replica
        scatter is no longer a unit-variance axis that fragments every cluster.
        An axis with almost no replica scatter is floored at
        ``floor_fraction`` of its total sd so a near-deterministic direction
        does not become infinitely fine."""
        raw = [self.raw_latent(g) for g in groups if len(g) >= 2]
        if not raw:
            raise ValueError("calibrate_to_noise needs genomes with >= 2 replicas")
        within = np.sqrt(np.mean([g.var(axis=0, ddof=1) for g in raw], axis=0))
        allz = np.concatenate(raw)
        total = allz.std(axis=0)
        std = np.maximum(within, floor_fraction * np.maximum(total, 1e-6))
        self.latent_mean = allz.mean(axis=0).astype(np.float32)
        self.latent_std = std.astype(np.float32)
        info = {
            "noise_sd_per_axis": within.tolist(),
            "total_sd_per_axis": total.tolist(),
            "spread_to_noise_per_axis": (total / np.maximum(within, 1e-9)).tolist(),
            "genomes": len(raw),
        }
        self.train_info["noise_calibration"] = info
        return info

    def reconstruction_error(self, x: np.ndarray) -> np.ndarray:
        """Per-row MSE in standardised feature units."""
        xs = self.standardise(np.atleast_2d(x))
        if self.kind == "pca":
            z = xs @ self.pca_components.T
            rec = z @ self.pca_components
        else:
            model = self._model()
            with torch.no_grad():
                rec = model(torch.as_tensor(xs, dtype=torch.float32))[0].numpy()
        return ((rec - xs) ** 2).mean(axis=1)

    def _model(self) -> _AutoEncoder:
        model = _AutoEncoder(len(self.names), self.latent_dim, self.hidden)
        model.load_state_dict({k: torch.as_tensor(v) for k, v in self.ae_state.items()})
        model.eval()
        return model

    # -- persistence --------------------------------------------------------- #

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "feat_mean": self.feat_mean,
            "feat_std": self.feat_std,
            "latent_mean": self.latent_mean,
            "latent_std": self.latent_std,
            "meta_json": np.array(
                json.dumps(
                    {
                        "names": list(self.names),
                        "kind": self.kind,
                        "latent_dim": self.latent_dim,
                        "hidden": list(self.hidden),
                        "train_info": self.train_info,
                    }
                )
            ),
        }
        if self.pca_components is not None:
            payload["pca_components"] = self.pca_components
        if self.ae_state is not None:
            for k, v in self.ae_state.items():
                payload[f"ae/{k}"] = np.asarray(v)
        np.savez_compressed(path, **payload)
        return path

    @classmethod
    def load(cls, path: str | Path) -> BehaviourSpace:
        with np.load(Path(path), allow_pickle=False) as f:
            meta = json.loads(str(f["meta_json"]))
            ae_state = {
                k[len("ae/"):]: f[k] for k in f.files if k.startswith("ae/")
            } or None
            return cls(
                names=tuple(meta["names"]),
                kind=meta["kind"],
                latent_dim=int(meta["latent_dim"]),
                feat_mean=f["feat_mean"],
                feat_std=f["feat_std"],
                latent_mean=f["latent_mean"],
                latent_std=f["latent_std"],
                pca_components=f["pca_components"] if "pca_components" in f.files else None,
                ae_state=ae_state,
                hidden=tuple(meta["hidden"]),
                train_info=meta.get("train_info", {}),
            )


def fit_space(x: np.ndarray, names: tuple[str, ...], cfg: TrainCfg) -> BehaviourSpace:
    """Fit a :class:`BehaviourSpace` on ``(N, D)`` raw features."""
    x = np.asarray(x, dtype=np.float32)
    x = x[np.isfinite(x).all(axis=1)]
    if x.shape[1] != len(names):
        raise ValueError(f"{x.shape[1]} features but {len(names)} names")
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    dead = std < cfg.std_floor
    std = np.where(dead, 1.0, std).astype(np.float32)
    xs = (x - mean) / std
    xs[:, dead] = 0.0
    info: dict = {"rows": len(x), "dead_features": [names[i] for i in np.flatnonzero(dead)]}

    if cfg.kind == "pca":
        # SVD on the centred, standardised features.
        _u, s, vt = np.linalg.svd(xs - xs.mean(axis=0), full_matrices=False)
        comps = vt[: cfg.latent_dim].astype(np.float32)
        var = s**2 / max(len(xs) - 1, 1)
        info["explained_variance_ratio"] = (var[: cfg.latent_dim] / var.sum()).tolist()
        info["explained_variance_ratio_total"] = float(var[: cfg.latent_dim].sum() / var.sum())
        z = xs @ comps.T
        space = BehaviourSpace(
            names=tuple(names), kind="pca", latent_dim=cfg.latent_dim,
            feat_mean=mean, feat_std=std, latent_mean=np.zeros(cfg.latent_dim, np.float32),
            latent_std=np.ones(cfg.latent_dim, np.float32), pca_components=comps,
        )
    elif cfg.kind == "ae":
        torch.manual_seed(cfg.seed)
        rng = np.random.default_rng(cfg.seed)
        perm = rng.permutation(len(xs))
        n_val = max(1, int(cfg.val_fraction * len(xs)))
        val_idx, tr_idx = perm[:n_val], perm[n_val:]
        xt = torch.as_tensor(xs)
        model = _AutoEncoder(xs.shape[1], cfg.latent_dim, cfg.hidden)
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        best, best_state, bad, curve = float("inf"), None, 0, []
        for epoch in range(cfg.epochs):
            model.train()
            order = rng.permutation(tr_idx)
            for s0 in range(0, len(order), cfg.batch_size):
                batch = xt[order[s0 : s0 + cfg.batch_size]]
                rec, _z = model(batch)
                loss = ((rec - batch) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                val = float(((model(xt[val_idx])[0] - xt[val_idx]) ** 2).mean())
            curve.append(val)
            if val < best - 1e-6:
                best, bad = val, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= cfg.patience:
                    break
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            z = model.encoder(xt).numpy()
        info.update({"val_mse": best, "epochs_run": len(curve), "val_curve": curve[::10]})
        space = BehaviourSpace(
            names=tuple(names), kind="ae", latent_dim=cfg.latent_dim,
            feat_mean=mean, feat_std=std, latent_mean=np.zeros(cfg.latent_dim, np.float32),
            latent_std=np.ones(cfg.latent_dim, np.float32),
            ae_state={k: v.numpy() for k, v in best_state.items()}, hidden=tuple(cfg.hidden),
        )
    else:
        raise ValueError(f"unknown encoder kind {cfg.kind!r}")

    space.latent_mean = z.mean(axis=0).astype(np.float32)
    space.latent_std = np.maximum(z.std(axis=0), 1e-6).astype(np.float32)
    info["train_mse"] = float(space.reconstruction_error(x).mean())
    space.train_info = info
    return space


# --------------------------------------------------------------------------- #
# Centroids, noise, coverage
# --------------------------------------------------------------------------- #


def latent_bounds(z: np.ndarray, pad: float = 0.10, lo_q: float = 1.0, hi_q: float = 99.0) -> np.ndarray:
    """``(L, 2)`` box: percentiles of the training latents, padded outward."""
    lo = np.percentile(z, lo_q, axis=0)
    hi = np.percentile(z, hi_q, axis=0)
    span = np.maximum(hi - lo, 1e-6)
    return np.stack([lo - pad * span, hi + pad * span], axis=1).astype(np.float32)


def make_centroids(k: int, bounds: np.ndarray, seed: int = 0, samples: int = 100_000) -> np.ndarray:
    """CVT centroids: k-means over uniform samples in ``bounds``."""
    from sklearn.cluster import KMeans

    rng = np.random.default_rng(seed)
    lo, hi = bounds[:, 0], bounds[:, 1]
    pts = rng.uniform(lo, hi, size=(samples, len(lo)))
    km = KMeans(n_clusters=k, n_init=1, random_state=seed).fit(pts)
    return km.cluster_centers_.astype(np.float32)


def nearest_centroid(z: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """``(N,)`` index of the closest centroid."""
    z = np.atleast_2d(z)
    d = ((z[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=-1)
    return d.argmin(axis=1)


def coverage(z: np.ndarray, centroids: np.ndarray) -> tuple[int, np.ndarray]:
    """Occupied centroid count and the occupied index set."""
    if len(z) == 0:
        return 0, np.zeros(0, dtype=int)
    idx = np.unique(nearest_centroid(z, centroids))
    return len(idx), idx


def replica_noise(z_per_replica: np.ndarray) -> tuple[float, np.ndarray]:
    """Latent scatter across replicas of the same genome.

    ``z_per_replica`` is ``(R, N, L)``. Per genome: RMS distance of each
    replica's latent from the genome's per-coordinate median. Returns the
    median of that over genomes, and the per-genome values."""
    z = np.asarray(z_per_replica)
    med = np.median(z, axis=0, keepdims=True)
    per = np.sqrt(((z - med) ** 2).sum(axis=-1).mean(axis=0))
    return float(np.median(per)), per


# --------------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------------- #


@dataclass
class ModeReport:
    labels: np.ndarray
    """DBSCAN label per elite; -1 = noise; clusters below the size floor are
    relabelled -2 ("fragment") so they are never counted as modes."""
    modes: list[dict]
    eps: float
    noise: float
    n_modes: int
    n_fragments: int
    n_noise: int
    min_inter_mode_distance_over_eps: float | None


def cluster_modes(
    z: np.ndarray,
    noise: float,
    bounds: np.ndarray,
    extra: dict[str, np.ndarray] | None = None,
    multiplier: float = NOISE_MULTIPLIER,
    min_elites: int = MIN_MODE_ELITES,
) -> ModeReport:
    """The pre-registered "distinct mode" rule, applied to verified elites.

    ``extra`` maps a name to an ``(N,)`` array reported per mode as a range
    (e.g. raw ``z_mean`` and ``joint speed`` so a one-genome frequency sweep
    shows up as a mode that spans one raw axis and nothing else).
    """
    from sklearn.cluster import DBSCAN

    z = np.atleast_2d(z)
    eps = multiplier * noise
    if len(z) == 0:
        return ModeReport(np.zeros(0, int), [], eps, noise, 0, 0, 0, None)
    labels = DBSCAN(eps=eps, min_samples=3).fit(z).labels_.copy()
    modes: list[dict] = []
    keep_ids = []
    for c in sorted(set(labels) - {-1}):
        members = np.flatnonzero(labels == c)
        if len(members) < min_elites:
            labels[members] = -2
            continue
        keep_ids.append(c)
    # per-axis occupancy on the fixed box
    edges = [np.linspace(bounds[a, 0], bounds[a, 1], AXIS_BINS + 1) for a in range(z.shape[1])]
    for new_id, c in enumerate(keep_ids):
        members = np.flatnonzero(labels == c)
        zm = z[members]
        per_axis = []
        for a in range(z.shape[1]):
            b = np.clip(np.digitize(zm[:, a], edges[a]) - 1, 0, AXIS_BINS - 1)
            per_axis.append(len(np.unique(b)))
        entry = {
            "mode": new_id,
            "elites": len(members),
            "centroid": zm.mean(axis=0).tolist(),
            "latent_std": zm.std(axis=0).tolist(),
            "axis_bins_occupied": per_axis,
            "axis_bins_total": AXIS_BINS,
            "axes_spanned_ge2_bins": int(sum(p >= 2 for p in per_axis)),
        }
        if extra:
            for k, v in extra.items():
                vv = np.asarray(v)[members]
                entry[f"{k}_range"] = [float(np.min(vv)), float(np.max(vv))]
        modes.append(entry)
        labels[members] = 1000 + new_id
    labels = np.where(labels >= 1000, labels - 1000, labels)
    min_d = None
    if len(modes) >= 2:
        cents = np.array([m["centroid"] for m in modes])
        d = np.sqrt(((cents[:, None] - cents[None]) ** 2).sum(-1))
        d[np.eye(len(cents), dtype=bool)] = np.inf
        min_d = float(d.min() / eps)
        for i, m in enumerate(modes):
            m["nearest_other_mode_distance_over_eps"] = float(d[i].min() / eps)
    return ModeReport(
        labels=labels,
        modes=modes,
        eps=eps,
        noise=noise,
        n_modes=len(modes),
        n_fragments=int(np.sum(labels == -2)),
        n_noise=int(np.sum(labels == -1)),
        min_inter_mode_distance_over_eps=min_d,
    )


def purity(labels: np.ndarray, reference: np.ndarray) -> dict:
    """How a clustering lines up with a post-hoc reference labelling.

    Reference labels (v4's walk/crawl) are used for REPORTING only; nothing
    in the insertion path sees them."""
    out = {}
    for c in sorted(set(labels.tolist())):
        if c < 0:
            continue
        members = reference[labels == c]
        vals, counts = np.unique(members, return_counts=True)
        out[int(c)] = {str(v): int(n) for v, n in zip(vals, counts)}
    return out
