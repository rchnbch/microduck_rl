"""Stage A''s two judgement calls, locked against quiet reversal.

Both were wrong on the first attempt in a way that produced plausible-looking
numbers, which is the reason they are tested rather than trusted:

* a calibration positive was "ends up ahead", which admitted a probe that
  lurches once and lies still — the exact profile P2' exists to exclude;
* the setting was chosen by maximising the margin, which once the probe set
  separates cleanly means "pick whatever the positives like best", i.e. the
  most permissive setting.
"""

from __future__ import annotations

import numpy as np
from qd.modes import ModeFeatures, WindowCfg
from qd.stage_a_prime import ProbeResult, choose_setting, split_positives

NW = WindowCfg().num_windows


def result(name, *, window_dx, positive=True, mode="crawl"):
    """A ProbeResult whose per-window displacements are given explicitly."""
    dx = np.asarray(window_dx, dtype=np.float32).reshape(NW, 1)
    ep = lambda v: np.full(1, float(v), dtype=np.float32)
    col = lambda v: np.full((NW, 1), float(v), dtype=np.float32)
    feats = ModeFeatures(
        window_dx=dx,
        window_f_body=col(0.9),
        window_f_air=col(0.0),
        window_f_feet=col(0.1),
        window_f_head=col(0.0),
        window_rotation_rate=col(0.0),
        f_body=ep(0.9),
        f_air=ep(0.0),
        f_feet=ep(0.1),
        f_head=ep(0.0),
        f_inverted=ep(0.0),
        rotation_rate=ep(0.0),
        rotation_rate_pitch=ep(0.0),
        rotation_rate_roll=ep(0.0),
        rotation_rate_yaw=ep(0.0),
        p95_az=ep(1.0),
        displacement=ep(float(dx.sum())),
        finite=np.ones(1, dtype=bool),
    )
    per_window = {1.5: feats, 2.0: feats, 3.0: feats}
    return ProbeResult(name, mode, positive, "scripted", per_window, feats.displacement)


# --------------------------------------------------------------------------- #
# Which positives calibrate
# --------------------------------------------------------------------------- #


def test_a_lurch_that_ends_up_ahead_is_not_a_calibration_positive():
    # tuck_and_flop: +0.05 m once, then nothing. Total displacement is positive
    # and every later window is flat — the front-loaded profile P2' excludes.
    lurch = result("lurch", window_dx=[0.20, 0.0, 0.0, 0.0, 0.0, 0.0])
    moving, stuck = split_positives([lurch], d_min=0.05)
    assert moving == []
    assert stuck == ["lurch"]


def test_a_probe_that_sustains_progress_calibrates():
    steady = result("steady", window_dx=[0.2] * NW)
    moving, stuck = split_positives([steady], d_min=0.05)
    assert moving == ["steady"]
    assert stuck == []


def test_the_exempt_first_window_does_not_disqualify_a_slow_starter():
    # A crawl spends the first window getting down; that window is exempt.
    starter = result("starter", window_dx=[0.0, 0.2, 0.2, 0.2, 0.2, 0.2])
    moving, _ = split_positives([starter], d_min=0.05)
    assert moving == ["starter"]


def test_negatives_are_never_calibration_positives():
    neg = result("neg", window_dx=[0.2] * NW, positive=False)
    moving, stuck = split_positives([neg])
    assert moving == [] and stuck == []


# --------------------------------------------------------------------------- #
# Which setting is chosen
# --------------------------------------------------------------------------- #


def sweep_row(worst_pos, best_neg, n_pos=5, n_neg=5):
    return {
        "per_probe": {},
        "worst_positive": worst_pos,
        "best_negative": best_neg,
        "margin": worst_pos - best_neg,
        "positives_at_or_above_0.95": n_pos if worst_pos >= 0.95 else n_pos - 1,
        "positives": n_pos,
        "negatives_at_or_below_0.05": n_neg if best_neg <= 0.05 else n_neg - 1,
        "negatives": n_neg,
    }


def test_the_strictest_clearing_setting_wins_over_the_largest_margin():
    # The measured case: W=3/d_min=0.1 has margin 1.0 and demands 3.3 cm/s;
    # W=2/d_min=0.1 has margin 0.992 and demands 5.0 cm/s. Maximising the
    # margin picks the weaker predicate on a difference of 0.008 in a number
    # that has stopped discriminating.
    sweep = {
        "W=3,d_min=0.1": sweep_row(1.0, 0.0),
        "W=2,d_min=0.1": sweep_row(0.992, 0.0),
        "W=2,d_min=0.05": sweep_row(1.0, 0.0),
    }
    chosen = choose_setting(sweep)
    assert chosen["setting"] == "W=2,d_min=0.1"
    assert chosen["cleared_both_bars"] is True
    assert chosen["required_speed_m_per_s"] == 0.05


def test_a_setting_that_fails_a_bar_is_not_selectable_however_strict():
    sweep = {
        "W=1.5,d_min=0.1": sweep_row(0.80, 0.0),  # strictest, but positives fail
        "W=2,d_min=0.05": sweep_row(1.0, 0.0),
    }
    assert choose_setting(sweep)["setting"] == "W=2,d_min=0.05"


def test_a_setting_admitting_a_negative_is_not_selectable():
    sweep = {
        "W=2,d_min=0.1": sweep_row(1.0, 1.0),  # a negative passes every replica
        "W=2,d_min=0.05": sweep_row(1.0, 0.0),
    }
    assert choose_setting(sweep)["setting"] == "W=2,d_min=0.05"


def test_when_nothing_clears_both_bars_that_is_reported_not_papered_over():
    sweep = {
        "W=2,d_min=0.05": sweep_row(0.4, 0.6),
        "W=3,d_min=0.05": sweep_row(0.5, 0.5),
    }
    chosen = choose_setting(sweep)
    assert chosen["cleared_both_bars"] is False
    assert "largest margin" in chosen["selection_rule"]
    assert chosen["settings_clearing_both_bars"] == []


def test_shorter_windows_break_ties_at_equal_required_speed():
    sweep = {
        "W=2,d_min=0.1": sweep_row(1.0, 0.0),
        "W=3,d_min=0.15": sweep_row(1.0, 0.0),  # same 5 cm/s, longer window
    }
    assert choose_setting(sweep)["setting"] == "W=2,d_min=0.1"
