"""CPU regression tests for the PGA-MAP-Elites pieces (qd/pga/).

The acceptance criterion for Phase 3 is that *both* variation operators
contribute archive insertions — a PG insertion rate near zero means the critic
or the reward wiring is broken, not that the operator is unhelpful. These tests
pin the wiring that would silently produce that: the genome's flat layout, the
critic actually being differentiable through the flat genome, and the shape of
the per-step reward the critic is trained on.

Walking-v2 adds the survival gate and the honest-physics switches to that list.
"""

import numpy as np
import pytest
import torch
from qd.common import FitnessCfg
from qd.pga.evaluate import (
    DR_EVENT_TERMS,
    KEEP_EVENT_TERMS,
    ShapedRewardCfg,
    Transitions,
)
from qd.pga.policy_genome import PolicySpec
from qd.pga.td3 import ReplayBuffer, Td3Cfg, Td3Trainer
from qd.pga.variation import isoline_variation, pg_variation

SPEC = PolicySpec()


def _transitions(n: int, spec: PolicySpec = SPEC, seed: int = 0) -> Transitions:
    g = torch.Generator().manual_seed(seed)
    return Transitions(
        obs=torch.randn(n, spec.obs_dim, generator=g),
        action=torch.randn(n, spec.action_dim, generator=g).clamp(-1, 1),
        reward=torch.randn(n, generator=g) * 0.01,
        next_obs=torch.randn(n, spec.obs_dim, generator=g),
        done=(torch.rand(n, generator=g) < 0.02).float(),
    )


# --------------------------------------------------------------------------- #
# Genome
# --------------------------------------------------------------------------- #


def test_genome_matches_the_61_to_14_contract():
    """The 61-D observation / 14-D action contract is a hard repo invariant."""
    assert SPEC.obs_dim == 61
    assert SPEC.action_dim == 14
    assert SPEC.hidden == (64, 64)
    # (61*64+64) + (64*64+64) + (64*14+14)
    assert SPEC.genome_dim == 3968 + 4160 + 910 == 9038


def test_unflatten_covers_the_genome_exactly():
    genomes = torch.arange(2 * SPEC.genome_dim, dtype=torch.float32).reshape(2, -1)
    layers = SPEC.unflatten(genomes)
    assert [tuple(w.shape[1:]) for w, _ in layers] == SPEC.layer_shapes
    total = sum(w[0].numel() + b[0].numel() for w, b in layers)
    assert total == SPEC.genome_dim
    # First slice is the first layer's weight, read row-major from the front.
    assert layers[0][0][0].flatten()[0] == 0.0
    assert layers[0][0][0].flatten()[-1] == float(64 * 61 - 1)


def test_forward_shapes_and_bounded_output():
    g = torch.Generator().manual_seed(0)
    genomes = SPEC.initial_population(5, g, "cpu")
    rollout = SPEC.forward(genomes, torch.randn(5, SPEC.obs_dim, generator=g))
    assert rollout.shape == (5, SPEC.action_dim)
    batched = SPEC.forward(genomes, torch.randn(5, 7, SPEC.obs_dim, generator=g))
    assert batched.shape == (5, 7, SPEC.action_dim)
    # tanh output: TD3's noise clipping assumes the action space is [-1, 1].
    assert rollout.abs().max() <= 1.0 and batched.abs().max() <= 1.0


def test_each_policy_in_the_batch_is_independent():
    """One shared observation, different genomes -> different actions."""
    g = torch.Generator().manual_seed(1)
    genomes = SPEC.initial_population(3, g, "cpu")
    obs = torch.randn(1, SPEC.obs_dim, generator=g).repeat(3, 1)
    actions = SPEC.forward(genomes, obs)
    assert not torch.allclose(actions[0], actions[1])
    # ...and duplicating a genome reproduces its action exactly.
    dup = torch.stack([genomes[0], genomes[0]])
    dup_actions = SPEC.forward(dup, obs[:2])
    torch.testing.assert_close(dup_actions[0], dup_actions[1])
    torch.testing.assert_close(dup_actions[0], actions[0])


def test_initial_population_is_per_layer_scaled():
    """A single global sigma over 9k weights saturates tanh on step one."""
    g = torch.Generator().manual_seed(2)
    genomes = SPEC.initial_population(64, g, "cpu")
    stds = [w.std().item() for w, _ in SPEC.unflatten(genomes)]
    # fan_in 61 vs 64 vs 64 -> the layers must not share one scale by accident,
    # and every layer must stay inside its own 1/sqrt(fan_in) bound.
    for (out_dim, in_dim), std in zip(SPEC.layer_shapes, stds):
        assert std < 1.0 / np.sqrt(in_dim)
        del out_dim
    actions = SPEC.forward(genomes, torch.randn(64, SPEC.obs_dim, generator=g))
    assert actions.abs().mean() < 0.9, "policies start saturated"


# --------------------------------------------------------------------------- #
# Replay buffer
# --------------------------------------------------------------------------- #


def test_buffer_wraps_and_keeps_the_newest():
    buf = ReplayBuffer(10, SPEC.obs_dim, SPEC.action_dim, "cpu")
    buf.add(_transitions(6))
    assert len(buf) == 6
    buf.add(_transitions(7, seed=1))
    assert len(buf) == 10  # capped, not grown
    g = torch.Generator().manual_seed(0)
    sample = buf.sample(4, generator=g)
    assert sample.obs.shape == (4, SPEC.obs_dim)


def test_buffer_sample_supports_a_per_policy_leading_axis():
    """PG variation gives each offspring its own independent transition batch."""
    buf = ReplayBuffer(100, SPEC.obs_dim, SPEC.action_dim, "cpu")
    buf.add(_transitions(100))
    g = torch.Generator().manual_seed(0)
    sample = buf.sample(5, 8, generator=g)
    assert sample.obs.shape == (5, 8, SPEC.obs_dim)
    assert sample.reward.shape == (5, 8)


def test_oversized_add_keeps_the_most_recent_slice():
    buf = ReplayBuffer(4, SPEC.obs_dim, SPEC.action_dim, "cpu")
    t = _transitions(10)
    buf.add(t)
    assert len(buf) == 4
    torch.testing.assert_close(buf.reward.sort().values, t.reward[-4:].sort().values)


# --------------------------------------------------------------------------- #
# Variation
# --------------------------------------------------------------------------- #


def test_isoline_interpolates_along_the_parent_direction():
    """With iso_sigma=0 the child must lie exactly on the a->b line."""
    g = torch.Generator().manual_seed(0)
    a = torch.randn(16, 32, generator=g)
    b = torch.randn(16, 32, generator=g)
    child = isoline_variation(a, b, g, iso_sigma=0.0, line_sigma=0.05)
    direction = b - a
    coeff = ((child - a) / direction)[:, :1]
    torch.testing.assert_close(a + coeff * direction, child, rtol=1e-4, atol=1e-6)


def test_isoline_scales_with_parent_separation():
    """The line term is scale-free — that is why it beats a fixed sigma on 9k
    weights."""
    g = torch.Generator().manual_seed(0)
    a = torch.zeros(2048, 16)
    near = isoline_variation(a, a + 0.01, g, iso_sigma=0.0, line_sigma=0.05)
    far = isoline_variation(a, a + 10.0, g, iso_sigma=0.0, line_sigma=0.05)
    assert far.std() > 100 * near.std()


def test_isoline_rejects_mismatched_parents():
    g = torch.Generator().manual_seed(0)
    with pytest.raises(ValueError):
        isoline_variation(torch.zeros(4, 8), torch.zeros(5, 8), g)


def test_pg_variation_is_a_noop_on_an_empty_buffer():
    """Before any transitions exist, PG offspring must be clean copies rather
    than garbage — otherwise iteration 1 poisons the archive."""
    trainer = Td3Trainer(Td3Cfg(replay_buffer_size=64), "cpu", seed=0, spec=SPEC)
    g = torch.Generator().manual_seed(0)
    parents = SPEC.initial_population(3, g, "cpu")
    out = pg_variation(parents, trainer, steps=5, spec=SPEC)
    torch.testing.assert_close(out, parents)


def test_pg_variation_moves_genomes_and_raises_the_critic_value():
    """The gradient must actually reach the flat genome: if `unflatten` copied
    instead of viewing, this silently optimises nothing."""
    cfg = Td3Cfg(replay_buffer_size=512, batch_size=32)
    trainer = Td3Trainer(cfg, "cpu", seed=0, spec=SPEC)
    trainer.buffer.add(_transitions(512))

    g = torch.Generator().manual_seed(0)
    parents = SPEC.initial_population(4, g, "cpu")
    batch = trainer.buffer.sample(4, 64, generator=trainer.generator)

    def mean_q(genomes):
        with torch.no_grad():
            actions = SPEC.forward(genomes, batch.obs)
            return float(
                trainer.critic.q1_value(
                    batch.obs.reshape(-1, SPEC.obs_dim),
                    actions.reshape(-1, SPEC.action_dim),
                ).mean()
            )

    before = mean_q(parents)
    children = pg_variation(parents, trainer, steps=40, spec=SPEC)
    assert not torch.allclose(children, parents), "PG variation did not move the genome"
    assert mean_q(children) > before, "PG steps did not increase the critic's Q"


# --------------------------------------------------------------------------- #
# TD3
# --------------------------------------------------------------------------- #


def test_td3_train_is_a_noop_until_the_buffer_fills():
    trainer = Td3Trainer(Td3Cfg(batch_size=256), "cpu", seed=0, spec=SPEC)
    trainer.buffer.add(_transitions(10))
    before = trainer.greedy_genome()
    trainer.train(steps=5)
    torch.testing.assert_close(trainer.greedy_genome(), before)


def test_td3_train_updates_critic_and_greedy_actor():
    cfg = Td3Cfg(replay_buffer_size=1024, batch_size=64)
    trainer = Td3Trainer(cfg, "cpu", seed=0, spec=SPEC)
    trainer.buffer.add(_transitions(1024))
    before_greedy = trainer.greedy_genome()
    before_critic = [p.clone() for p in trainer.critic.parameters()]

    losses = trainer.train(steps=20)
    assert np.isfinite(losses["critic_loss"])
    assert not torch.allclose(trainer.greedy_genome(), before_greedy)
    assert any(
        not torch.allclose(a, b)
        for a, b in zip(before_critic, trainer.critic.parameters())
    )


def test_greedy_genome_is_a_detached_copy():
    """It gets evaluated and inserted like any elite; it must not carry grad
    or alias the live parameter."""
    trainer = Td3Trainer(Td3Cfg(), "cpu", seed=0, spec=SPEC)
    genome = trainer.greedy_genome()
    assert not genome.requires_grad
    genome.add_(1.0)
    assert not torch.allclose(genome, trainer.greedy_genome())


# --------------------------------------------------------------------------- #
# Reward / env-stripping wiring
# --------------------------------------------------------------------------- #


def test_velocity_term_alone_still_sums_to_displacement():
    """v1's identity, now scoped to the one term that still carries it.

    Walking-v2 deliberately breaks the *whole* reward's equality with the
    episodic objective — the critic gets alive and upright terms so it can
    teach balance (see :class:`ShapedRewardCfg`). What must not drift is the
    velocity term's meaning: at ``vel_weight`` 1.0 it is still literally
    metres of forward progress, which is what keeps the critic on v1's scale
    and the shaping weights interpretable as "metres per second of posture".
    """
    rc = ShapedRewardCfg()
    assert rc.vel_weight == 1.0
    dt, vx, steps = 0.02, 0.3, 200
    assert sum(rc.vel_weight * vx * dt for _ in range(steps)) == pytest.approx(
        vx * dt * steps, abs=1e-9
    )


def test_shaped_reward_prices_survival_above_a_dive():
    """A policy that dives must not out-earn one that stands still.

    v1's failure in one inequality. Its numbers: a diver that covers 0.4 m and
    falls at 2 s of a 7 s episode scored 0.4 - 0.25*(5/7) = +0.22, while a
    policy that stayed up and went nowhere scored 0.0. The dive won, and the
    archive filled with divers. With the gate the diver is not inserted at
    all; and the critic that guides PG variation now agrees, because the
    terminal penalty is 4x larger and the alive/upright terms pay for the
    5 s the diver spent on the floor.
    """
    rc = ShapedRewardCfg()
    total, fall_step, dt = 350, 100, 0.02

    diver = 0.4 - rc.fall_penalty * (total - fall_step) / total
    stander = (rc.alive_bonus + rc.upright_weight * 1.0) * dt * total
    assert diver < stander


def test_every_velocity_event_term_is_classified():
    """A DR term the stripper does not know about would leave DR silently on."""
    from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
        make_microduck_velocity_env_cfg,
    )

    cfg = make_microduck_velocity_env_cfg(play=False, rough=False)
    unclassified = set(cfg.events) - set(DR_EVENT_TERMS) - set(KEEP_EVENT_TERMS)
    assert not unclassified, (
        f"velocity cfg grew event terms {sorted(unclassified)}; classify them in "
        "qd/pga/evaluate.py before running PGA-ME"
    )
    # BAM's model-field expansion is not DR and must survive the strip.
    assert "expand_bam_friction_fields" in KEEP_EVENT_TERMS
    assert "expand_bam_friction_fields" not in DR_EVENT_TERMS
