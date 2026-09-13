"""Both predictor heads: output shapes from random CPU inputs, identical architecture, geometry."""

from __future__ import annotations

import pytest
import torch

from hyperbolic_world_model.geometry import Euclidean, Lorentz, Manifold, PoincareBall
from hyperbolic_world_model.models.predictors import (
    ActionConditionedPredictor,
    EuclideanHead,
    HyperbolicHead,
)
from hyperbolic_world_model.models.registry import build_predictor

B, T, ENC, LAT, ACT, HID, AEMB = 4, 6, 32, 8, 5, 24, 12
KW = dict(
    encoder_dim=ENC,
    latent_dim=LAT,
    action_dim=ACT,
    hidden_dim=HID,
    n_layers=2,
    action_embed_dim=AEMB,
    seed=3,
)

HEAD_CASES = [
    pytest.param(lambda: EuclideanHead(**KW), id="euclidean"),
    pytest.param(lambda: HyperbolicHead(manifold=Euclidean(), **KW), id="hyperbolic-on-euclidean"),
    pytest.param(
        lambda: HyperbolicHead(manifold=PoincareBall(c=-1.0), **KW), id="hyperbolic-poincare"
    ),
    pytest.param(
        lambda: HyperbolicHead(manifold=PoincareBall(c=-0.5), **KW),
        id="hyperbolic-poincare(c=-0.5)",
    ),
    pytest.param(lambda: HyperbolicHead(manifold=Lorentz(c=-1.0), **KW), id="hyperbolic-lorentz"),
]


@pytest.fixture(params=HEAD_CASES)
def head(request: pytest.FixtureRequest) -> ActionConditionedPredictor:
    return request.param()


def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
    return torch.randn(B, T, ENC), torch.randn(B, T - 1, ACT)


def test_output_shapes_from_random_inputs_on_cpu(head: ActionConditionedPredictor) -> None:
    z, actions = _inputs()
    d = head.ambient_dim
    assert d == LAT + head.manifold.ambient_dim_offset
    state = head.embed(z)
    assert state.shape == (B, T, d) and state.device.type == "cpu"
    nxt = head.step(state[:, 0], actions[:, 0])
    assert nxt.shape == (B, d)
    roll = head.rollout(state[:, 0], actions)
    assert roll.shape == (B, T - 1, d)
    pred, target = head(z, actions)
    assert pred.shape == target.shape == (B, T - 1, d)
    loss = head.loss(pred, target)
    assert loss.shape == () and torch.isfinite(loss) and loss >= 0
    assert head.fuse(state[:, 0], actions[:, 0]).shape == (B, LAT + AEMB)
    assert head.delta(state[:, 0], actions[:, 0]).shape == (B, LAT)
    assert head.coordinates(state).shape == (B, T, LAT)


def test_outputs_stay_on_the_manifold_and_updates_are_bounded(
    head: ActionConditionedPredictor,
) -> None:
    z, actions = _inputs()
    m: Manifold = head.manifold
    state = head.embed(z * 3)
    assert m.check_point(state).all()
    # Force a large proposed update so the max_step clip is exercised.
    with torch.no_grad():
        head.dynamics[-1].weight.fill_(2.0)
        head.dynamics[-1].bias.fill_(2.0)
    roll = head.rollout(state[:, 0], actions)
    assert m.check_point(roll).all()
    delta = head.delta(state[:, 0], actions[:, 0])
    assert torch.all(delta.norm(dim=-1) <= head.max_step + 1e-5)
    assert torch.all(
        m.dist(state[:, 0], head.step(state[:, 0], actions[:, 0])) <= head.max_step + 1e-3
    )


def test_gradients_reach_every_trainable_parameter_and_not_the_projection(
    head: ActionConditionedPredictor,
) -> None:
    z, actions = _inputs()
    with torch.no_grad():  # non-zero output layer so the loss depends on every parameter
        head.dynamics[-1].weight.normal_(std=0.1)
        head.dynamics[-1].bias.normal_(std=0.1)
    proj_before = head.projection.clone()
    pred, target = head(z, actions)
    head.loss(pred, target).backward()
    for name, p in head.named_parameters():
        assert p.requires_grad, name
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    assert not head.projection.requires_grad and torch.equal(head.projection, proj_before)
    assert len(head.trainable_parameters()) == len(list(head.parameters()))


def test_action_conditions_the_prediction(head: ActionConditionedPredictor) -> None:
    z, _ = _inputs()
    with torch.no_grad():
        head.dynamics[-1].weight.normal_(std=0.1)
    state = head.embed(z)[:, 0]
    a1, a2 = torch.randn(B, ACT), torch.randn(B, ACT)
    assert not torch.allclose(head.step(state, a1), head.step(state, a2))


def test_heads_are_architecturally_identical() -> None:
    euc = EuclideanHead(**KW)
    for m in (PoincareBall(c=-1.0), Lorentz(c=-1.0), Euclidean()):
        hyp = HyperbolicHead(manifold=m, **KW)
        assert euc.architecture_signature() == hyp.architecture_signature()
        assert list(euc.state_dict()) == list(hyp.state_dict())
        assert sum(p.numel() for p in euc.parameters()) == sum(p.numel() for p in hyp.parameters())
        # Same seed -> identical frozen projection and identical initial weights.
        for (n1, p1), (_n2, p2) in zip(
            euc.state_dict().items(), hyp.state_dict().items(), strict=True
        ):
            assert torch.equal(p1, p2), n1


def test_same_seed_gives_identical_initial_weights_and_different_seeds_differ() -> None:
    a = HyperbolicHead(manifold=PoincareBall(c=-1.0), **KW)
    torch.randn(1000)  # perturb the global RNG between constructions
    b = HyperbolicHead(manifold=Lorentz(c=-1.0), **KW)
    for (n, p1), (_, p2) in zip(a.state_dict().items(), b.state_dict().items(), strict=True):
        assert torch.equal(p1, p2), n
    c = HyperbolicHead(manifold=PoincareBall(c=-1.0), **{**KW, "seed": 4})
    assert not torch.equal(a.action_embed[0].weight, c.action_embed[0].weight)
    assert not torch.equal(a.projection, c.projection)


def test_max_radius_retraction_bounds_every_latent(head: ActionConditionedPredictor) -> None:
    m = head.manifold
    far = head.embed(
        torch.randn(B, T, ENC) * 50
    )  # would land far beyond max_radius without the guard
    assert torch.all(m.dist0(far) <= head.max_radius + 1e-3)
    near = head.embed(torch.randn(B, T, ENC) * 0.1)
    assert torch.allclose(head.retract(near), near)  # no-op inside the radius
    # Retraction keeps the direction: the retracted point lies on the geodesic from the origin.
    v_far = torch.randn(B, LAT)
    raw = m.expmap0(
        m.tangent0_from_euclidean(12.0 * v_far / v_far.norm(dim=-1, keepdim=True))
    )  # 12 units out, finite everywhere
    pulled = head.retract(raw)
    assert torch.all(m.dist0(pulled) <= head.max_radius + 1e-3)
    unit_raw = m.logmap0(raw) / m.dist0(raw).unsqueeze(-1)
    unit_pulled = m.logmap0(pulled) / m.dist0(pulled).unsqueeze(-1)
    assert torch.allclose(unit_raw, unit_pulled, atol=1e-3)
    off = HyperbolicHead(manifold=Euclidean(), max_radius=None, **KW)
    assert off.max_radius is None and torch.equal(
        off.retract(raw[:, : off.ambient_dim]), raw[:, : off.ambient_dim]
    )


def test_steps_are_measured_in_geodesic_units_in_every_geometry(
    head: ActionConditionedPredictor,
) -> None:
    """A proposed update of norm r moves the state by geodesic distance r, whatever the geometry."""
    m = head.manifold
    z = torch.randn(B, ENC) * 0.05
    state = head.embed(z)
    # Make the dynamics output a constant vector of known norm.
    with torch.no_grad():
        head.dynamics[-1].weight.zero_()
        head.dynamics[-1].bias.zero_()
        head.dynamics[-1].bias[0] = 1.5
    nxt = head.step(state, torch.randn(B, ACT))
    assert torch.allclose(m.dist(state, nxt), torch.full((B,), 1.5), atol=1e-4)
    assert torch.allclose(head.coordinates(state).norm(dim=-1), m.dist0(state), atol=1e-4)


def test_poincare_and_lorentz_heads_are_isometric_twins() -> None:
    """Same weights, same curvature: the ball head and the hyperboloid head trace the same rollout."""
    from hyperbolic_world_model.geometry.lorentz import lorentz_to_poincare

    c = -0.5
    ball = HyperbolicHead(manifold=PoincareBall(c=c), **KW).double()
    hyp = HyperbolicHead(manifold=Lorentz(c=c), **KW).double()
    with torch.no_grad():
        ball.dynamics[-1].weight.normal_(std=0.2)
        ball.dynamics[-1].bias.normal_(std=0.2)
    hyp.load_state_dict(ball.state_dict())
    z, actions = (
        torch.randn(B, T, ENC, dtype=torch.float64),
        torch.randn(B, T - 1, ACT, dtype=torch.float64),
    )
    sb, sh = ball.embed(z), hyp.embed(z)
    assert torch.allclose(lorentz_to_poincare(sh, c), sb, atol=1e-9)
    rb, rh = ball.rollout(sb[:, 0], actions), hyp.rollout(sh[:, 0], actions)
    assert torch.allclose(lorentz_to_poincare(rh, c), rb, atol=1e-7)
    pb, tb = ball(z, actions)
    ph, th = hyp(z, actions)
    assert torch.allclose(ball.loss(pb, tb), hyp.loss(ph, th), atol=1e-8)


def test_hyperbolic_head_on_euclidean_manifold_equals_euclidean_head() -> None:
    euc, hyp = EuclideanHead(**KW), HyperbolicHead(manifold=Euclidean(), **KW)
    with torch.no_grad():
        euc.dynamics[-1].weight.normal_(std=0.1)
        euc.dynamics[-1].bias.normal_(std=0.1)
    hyp.load_state_dict(euc.state_dict())
    z, actions = _inputs()
    pe, te = euc(z, actions)
    ph, th = hyp(z, actions)
    assert torch.allclose(pe, ph, atol=1e-6) and torch.equal(te, th)
    assert torch.allclose(euc.loss(pe, te), hyp.loss(ph, th))
    assert torch.allclose(euc.rollout(te[:, 0], actions), hyp.rollout(th[:, 0], actions), atol=1e-6)


def test_embedding_uses_expmap0_and_is_frozen() -> None:
    m = PoincareBall(c=-1.0)
    head = HyperbolicHead(manifold=m, **KW)
    z = torch.randn(B, ENC)
    from hyperbolic_world_model.geometry.utils import clip_norm

    v = clip_norm(z @ head.projection * head.embed_scale, head.max_radius)
    assert torch.allclose(head.embed(z), m.expmap0(m.tangent0_from_euclidean(v / m.lambda0)))
    assert torch.allclose(m.dist0(head.embed(z)), v.norm(dim=-1), atol=1e-5)  # geodesic units
    assert head.projection.shape == (ENC, LAT) and "projection" in dict(head.named_buffers())
    # Orthonormal columns (or rows when latent_dim > encoder_dim).
    assert torch.allclose(head.projection.T @ head.projection, torch.eye(LAT), atol=1e-5)
    small = HyperbolicHead(manifold=m, encoder_dim=4, latent_dim=8, action_dim=ACT)
    assert torch.allclose(small.projection @ small.projection.T, torch.eye(4), atol=1e-5)


def test_euclidean_head_rejects_curved_manifold_and_registry_builds_both() -> None:
    with pytest.raises(TypeError):
        EuclideanHead(manifold=PoincareBall(c=-1.0), **KW)
    cfg = dict(type="euclidean", latent_dim=LAT, hidden_dim=HID, n_layers=2, action_embed_dim=AEMB)
    e = build_predictor(cfg, Euclidean(), ENC, ACT)
    h = build_predictor({**cfg, "type": "hyperbolic"}, Lorentz(c=-2.0), ENC, ACT)
    assert isinstance(e, EuclideanHead) and isinstance(h, HyperbolicHead)
    assert e.architecture_signature() == h.architecture_signature()
    with pytest.raises(ValueError, match="geometry=euclidean"):
        build_predictor(cfg, PoincareBall(c=-1.0), ENC, ACT)
    with pytest.raises(KeyError):
        build_predictor({**cfg, "type": "spherical"}, Euclidean(), ENC, ACT)
    with pytest.raises(ValueError):
        e(torch.randn(B, T, ENC), torch.randn(B, T, ACT))  # actions must be T - 1 long


def test_max_radius_is_capped_at_the_representable_radius() -> None:
    """A radius the ball cannot represent is reduced, identically for both hyperbolic heads."""
    from hyperbolic_world_model.geometry.utils import reliable_radius

    cap = reliable_radius(-4.0, torch.float32)
    ball = HyperbolicHead(manifold=PoincareBall(c=-4.0), max_radius=8.0, **KW)
    hyp = HyperbolicHead(manifold=Lorentz(c=-4.0), max_radius=8.0, **KW)
    assert ball.requested_max_radius == 8.0 and ball.max_radius == pytest.approx(cap)
    assert hyp.max_radius == pytest.approx(cap) and cap < 3.2
    far = torch.randn(B, T, ENC) * 50
    r_ball, r_hyp = ball.manifold.dist0(ball.embed(far)), hyp.manifold.dist0(hyp.embed(far))
    assert torch.all(r_ball <= cap + 1e-3) and torch.all(r_hyp <= cap + 1e-3)
    # Isometric twins: the same guard gives the same radii in both models.
    assert torch.allclose(r_ball, r_hyp, atol=1e-3)
    # Within the representable region nothing changes; flat space is never capped.
    assert HyperbolicHead(manifold=PoincareBall(c=-1.0), max_radius=4.0, **KW).max_radius == 4.0
    assert EuclideanHead(max_radius=8.0, **KW).max_radius == 8.0
    assert HyperbolicHead(manifold=PoincareBall(c=-1.0), **KW).max_radius == 4.0  # default
