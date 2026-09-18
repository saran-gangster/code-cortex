import pytest

torch = pytest.importorskip("torch")

from aeroguard.models import GatedResidualFiLM


def test_zero_initialized_film_is_identity_for_single_map_and_pyramid():
    module = GatedResidualFiLM([3, 5], state_dim=8).eval()
    state = torch.randn(2, 8)
    maps = (torch.randn(2, 3, 4, 4), torch.randn(2, 5, 2, 2))
    output = module(maps, state, torch.ones(2))
    assert isinstance(output, tuple)
    assert torch.equal(output[0], maps[0])
    assert torch.equal(output[1], maps[1])
    single_level = GatedResidualFiLM(3, state_dim=8).eval()
    assert torch.equal(single_level(maps[0], state, torch.ones(2)), maps[0])


def test_nonfinite_state_is_sanitized_and_mask_is_per_sample():
    module = GatedResidualFiLM(3, state_dim=8)
    with torch.no_grad():
        module.projections[0].weight.normal_()
        module.projections[0].bias.normal_()
    features = torch.ones(2, 3, 2, 2)
    state = torch.tensor([[float("nan"), 1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7, float("inf")]])
    output = module(features, state, torch.tensor([1.0, 0.0]))
    assert torch.isfinite(output).all()
    assert torch.equal(output[1], features[1])
    assert not torch.equal(output[0], features[0])


def test_final_projection_receives_gradients_without_hidden_state():
    module = GatedResidualFiLM(2, state_dim=8)
    features = torch.randn(2, 2, 3, 3, requires_grad=True)
    state = torch.randn(2, 8)
    loss = module(features, state, torch.ones(2)).square().mean()
    loss.backward()
    assert torch.isfinite(features.grad).all()
    assert module.projections[0].weight.grad is not None
    assert torch.isfinite(module.projections[0].weight.grad).all()
