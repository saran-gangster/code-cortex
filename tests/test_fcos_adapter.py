import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from aeroguard.models.fcos import build_flight_aware_fcos


def test_flight_aware_fcos_runs_explicit_state_training_path():
    model = build_flight_aware_fcos(pretrained=False, min_size=64, max_size=96).train()
    image = torch.rand(3, 64, 80)
    state = torch.zeros(1, 8)
    mask = torch.ones(1)
    target = {
        "boxes": torch.tensor([[8.0, 8.0, 35.0, 40.0]]),
        "labels": torch.tensor([2], dtype=torch.int64),
    }
    losses = model([image], state, mask, [target])
    total = sum(losses.values())
    assert torch.isfinite(total)
    total.backward()
    projection = model.film.projections[0]
    assert projection.weight.grad is not None
    assert torch.isfinite(projection.weight.grad).all()


def test_flight_aware_fcos_rejects_misaligned_state_batch():
    model = build_flight_aware_fcos(pretrained=False, min_size=64, max_size=96).eval()
    with pytest.raises(ValueError, match="state must have shape"):
        model([torch.rand(3, 64, 64)], torch.zeros(2, 8), torch.ones(1))
