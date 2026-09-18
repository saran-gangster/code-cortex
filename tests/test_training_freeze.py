import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lightning")

from aeroguard.training import AeroGuardDetectorModule


class TinyFlightAwareDetector(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.detector = torch.nn.Sequential(
            torch.nn.Conv2d(3, 4, kernel_size=1),
            torch.nn.BatchNorm2d(4),
        )
        self.film = torch.nn.Linear(8, 8)


def test_frozen_visual_mode_keeps_batch_norm_buffers_in_evaluation_mode():
    detector = TinyFlightAwareDetector()
    for parameter in detector.detector.parameters():
        parameter.requires_grad_(False)

    module = AeroGuardDetectorModule(detector, freeze_visual_detector=True)
    module.train()

    assert module.training is True
    assert detector.training is True
    assert detector.detector.training is True
    assert detector.detector[0].training is True
    assert detector.detector[1].training is False
    assert detector.film.training is True


def test_frozen_visual_mode_rejects_trainable_visual_parameters():
    with pytest.raises(ValueError, match="requires_grad=False"):
        AeroGuardDetectorModule(
            TinyFlightAwareDetector(),
            freeze_visual_detector=True,
        )
