"""Create one immutable random initialization shared by matched smoke arms."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from aeroguard.models.fcos import build_flight_aware_fcos


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("checkpoints/shared_fcos_film_init.pt"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(17)
    model = build_flight_aware_fcos(pretrained=False, min_size=320, max_size=576)
    torch.save(model.state_dict(), args.output)
    summary = {
        "artifact_kind": "shared_random_initialization",
        "seed": 17,
        "weights_origin": "random_no_pretrained_weights",
        "sha256": sha256_file(args.output),
    }
    (args.output.parent / "shared_fcos_film_init.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
