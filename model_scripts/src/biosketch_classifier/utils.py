from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def save_json(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def load_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def inverse_frequency_class_weights(targets: list[int], num_classes: int) -> torch.Tensor:
    counts = np.bincount(np.array(targets, dtype=np.int64), minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    total = counts.sum()
    weights = total / (num_classes * counts)
    # Normalize around 1.0 so the loss scale stays stable.
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def sample_weights_from_targets(targets: list[int], num_classes: int) -> torch.DoubleTensor:
    class_weights = inverse_frequency_class_weights(targets, num_classes).double().numpy()
    return torch.DoubleTensor([class_weights[t] for t in targets])
