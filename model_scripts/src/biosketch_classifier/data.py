from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .preprocessing import adobe_scan_like_preprocess, pil_open_rgb, square_pad_pil


class ManifestDataset(Dataset):
    def __init__(
        self,
        dataset_root: str | Path,
        manifest_csv: str | Path,
        class_to_idx: dict[str, int],
        *,
        split: str,
        transform=None,
        scan_preprocess: bool = False,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.manifest_csv = Path(manifest_csv)
        self.class_to_idx = class_to_idx
        self.split = split
        self.transform = transform
        self.scan_preprocess = scan_preprocess
        self.samples: list[dict[str, Any]] = []

        with self.manifest_csv.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["split"] == split:
                    label = row["label"]
                    self.samples.append(
                        {
                            "path": self.dataset_root / row["relative_path"],
                            "relative_path": row["relative_path"],
                            "label": label,
                            "label_idx": self.class_to_idx[label],
                            "group_id": row.get("group_id", ""),
                        }
                    )
        if not self.samples:
            raise ValueError(f"No samples found for split={split!r} in {manifest_csv}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        item = self.samples[idx]
        path = item["path"]
        if self.scan_preprocess:
            img = adobe_scan_like_preprocess(path)
        else:
            img = square_pad_pil(pil_open_rgb(path))
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.tensor(item["label_idx"], dtype=torch.long)

    @property
    def targets(self) -> list[int]:
        return [int(s["label_idx"]) for s in self.samples]
