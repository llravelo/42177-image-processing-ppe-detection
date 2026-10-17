"""Train YOLO26-M / RT-DETR-L with Ultralytics' own augmentation, configured to the shared family.

Remaining protocol differences from the torchvision models: Ultralytics resizes with its own logic rather than a plain
stretch (a no-op for the 640×640 images that make up most of the dataset), filters cropped boxes with its own
thresholds, and its built-in Albumentations step applies blur / median blur / grayscale / CLAHE at p=0.01 each.
"""

from pathlib import Path

import yaml

from ..classes import DEFAULT_CLASS_NAMES
from .augmentations import BRIGHTNESS, FLIP_P, SCALE_RANGE, TRANSLATE_RANGE

# Override default augmentation parameters to match the shared family.
ULTRALYTICS_AUGMENTATION = {
    "fliplr": FLIP_P,                       # Horizontal flip   (p=0.5)
    "scale": round(SCALE_RANGE[1] - 1, 2),  # Scale jitter      (p=1.0, range 0.75–1.25)
    "translate": TRANSLATE_RANGE[1],        # Translate jitter  (p=1.0, range 10 %)
    "hsv_v": BRIGHTNESS,                    # Brightness jitter (p=1.0, range ±0.4)
    "hsv_s": 0.0,
    "hsv_h": 0.0,
    "mosaic": 0.0,
}


def write_ultralytics_yaml(root: str | Path, class_names: tuple[str, ...] = DEFAULT_CLASS_NAMES) -> Path:
    # Relative paths in data.yaml are resolved against the CWD by Ultralytics, so write absolute ones.
    root = Path(root).resolve()
    config = {
        "path": str(root),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "names": dict(enumerate(class_names)),
    }
    path = root / "ultralytics.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path
