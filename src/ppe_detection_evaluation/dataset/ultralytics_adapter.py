"""Train YOLO26-M / RT-DETR-L with Ultralytics' default augmentation, except mosaic, which is turned off."""

from pathlib import Path

import yaml

from ..classes import DEFAULT_CLASS_NAMES

ULTRALYTICS_AUGMENTATION = {"mosaic": 0.0}


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
