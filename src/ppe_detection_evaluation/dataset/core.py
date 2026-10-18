import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np
import yaml

from ..classes import DEFAULT_CLASS_NAMES
from .augmentations import resize_example

DATASET_URL = "https://www.kaggle.com/api/v1/datasets/download/snehilsanyal/construction-site-safety-image-dataset-roboflow"
# The archive also holds a reference YOLOv8 run and raw videos; only this folder is the dataset.
ARCHIVE_DATA_DIR = "css-data"
SPLITS = ("train", "valid", "test")
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")


def download_dataset(root: str | Path, url: str = DATASET_URL) -> Path:
    """Download the Construction Site Safety dataset into `root` as <split>/{images,labels}; no-op if it is already there.

    All ten classes are kept. The archive has no data.yaml, so one listing DEFAULT_CLASS_NAMES is written; the label
    ids already follow that order (the dataset page's class list).
    """
    root = Path(root)
    if (root / "data.yaml").exists():
        return root

    root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root.parent) as tmp:
        tmp = Path(tmp)
        archive = tmp / "dataset.zip"
        with urllib.request.urlopen(url) as resp, archive.open("wb") as f:
            shutil.copyfileobj(resp, f)
        with zipfile.ZipFile(archive) as zf:
            members = [m for m in zf.namelist() if m.startswith(f"{ARCHIVE_DATA_DIR}/")]
            zf.extractall(tmp / "src", members=members)
        staging = tmp / "src" / ARCHIVE_DATA_DIR

        config = {"train": "train/images", "val": "valid/images", "test": "test/images"}
        config |= {"nc": len(DEFAULT_CLASS_NAMES), "names": list(DEFAULT_CLASS_NAMES)}
        (staging / "data.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        staging.rename(root)  # last step, so an interrupted download never leaves a half-built dataset
    return root


def check_dataset(root: str | Path, class_names: tuple[str, ...] = DEFAULT_CLASS_NAMES) -> dict[str, int]:
    """Check the class order and return the number of images per split."""
    root = Path(root)
    names = yaml.safe_load((root / "data.yaml").read_text())["names"]
    names = tuple(names.values() if isinstance(names, dict) else names)
    if names != tuple(class_names):
        raise ValueError(f"Class order in data.yaml {names} does not match {tuple(class_names)}")
    return {split: len(list_images(root, split)) for split in SPLITS if (root / split).exists()}


def class_box_counts(
    root: str | Path, split: str = "train", class_names: tuple[str, ...] = DEFAULT_CLASS_NAMES
) -> dict[str, int]:
    counts = np.zeros(len(class_names), dtype=int)
    for image_path in list_images(root, split):
        labels, _ = read_yolo_labels(label_path_for(image_path))
        np.add.at(counts, labels.astype(int), 1)
    return dict(zip(class_names, counts.tolist()))


def list_images(root: str | Path, split: str) -> list[Path]:
    images_dir = Path(root) / split / "images"
    return sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def label_path_for(image_path: str | Path) -> Path:
    # Same rule as Ultralytics' img2label_paths, so both stacks pair images and labels identically.
    image_path = Path(image_path)
    return image_path.parent.parent / "labels" / f"{image_path.stem}.txt"


def read_yolo_labels(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Return 0-based class ids and normalized cxcywh boxes; a missing file means no objects."""
    path = Path(path)
    rows = []
    if path.exists():
        rows = [line.split() for line in path.read_text().splitlines() if line.strip()]
    # Polygon rows (class + >4 coords) are reduced to their bounding box, as Ultralytics does.
    labels, boxes = [], []
    for row in rows:
        values = np.asarray(row[1:], dtype=np.float64)
        if len(values) > 4:
            xy = values.reshape(-1, 2)
            (x1, y1), (x2, y2) = xy.min(0), xy.max(0)
            values = np.array([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1])
        labels.append(int(row[0]))
        boxes.append(values)
    return np.asarray(labels, dtype=np.int64), np.asarray(boxes, dtype=np.float64).reshape(-1, 4)


def cxcywh_to_xyxy(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4) * [width, height, width, height]
    xyxy = np.column_stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2])
    return np.clip(xyxy, 0, [width, height, width, height])


def xyxy_to_cxcywh(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    cxcywh = np.column_stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2, b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]])
    return cxcywh / [width, height, width, height]


def load_example(image_path: str | Path, label_path: str | Path | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load an RGB uint8 image with pixel xyxy boxes and 0-based labels."""
    image_path = Path(image_path)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    labels, boxes = read_yolo_labels(label_path or label_path_for(image_path))
    h, w = image.shape[:2]
    return image, cxcywh_to_xyxy(boxes, w, h).astype(np.float32), labels


def load_eval_example(image_path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validation/test preprocessing: no augmentation, only the resize to 640×640."""
    return resize_example(*load_example(image_path))

