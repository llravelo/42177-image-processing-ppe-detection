import hashlib
import shutil
import tempfile
import urllib.request
import zipfile
from collections.abc import Callable, Iterable
from pathlib import Path

import cv2
import numpy as np
import yaml

from ..classes import DEFAULT_CLASS_NAMES
from .augmentations import SharedAugmentation, resize_example

DATASET_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip"
# Class order in the archive's data.yaml; the ids are kept, only "none" is renamed (see classes.py).
SOURCE_CLASS_NAMES = (
    "helmet", "gloves", "vest", "boots", "goggles", "none", "Person", "no_helmet", "no_goggle", "no_gloves", "no_boots",
)
SPLITS = ("train", "valid", "test")
SOURCE_SPLITS = {"train": "train", "valid": "val", "test": "test"}
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")


def download_dataset(root: str | Path, url: str = DATASET_URL) -> Path:
    """Download Ultralytics Construction-PPE into `root` as <split>/{images,labels}; no-op if it is already there."""
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
            zf.extractall(tmp / "src")
        src = tmp / "src"

        names = yaml.safe_load((src / "data.yaml").read_text())["names"]
        names = tuple(names.values() if isinstance(names, dict) else names)
        if names != SOURCE_CLASS_NAMES:
            raise ValueError(f"Unexpected class order in the archive: {names}")

        # The archive uses images/<split> and labels/<split>; this repo pairs <split>/images with <split>/labels.
        staging = tmp / "out"
        for split, source in SOURCE_SPLITS.items():
            (staging / split).mkdir(parents=True)
            shutil.move(src / "images" / source, staging / split / "images")
            shutil.move(src / "labels" / source, staging / split / "labels")
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


def class_box_stats(
    root: str | Path, split: str = "train", class_names: tuple[str, ...] = DEFAULT_CLASS_NAMES
) -> dict[str, dict[str, float]]:
    """Per-class box counts and the share of boxes whose shorter side is under 16 / 32 px."""
    sides: dict[int, list[float]] = {i: [] for i in range(len(class_names))}
    for image_path in list_images(root, split):
        h, w = cv2.imread(str(image_path)).shape[:2]
        labels, boxes = read_yolo_labels(label_path_for(image_path))
        for label, (_, _, bw, bh) in zip(labels, boxes):
            sides[int(label)].append(min(bw * w, bh * h))
    stats = {}
    for i, name in enumerate(class_names):
        s = np.asarray(sides[i])
        stats[name] = {
            "count": len(s),
            "under_16px": float((s < 16).mean()) if len(s) else float("nan"),
            "under_32px": float((s < 32).mean()) if len(s) else float("nan"),
        }
    return stats


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


def load_training_example(
    image_path: str | Path,
    augmentation: SharedAugmentation,
    epoch: int,
    label_path: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The single entry point every model's training dataset goes through."""
    image, boxes, labels = load_example(image_path, label_path)
    return augmentation(image, boxes, labels, epoch=epoch, key=Path(image_path).name)


def load_eval_example(image_path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validation/test counterpart of `load_training_example`: no augmentation, only the resize to 640×640."""
    return resize_example(*load_example(image_path))


Example = tuple[str, np.ndarray, np.ndarray, np.ndarray]


def verify_same_examples(
    get_example: Callable[[int], Example],
    indices: Iterable[int],
    augmentation: SharedAugmentation,
    epoch: int,
) -> str:
    """Check a model's training dataset against the shared pipeline and return a fingerprint.

    `get_example(i)` must return (image path, RGB uint8 image, pixel xyxy boxes, 0-based labels) as the
    model's dataset produces them for the current epoch. The fingerprint covers image bytes and labels
    only (box coordinates pass through float32 in some stacks), so it is equal across notebooks iff
    the models see the same examples.
    """
    digest = hashlib.sha256()
    for i in indices:
        path, image, boxes, labels = get_example(i)
        ref_image, ref_boxes, ref_labels = load_training_example(path, augmentation, epoch)
        if not np.array_equal(image, ref_image):
            raise AssertionError(f"{path}: image differs from the shared pipeline")
        if not np.array_equal(labels, ref_labels) or not np.allclose(boxes, ref_boxes, atol=1e-2):
            raise AssertionError(f"{path}: boxes/labels differ from the shared pipeline")
        digest.update(Path(path).name.encode())
        digest.update(ref_image.tobytes())
        digest.update(ref_labels.tobytes())
    return digest.hexdigest()[:16]
