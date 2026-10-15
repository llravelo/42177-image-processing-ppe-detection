import hashlib
import json
import shutil
import time
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Callable, Iterable
from pathlib import Path

import cv2
import numpy as np
import yaml

from .augment import SharedAugmentation
from .classes import DEFAULT_CLASS_NAMES

ROBOFLOW_WORKSPACE = "roboflow-universe-projects"
ROBOFLOW_PROJECT = "construction-site-safety"
ROBOFLOW_VERSION = 30
SPLITS = ("train", "valid", "test")
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")


def download_dataset(
    root: str | Path,
    api_key: str,
    *,
    workspace: str = ROBOFLOW_WORKSPACE,
    project: str = ROBOFLOW_PROJECT,
    version: int = ROBOFLOW_VERSION,
    export_format: str = "yolov8",
    timeout_s: float = 600.0,
) -> Path:
    """Download a Roboflow export in YOLO TXT format into `root`; no-op if it is already there."""
    root = Path(root)
    if (root / "data.yaml").exists():
        return root

    query = urllib.parse.urlencode({"api_key": api_key})
    url = f"https://api.roboflow.com/{workspace}/{project}/{version}/{export_format}?{query}"
    # Roboflow generates exports lazily; the endpoint returns progress until the zip link is ready.
    deadline = time.monotonic() + timeout_s
    while True:
        with urllib.request.urlopen(url) as resp:
            payload = json.load(resp)
        link = payload.get("export", {}).get("link")
        if link:
            break
        if time.monotonic() > deadline:
            raise TimeoutError(f"Roboflow export not ready after {timeout_s:.0f}s: {payload}")
        time.sleep(5)

    root.mkdir(parents=True, exist_ok=True)
    archive = root / "export.zip"
    with urllib.request.urlopen(link) as resp, archive.open("wb") as f:
        shutil.copyfileobj(resp, f)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(root)
    archive.unlink()
    return root


def check_dataset(root: str | Path, class_names: tuple[str, ...] = DEFAULT_CLASS_NAMES) -> dict:
    """Check class order and report split sizes plus the Roboflow preprocessing/augmentation notes."""
    root = Path(root)
    data_yaml = yaml.safe_load((root / "data.yaml").read_text())
    names = data_yaml["names"]
    names = tuple(names.values() if isinstance(names, dict) else names)
    if names != tuple(class_names):
        raise ValueError(f"Class order in data.yaml {names} does not match {tuple(class_names)}")

    notes = []
    readme = root / "README.roboflow.txt"
    if readme.exists():
        notes = [
            line.strip()
            for line in readme.read_text().splitlines()
            if any(word in line.lower() for word in ("augment", "resize", "orient", "following"))
        ]
    return {
        "splits": {split: len(list_images(root, split)) for split in SPLITS if (root / split).exists()},
        "roboflow_notes": notes,
    }


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


def write_ultralytics_yaml(root: str | Path, class_names: tuple[str, ...] = DEFAULT_CLASS_NAMES) -> Path:
    # Roboflow's data.yaml uses relative paths that Ultralytics resolves against the CWD, so write absolute ones.
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
