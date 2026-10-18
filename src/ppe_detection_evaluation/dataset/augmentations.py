"""Training-time augmentation for the torchvision models: horizontal flip only.

The Ultralytics models use Ultralytics' defaults with mosaic off instead (`ULTRALYTICS_AUGMENTATION` in
`ultralytics_adapter.py`).
"""

import albumentations as A
import cv2
import numpy as np

from ..experiment import IMAGE_SIZE

FLIP_P = 0.5


def build_train_transform(seed: int | None = None) -> A.Compose:
    return A.Compose(
        [
            A.Resize(IMAGE_SIZE, IMAGE_SIZE),
            A.HorizontalFlip(p=FLIP_P),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
        seed=seed,
    )


def resize_example(
    image: np.ndarray, boxes: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluation preprocessing: stretch to 640×640 like training does, scaling the boxes and never dropping any."""
    h, w = image.shape[:2]
    resized = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)  # A.Resize's default
    scale = np.array([IMAGE_SIZE / w, IMAGE_SIZE / h, IMAGE_SIZE / w, IMAGE_SIZE / h], dtype=np.float32)
    return resized, (boxes.reshape(-1, 4) * scale).astype(np.float32), labels
