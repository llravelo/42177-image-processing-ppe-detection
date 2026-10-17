"""Training-time augmentation family shared by all four detectors (see AUGMENTATIONS.md).

The torchvision models use `build_train_transform`; the Ultralytics models get the same steps and ranges through
`ULTRALYTICS_AUGMENTATION` in `ultralytics_adapter.py`, which is derived from the constants below.
"""

import albumentations as A
import cv2
import numpy as np

from ..experiment import IMAGE_SIZE

# A1 — horizontal flip
FLIP_P = 0.5
# A2 — scale + translate jitter
SCALE_RANGE = (0.75, 1.25)
TRANSLATE_RANGE = (-0.1, 0.1)
FILL_VALUE = 114
# A3 — brightness jitter. Saturation and hue stay fixed: hi-vis colour is what separates vest from no_vest.
BRIGHTNESS = 0.4

# A box that loses more than 60 % of its area to the frame edge no longer shows enough of the
# head/torso to tell e.g. helmet from no_helmet, so it is dropped rather than kept as label noise.
MIN_VISIBILITY = 0.4
MIN_BOX_SIZE = 3

def build_train_transform(seed: int | None = None) -> A.Compose:
    return A.Compose(
        [
            A.Resize(IMAGE_SIZE, IMAGE_SIZE),
            A.HorizontalFlip(p=FLIP_P),
            A.Affine(
                scale=SCALE_RANGE,
                keep_ratio=True,  # one gain for both axes, as in Ultralytics
                translate_percent=TRANSLATE_RANGE,
                fill=FILL_VALUE,
                border_mode=cv2.BORDER_CONSTANT,
                p=1.0,
            ),
            # contrast, saturation and hue must be passed as 0: ColorJitter's defaults would jitter them.
            A.ColorJitter(brightness=BRIGHTNESS, contrast=0.0, saturation=0.0, hue=0.0, p=1.0),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["labels"],
            min_visibility=MIN_VISIBILITY,
            min_width=MIN_BOX_SIZE,
            min_height=MIN_BOX_SIZE,
            clip=True,
        ),
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
