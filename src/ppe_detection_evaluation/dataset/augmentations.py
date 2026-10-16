"""Shared training-time augmentation (see AUGMENTATIONS.md).

Every detector trains on the output of `SharedAugmentation`, which reseeds the pipeline from
(seed, epoch, image name). The augmented example for a given image and epoch is therefore
identical across models, DataLoader workers, batch sizes and shuffle orders.
"""

import hashlib

import albumentations as A
import cv2
import numpy as np

IMAGE_SIZE = 640

# A1 — horizontal flip
FLIP_P = 0.5
# A2 — scale + translate jitter
SCALE_RANGE = (0.75, 1.25)
TRANSLATE_RANGE = (-0.1, 0.1)
FILL_VALUE = 114
# A3 — photometric jitter
BRIGHTNESS = 0.4
CONTRAST = 0.3
SATURATION = 0.5
HUE = 0.015
# A4 — capture degradation, one of blur / noise / JPEG
DEGRADATION_P = 0.25
BLUR_SIGMA_RANGE = (0.1, 1.5)
NOISE_STD_RANGE = (0.02, 0.06)
JPEG_QUALITY_RANGE = (40, 90)

# A box that loses more than 60 % of its area to the frame edge no longer shows enough of the
# head/torso to tell e.g. helmet from no_helmet, so it is dropped rather than kept as label noise.
MIN_VISIBILITY = 0.4
MIN_BOX_SIZE = 3


def _bbox_params() -> A.BboxParams:
    return A.BboxParams(
        format="pascal_voc",
        label_fields=["labels"],
        min_visibility=MIN_VISIBILITY,
        min_width=MIN_BOX_SIZE,
        min_height=MIN_BOX_SIZE,
        clip=True,
    )


def build_train_transform(seed: int | None = None) -> A.Compose:
    return A.Compose(
        [
            A.Resize(IMAGE_SIZE, IMAGE_SIZE),
            A.HorizontalFlip(p=FLIP_P),
            A.Affine(
                scale=SCALE_RANGE,
                translate_percent=TRANSLATE_RANGE,
                fill=FILL_VALUE,
                border_mode=cv2.BORDER_CONSTANT,
                p=1.0,
            ),
            A.ColorJitter(brightness=BRIGHTNESS, contrast=CONTRAST, saturation=SATURATION, hue=HUE, p=1.0),
            A.OneOf(
                [
                    # blur_limit=0 derives the kernel size from sigma
                    A.GaussianBlur(blur_limit=0, sigma_limit=BLUR_SIGMA_RANGE, p=1.0),
                    A.GaussNoise(std_range=NOISE_STD_RANGE, p=1.0),
                    A.ImageCompression(quality_range=JPEG_QUALITY_RANGE, p=1.0),
                ],
                p=DEGRADATION_P,
            ),
        ],
        bbox_params=_bbox_params(),
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


def sample_seed(seed: int, epoch: int, key: str) -> int:
    # hashlib rather than hash(): str hashes are salted per process, which would differ per worker.
    digest = hashlib.blake2b(f"{seed}:{epoch}:{key}".encode(), digest_size=4).digest()
    return int.from_bytes(digest, "little")


class SharedAugmentation:
    """The training augmentation, reseeded per image so it is identical across models and DataLoader workers.

    Albumentations otherwise draws from a generator whose state depends on the worker process and on how many
    images it has already augmented (batch size, shuffle order), so each model would see different examples.
    """

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self.transform = build_train_transform(seed)

    def __call__(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
        labels: np.ndarray,
        *,
        epoch: int,
        key: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Augment an RGB uint8 image with pixel xyxy `boxes` and 0-based class `labels`."""
        self.transform.set_random_seed(sample_seed(self.seed, epoch, key))
        out = self.transform(image=image, bboxes=boxes.reshape(-1, 4), labels=labels)
        return (
            out["image"],
            np.asarray(out["bboxes"], dtype=np.float32).reshape(-1, 4),
            np.asarray(out["labels"], dtype=np.int64),
        )
