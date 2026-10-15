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
# head/torso to tell e.g. Hardhat from NO-Hardhat, so it is dropped rather than kept as label noise.
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


def build_eval_transform() -> A.Compose:
    # Identity for the 640×640 Roboflow export; only resizes if a differently sized image slips in.
    return A.Compose([A.Resize(IMAGE_SIZE, IMAGE_SIZE)], bbox_params=_bbox_params())


def sample_seed(seed: int, epoch: int, key: str) -> int:
    # hashlib rather than hash(): str hashes are salted per process, which would differ per worker.
    digest = hashlib.blake2b(f"{seed}:{epoch}:{key}".encode(), digest_size=4).digest()
    return int.from_bytes(digest, "little")


class SharedAugmentation:
    def __init__(self, seed: int = 0, train: bool = True) -> None:
        self.seed = seed
        self.train = train
        self.transform = build_train_transform(seed) if train else build_eval_transform()

    def __call__(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
        labels: np.ndarray,
        *,
        epoch: int = 0,
        key: str = "",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Augment an RGB uint8 image with pixel xyxy `boxes` and 0-based class `labels`."""
        if self.train:
            self.transform.set_random_seed(sample_seed(self.seed, epoch, key))
        out = self.transform(image=image, bboxes=boxes.reshape(-1, 4), labels=labels)
        return (
            out["image"],
            np.asarray(out["bboxes"], dtype=np.float32).reshape(-1, 4),
            np.asarray(out["labels"], dtype=np.int64),
        )
