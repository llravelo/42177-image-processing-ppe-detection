"""Train YOLO26-M / RT-DETR-L on the shared augmentation instead of Ultralytics' own.

The training datasets override `build_transforms`, Ultralytics' hook for defining augmentation, with two steps:
`SharedAugmentationTransform`, which produces each example through `load_training_example` (the same function the
torchvision dataset uses), and Ultralytics' `Format`, which converts it to the tensors the trainer expects.
"""

from pathlib import Path
from typing import Any

import numpy as np
import yaml
from ultralytics.cfg import get_cfg
from ultralytics.data.augment import Compose, Format
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.models.rtdetr.val import RTDETRDataset
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import colorstr
from ultralytics.utils.instance import Instances
from ultralytics.utils.torch_utils import unwrap_model

from ..classes import DEFAULT_CLASS_NAMES
from .augmentations import SharedAugmentation
from .core import Example, load_training_example

# The datasets never build Ultralytics' augmentation, so these are zeroed only so the saved args.yaml documents that
# none ran. multi_scale is the exception: the trainer applies it to whole batches, outside the dataset.
DISABLE_BUILTIN_AUGMENTATION: dict[str, Any] = {
    "mosaic": 0.0,
    "close_mosaic": 0,
    "mixup": 0.0,
    "cutmix": 0.0,
    "copy_paste": 0.0,
    "fliplr": 0.0,
    "flipud": 0.0,
    "scale": 0.0,
    "translate": 0.0,
    "degrees": 0.0,
    "shear": 0.0,
    "perspective": 0.0,
    "hsv_h": 0.0,
    "hsv_s": 0.0,
    "hsv_v": 0.0,
    "bgr": 0.0,
    "multi_scale": 0.0,
    "rect": False,
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

class SharedAugmentationTransform:
    """Ultralytics transform that replaces a sample's image and labels with the shared pipeline's output.

    The image is re-read through `load_training_example` instead of using the one Ultralytics loaded: Ultralytics
    resizes YOLO images keeping their aspect ratio, and its label cache can drop or merge boxes, so either would
    make these models see different examples than the torchvision ones.
    """

    def __init__(self, dataset: "_SharedAugmentationMixin") -> None:
        self.dataset = dataset

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        image, boxes, labels = load_training_example(
            sample["im_file"], self.dataset.shared_augmentation, self.dataset.epoch
        )
        h, w = image.shape[:2]
        sample.update(
            img=np.ascontiguousarray(image[..., ::-1]),  # Ultralytics works in BGR; Format flips back
            ori_shape=(h, w),
            resized_shape=(h, w),
            ratio_pad=(1.0, 1.0),
            cls=labels.reshape(-1, 1).astype(np.float32),
            instances=Instances(
                boxes.astype(np.float32),
                np.zeros((0, 1000, 2), dtype=np.float32),
                bbox_format="xyxy",
                normalized=False,
            ),
        )
        return sample


class _SharedAugmentationMixin:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.epoch = 0
        self.shared_augmentation = SharedAugmentation(seed=kwargs["hyp"].seed)
        super().__init__(*args, **kwargs)

    def build_transforms(self, hyp) -> Compose:
        return Compose([SharedAugmentationTransform(self), Format()])

    def as_raw_example(self, index: int) -> Example:
        """What the model is actually fed, converted back to (path, RGB uint8, xyxy, 0-based labels)."""
        sample = self[index]
        image = sample["img"].permute(1, 2, 0).numpy()
        h, w = image.shape[:2]
        cx, cy, bw, bh = (sample["bboxes"].numpy() * [w, h, w, h]).T
        boxes = np.column_stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2])
        return sample["im_file"], image, boxes, sample["cls"].numpy().reshape(-1).astype(np.int64)


class SharedAugmentationYOLODataset(_SharedAugmentationMixin, YOLODataset):
    pass


class SharedAugmentationRTDETRDataset(_SharedAugmentationMixin, RTDETRDataset):
    pass


def _sync_dataset_epoch(trainer) -> None:
    dataset = trainer.train_loader.dataset
    if dataset.epoch != trainer.epoch:
        dataset.epoch = trainer.epoch
        # Workers are persistent and hold their own dataset copy; restart them so they see the new epoch.
        # This also discards batches prefetched with the old epoch.
        trainer.train_loader.reset()


def _train_dataset(cls, args, data: dict, img_path: str, batch: int, **kwargs: Any):
    return cls(
        img_path=img_path,
        imgsz=args.imgsz,
        batch_size=batch,
        augment=True,
        hyp=args,
        rect=False,
        cache=None,
        single_cls=args.single_cls or False,
        prefix=colorstr("train: "),
        classes=args.classes,
        data=data,
        fraction=args.fraction,
        **kwargs,
    )


def build_train_dataset(data_yaml: str | Path, seed: int, rtdetr: bool = False):
    """The training dataset exactly as the trainers build it, for previews and the same-examples check."""
    args = get_cfg(overrides={"data": str(data_yaml), "seed": seed, **DISABLE_BUILTIN_AUGMENTATION})
    data = check_det_dataset(str(data_yaml))
    if rtdetr:
        return _train_dataset(SharedAugmentationRTDETRDataset, args, data, data["train"], args.batch)
    return _train_dataset(SharedAugmentationYOLODataset, args, data, data["train"], args.batch, stride=32, pad=0.0)


class SharedAugmentationDetectionTrainer(DetectionTrainer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.add_callback("on_train_epoch_start", _sync_dataset_epoch)

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        if mode != "train":
            return super().build_dataset(img_path, mode, batch)
        stride = max(int(unwrap_model(self.model).stride.max()), 32)
        return _train_dataset(SharedAugmentationYOLODataset, self.args, self.data, img_path, batch, stride=stride, pad=0.0)


class SharedAugmentationRTDETRTrainer(RTDETRTrainer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.add_callback("on_train_epoch_start", _sync_dataset_epoch)

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        if mode != "train":
            return super().build_dataset(img_path, mode, batch)
        return _train_dataset(SharedAugmentationRTDETRDataset, self.args, self.data, img_path, batch)
