"""Train YOLO26-M / RT-DETR-L on the shared augmentation instead of Ultralytics' own.

The training dataset bypasses Ultralytics' augmentation stack entirely: `__getitem__` loads each image
through `load_training_example`, the same function the torchvision dataset uses, and only Ultralytics'
eval-mode transforms (a no-op 640×640 letterbox and `Format`) run afterwards.
"""

from pathlib import Path
from typing import Any

import numpy as np
import yaml
from ultralytics.cfg import get_cfg
from ultralytics.data.augment import Compose
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

# Zeroed so the saved args.yaml documents that no built-in augmentation ran. The shared dataset ignores
# these anyway, except multi_scale, which the trainer applies to whole batches.
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


class _SharedAugmentationMixin:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.epoch = 0
        self.shared_augmentation = SharedAugmentation(seed=kwargs["hyp"].seed)
        super().__init__(*args, **kwargs)

    def build_transforms(self, hyp) -> Compose:
        # Reuse the eval-mode pipeline (LetterBox, a no-op at 640×640, then Format) so none of
        # Ultralytics' training augmentations are constructed.
        augment, self.augment = self.augment, False
        try:
            return super().build_transforms(hyp)
        finally:
            self.augment = augment

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = self.im_files[index]
        image, boxes, labels = load_training_example(path, self.shared_augmentation, self.epoch, self.label_files[index])
        h, w = image.shape[:2]
        sample = {
            "im_file": path,
            "img": np.ascontiguousarray(image[..., ::-1]),  # Ultralytics works in BGR; Format flips back
            "ori_shape": (h, w),
            "resized_shape": (h, w),
            "ratio_pad": (1.0, 1.0),
            "cls": labels.reshape(-1, 1).astype(np.float32),
            "instances": Instances(
                boxes.astype(np.float32),
                np.zeros((0, 1000, 2), dtype=np.float32),
                bbox_format="xyxy",
                normalized=False,
            ),
        }
        return self.transforms(sample)

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
