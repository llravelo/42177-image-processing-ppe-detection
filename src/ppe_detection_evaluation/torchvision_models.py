import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, ssd300_vgg16
from torchvision.models.detection._utils import retrieve_out_channels
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.ssd import SSDClassificationHead

from .augment import SharedAugmentation
from .classes import DEFAULT_CLASS_NAMES
from .data import list_images, load_training_example, xyxy_to_cxcywh
from .framework import EvaluationFramework
from .model_configs import ModelConfig

NUM_CLASSES = len(DEFAULT_CLASS_NAMES) + 1  # + background at index 0


class PPEDataset(Dataset):
    """Images from one split, passed through the shared augmentation for the current epoch.

    Targets use torchvision's convention: pixel xyxy boxes and labels shifted by +1 (0 is background),
    matching `label_offset=1` in model_configs. Images whose boxes are all dropped by the augmentation
    stay in as background examples.
    """

    def __init__(self, root: str | Path, split: str, augmentation: SharedAugmentation) -> None:
        self.image_paths = list_images(root, split)
        self.augmentation = augmentation
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        # DataLoader workers are re-created every epoch (persistent_workers=False), so they pick this up.
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.image_paths)

    def raw_example(self, index: int) -> tuple[str, np.ndarray, np.ndarray, np.ndarray]:
        path = self.image_paths[index]
        return (str(path), *load_training_example(path, self.augmentation, self.epoch))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        _, image, boxes, labels = self.raw_example(index)
        image = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float() / 255
        target = {
            "boxes": torch.from_numpy(boxes).float().reshape(-1, 4),
            "labels": torch.from_numpy(labels + 1).long(),
        }
        return image, target


def as_raw_example(dataset: PPEDataset, index: int) -> tuple[str, np.ndarray, np.ndarray, np.ndarray]:
    """What the model is actually fed, converted back to (path, RGB uint8, xyxy, 0-based labels)."""
    image, target = dataset[index]
    image = (image * 255).round().byte().permute(1, 2, 0).numpy()
    return str(dataset.image_paths[index]), image, target["boxes"].numpy(), target["labels"].numpy() - 1


def collate(batch):
    return tuple(zip(*batch))


def make_train_loader(dataset: PPEDataset, batch_size: int, seed: int, num_workers: int = 4) -> DataLoader:
    # No worker_init_fn needed: augmentation randomness comes from the per-sample seed, not worker RNG state.
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate,
        generator=torch.Generator().manual_seed(seed),
    )


def build_faster_rcnn(weights: str | None = "COCO_V1") -> torch.nn.Module:
    # min/max_size=640 keeps the native 640×640 input instead of upsampling to 800, the same input
    # resolution the Ultralytics models get.
    model = fasterrcnn_resnet50_fpn_v2(
        weights=weights,
        weights_backbone=None,
        min_size=640,
        max_size=640,
        box_score_thresh=0.001,
        box_detections_per_img=100,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    return model


def build_ssd300(weights: str | None = "COCO_V1") -> torch.nn.Module:
    model = ssd300_vgg16(weights=weights, weights_backbone=None, score_thresh=0.001, detections_per_img=100)
    in_channels = retrieve_out_channels(model.backbone, (300, 300))
    num_anchors = model.anchor_generator.num_anchors_per_location()
    model.head.classification_head = SSDClassificationHead(in_channels, num_anchors, NUM_CLASSES)
    return model


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    warmup_iters: int = 0,
    log_every: int = 50,
) -> float:
    model.train()
    loader.dataset.set_epoch(epoch)
    base_lrs = [g["lr"] for g in optimizer.param_groups]
    total, count = 0.0, 0
    for step, (images, targets) in enumerate(loader):
        it = epoch * len(loader) + step
        if it < warmup_iters:
            for group, lr in zip(optimizer.param_groups, base_lrs):
                group["lr"] = lr * (it + 1) / warmup_iters
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss = sum(model(images, targets).values())
        if not math.isfinite(loss.item()):
            raise FloatingPointError(f"Loss is {loss.item()} at epoch {epoch}, step {step}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
        count += 1
        if log_every and step % log_every == 0:
            print(f"epoch {epoch} step {step}/{len(loader)} loss {loss.item():.4f}")
    for group, lr in zip(optimizer.param_groups, base_lrs):
        group["lr"] = lr
    return total / max(count, 1)


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    root: str | Path,
    split: str,
    config: ModelConfig,
    device: torch.device,
) -> EvaluationFramework:
    model.eval()
    eval_aug = SharedAugmentation(train=False)
    fw = EvaluationFramework(config)
    for path in list_images(root, split):
        image, gt_boxes, gt_labels = load_training_example(path, eval_aug, epoch=0)
        tensor = torch.from_numpy(image).permute(2, 0, 1).float().div(255).to(device)
        _sync(device)
        start = time.perf_counter()
        out = model([tensor])[0]
        _sync(device)
        latency_ms = (time.perf_counter() - start) * 1000
        h, w = image.shape[:2]
        fw.add(
            path.stem,
            out["boxes"].cpu().numpy(),
            out["scores"].cpu().numpy(),
            out["labels"].cpu().numpy(),
            xyxy_to_cxcywh(gt_boxes, w, h),
            gt_labels,
            latency_ms=latency_ms,
        )
    return fw


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()
