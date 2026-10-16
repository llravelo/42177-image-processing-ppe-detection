from pathlib import Path

import numpy as np
import torch

from .augmentations import SharedAugmentation
from .core import Example, list_images, load_training_example


class TorchvisionDataset(torch.utils.data.Dataset):
    """Training images through the shared augmentation, in torchvision's detection format.

    Returns a float image in [0, 1] and a target with pixel xyxy `boxes` and `labels` shifted by +1 (0 is background,
    matching `label_offset=1` in model_configs). Images whose boxes are all dropped stay in as background examples.
    Lives in the package rather than a notebook because DataLoader workers must be able to import it.
    """

    def __init__(self, root: str | Path, split: str, augmentation: SharedAugmentation) -> None:
        self.image_paths = list_images(root, split)
        self.augmentation = augmentation
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        # Workers are re-created every epoch (persistent_workers=False), so they pick this up.
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        image, boxes, labels = load_training_example(self.image_paths[index], self.augmentation, self.epoch)
        image = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float() / 255
        return image, {"boxes": torch.from_numpy(boxes).reshape(-1, 4), "labels": torch.from_numpy(labels + 1)}

    def as_raw_example(self, index: int) -> Example:
        """What the model is actually fed, converted back to (path, RGB uint8, xyxy, 0-based labels)."""
        image, target = self[index]
        image = (image * 255).round().byte().permute(1, 2, 0).numpy()
        return str(self.image_paths[index]), image, target["boxes"].numpy(), target["labels"].numpy() - 1


def collate(batch):
    """Detection batches are lists: images and targets differ in size per example."""
    return tuple(zip(*batch))
