from pathlib import Path

import albumentations as A
import numpy as np
import torch

from .core import list_images, load_example


class TorchvisionDataset(torch.utils.data.Dataset):
    """Training images through the shared augmentation, in torchvision's detection format.

    Returns a float image in [0, 1] and a target with pixel xyxy `boxes` and `labels` shifted by +1 (0 is background,
    matching `label_offset=1` in model_configs). Images whose boxes are all dropped stay in as background examples.
    Lives in the package rather than a notebook because DataLoader workers must be able to import it.
    """

    def __init__(self, root: str | Path, split: str, transform: A.Compose) -> None:
        self.image_paths = list_images(root, split)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        image, boxes, labels = load_example(self.image_paths[index])
        out = self.transform(image=image, bboxes=boxes, labels=labels)
        image = torch.from_numpy(np.ascontiguousarray(out["image"])).permute(2, 0, 1).float() / 255
        boxes = torch.as_tensor(np.asarray(out["bboxes"], dtype=np.float32)).reshape(-1, 4)
        labels = torch.as_tensor(np.asarray(out["labels"], dtype=np.int64))
        return image, {"boxes": boxes, "labels": labels + 1}


def seed_worker(worker_id: int) -> None:
    # Forked workers inherit identical Albumentations RNG state and would all draw the same augmentations.
    # torch.initial_seed() differs per worker and follows the DataLoader's generator, so runs stay reproducible.
    dataset = torch.utils.data.get_worker_info().dataset
    dataset.transform.set_random_seed(torch.initial_seed() % 2**32)


def collate(batch):
    """Detection batches are lists: images and targets differ in size per example."""
    return tuple(zip(*batch))
