"""Settings every model notebook imports, so all four detectors train on the same examples.

The augmented example for an image is a function of (SEED, epoch, image name) only, so a shared SEED means every
model sees exactly the same example for a given image and epoch.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data" / "construction-ppe"
RESULTS_DIR = PROJECT_ROOT / "results"
# All model weights: downloaded COCO weights in pretrained/, and one folder per training run
# (models/<model>/weights/best.pt plus that run's training curves).
MODELS_DIR = PROJECT_ROOT / "models"
PRETRAINED_DIR = MODELS_DIR / "pretrained"
SEED = 0
# Every model trains and is evaluated on IMAGE_SIZE × IMAGE_SIZE inputs.
IMAGE_SIZE = 640
# Upper bound on epochs for training runs; early stopping usually ends a run sooner.
MAX_EPOCHS = 200
# Early stopping and batch size are not shared: each training notebook sets them in its hyperparameter cell.
