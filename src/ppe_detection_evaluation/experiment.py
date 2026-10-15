"""Settings every model notebook imports, so all four detectors train on the same examples.

The augmented example for an image is a function of (SEED, epoch, image name) only, so equal SEED and
EPOCHS mean every model sees exactly the same sequence of training examples.
"""

from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "construction-site-safety-v30"
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
SEED = 0
EPOCHS = 50
BATCH_SIZE = 8
