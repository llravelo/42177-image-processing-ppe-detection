from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np

from ..classes import DEFAULT_CLASS_NAMES


def plot_examples(
    examples: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
    titles: Sequence[str] | None = None,
    ncols: int = 4,
    class_names: Sequence[str] = DEFAULT_CLASS_NAMES,
    size: float = 3.5,
):
    """Grid of RGB images with pixel xyxy boxes and 0-based labels drawn on top."""
    nrows = -(-len(examples) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(size * ncols, size * nrows), squeeze=False)
    cmap = plt.get_cmap("tab10")
    for ax in axes.flat:
        ax.axis("off")
    for i, (image, boxes, labels) in enumerate(examples):
        ax = axes.flat[i]
        ax.imshow(image)
        for (x1, y1, x2, y2), label in zip(boxes, labels):
            color = cmap(int(label) % 10)
            ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color=color, linewidth=1.2))
            ax.text(x1, y1, class_names[int(label)], color="white", fontsize=6, backgroundcolor=color, va="bottom")
        if titles:
            ax.set_title(titles[i], fontsize=8)
    fig.tight_layout()
    return fig
