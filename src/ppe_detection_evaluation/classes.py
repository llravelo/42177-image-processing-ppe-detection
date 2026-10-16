# The subset of Ultralytics Construction-PPE this project detects, in our own id order (see KEPT_SOURCE_CLASSES in
# dataset/core.py for how source ids map onto these).
DEFAULT_CLASS_NAMES: tuple[str, ...] = (
    "person",
    "vest",
    "no_vest",
    "helmet",
    "no_helmet",
)

PPE_PAIRS: dict[str, str] = {
    "helmet": "no_helmet",
    "vest": "no_vest",
}

PPE_CLASSES: tuple[str, ...] = (*PPE_PAIRS, *PPE_PAIRS.values())

VIOLATION_CLASSES: tuple[str, ...] = tuple(PPE_PAIRS.values())


def default_class_groups(class_names: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    return {
        "all": tuple(class_names),
        "ppe": PPE_CLASSES,
        "violation": VIOLATION_CLASSES,
    }
