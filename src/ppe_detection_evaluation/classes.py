# Ultralytics Construction-PPE, in its own id order. The source calls class 5 "none"; it marks a torso
# without a safety vest, so it is named "no_vest" here.
DEFAULT_CLASS_NAMES: tuple[str, ...] = (
    "helmet",
    "gloves",
    "vest",
    "boots",
    "goggles",
    "no_vest",
    "Person",
    "no_helmet",
    "no_goggle",
    "no_gloves",
    "no_boots",
)

PPE_PAIRS: dict[str, str] = {
    "helmet": "no_helmet",
    "vest": "no_vest",
    "goggles": "no_goggle",
    "gloves": "no_gloves",
    "boots": "no_boots",
}

PPE_CLASSES: tuple[str, ...] = (*PPE_PAIRS, *PPE_PAIRS.values())

VIOLATION_CLASSES: tuple[str, ...] = tuple(PPE_PAIRS.values())


def default_class_groups(class_names: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    return {
        "all": tuple(class_names),
        "ppe": PPE_CLASSES,
        "violation": VIOLATION_CLASSES,
    }
