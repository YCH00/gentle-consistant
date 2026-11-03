"""Three-component ablations, isolated from historical main/shift experiments."""

PROTOCOL = "component-ablations-v1"
ARMS = {
    "backbone": (0, 0, 0),
    "real_only": (1, 0, 0),
    "no_real": (0, 1, 1),
    "no_virtual_recovery": (1, 0, 1),
    "no_virtual_rl": (1, 1, 0),
    "full": (1, 1, 1),
}
PAIRS = [
    ("full", "backbone"), ("real_only", "backbone"),
    ("full", "no_real"), ("full", "no_virtual_recovery"),
    ("full", "no_virtual_rl"),
]
