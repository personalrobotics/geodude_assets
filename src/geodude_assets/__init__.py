# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""MuJoCo models for the Geodude bimanual robot.

This package provides MuJoCo model files for the Geodude robot, including:
- Vention frame base
- Universal Robots UR5e arms
- Robotiq 2F-140 gripper
- Psyonic Ability Hand (left and right)
- Pre-assembled geodude model (two UR5e arms on Vention base)

Example:
    >>> import mujoco
    >>> from geodude_assets import get_model_path
    >>> model = mujoco.MjModel.from_xml_path(str(get_model_path("geodude")))
"""

from pathlib import Path

__version__ = "0.1.0"

# Models directory is inside the package
MODELS_DIR = Path(__file__).parent / "models"

# Available models
AVAILABLE_MODELS = [
    "geodude",  # Full robot: two UR5e arms on Vention base with grippers
    "universal_robots_ur5e",  # Single UR5e arm
    "robotiq_2f140",  # Robotiq 2F-140 gripper
    "vention",  # Vention frame base
    "abh_left_small",  # Psyonic Ability Hand (left, small)
    "abh_right_small",  # Psyonic Ability Hand (right, small)
]


def get_model_path(name: str = "geodude") -> Path:
    """Get the path to a MuJoCo model XML file.

    Args:
        name: Model name. One of: geodude, universal_robots_ur5e, robotiq_2f140,
              vention, abh_left_small, abh_right_small. Defaults to "geodude".

    Returns:
        Path to the model's main XML file.

    Raises:
        FileNotFoundError: If the model doesn't exist.

    Example:
        >>> path = get_model_path("geodude")
        >>> model = mujoco.MjModel.from_xml_path(str(path))
    """
    model_dir = MODELS_DIR / name

    if not model_dir.exists():
        raise FileNotFoundError(f"Model '{name}' not found. Available models: {AVAILABLE_MODELS}")

    # Primary XML file naming conventions
    xml_candidates = [
        model_dir / f"{name}.xml",  # geodude/geodude.xml
        model_dir / "scene.xml",  # some models use scene.xml
        model_dir / "ur5e.xml",  # universal_robots_ur5e/ur5e.xml
        model_dir / "2f140.xml",  # robotiq_2f140/2f140.xml
    ]

    for xml_path in xml_candidates:
        if xml_path.exists():
            return xml_path

    raise FileNotFoundError(f"No XML file found in {model_dir}. Tried: {[p.name for p in xml_candidates]}")


def get_geodude_path() -> Path:
    """Get path to the main geodude model (convenience function).

    Returns:
        Path to geodude/geodude.xml
    """
    return get_model_path("geodude")


def get_component_path(component: str) -> Path:
    """Get path to a component model for custom assembly.

    This is useful when using the assembly module to build custom robot
    configurations.

    Args:
        component: One of "ur5e", "2f140", "vention", "abh_left", "abh_right"

    Returns:
        Path to the component XML file.
    """
    component_map = {
        "ur5e": ("universal_robots_ur5e", "ur5e.xml"),
        "2f140": ("robotiq_2f140", "2f140.xml"),
        "vention": ("vention", "vention.xml"),
        "abh_left": ("abh_left_small", "abh_left_small.xml"),
        "abh_right": ("abh_right_small", "abh_right_small.xml"),
    }

    if component not in component_map:
        raise ValueError(f"Unknown component '{component}'. Available: {list(component_map.keys())}")

    model_name, xml_name = component_map[component]
    return MODELS_DIR / model_name / xml_name
