"""Robot assembly utilities for building custom Geodude configurations.

This module uses dm_control.mjcf to programmatically combine component models
(UR5e arms, grippers, Vention base) into complete robot assemblies.

Requires the 'assembly' optional dependency:
    pip install geodude_assets[assembly]
"""

import argparse
from pathlib import Path

import mujoco
import numpy as np
from dm_control import mjcf

from geodude_assets import MODELS_DIR

_UR5_XML = MODELS_DIR / "universal_robots_ur5e" / "ur5e.xml"
_2F140_XML = MODELS_DIR / "robotiq_2f140" / "2f140.xml"
_ABHL_XML = MODELS_DIR / "abh_left_small" / "abh_left_small.xml"
_ABHR_XML = MODELS_DIR / "abh_right_small" / "abh_right_small.xml"
_VENTION_XML = MODELS_DIR / "vention" / "vention.xml"

_LEFT_ARM_ATTACHMENT_SITE_NAME = "left_arm_attachment_site"
_RIGHT_ARM_ATTACHMENT_SITE_NAME = "right_arm_attachment_site"
_GRIPPER_ATTACHMENT_SITE_NAME = "gripper_attachment_site"


def load_ur5e_arm(
    prefix: str, gripper_type: str | None
) -> tuple[mjcf.RootElement, np.ndarray, np.ndarray]:
    """Load a ur5e arm with gripper with a named prefix."""
    ur5e_model = mjcf.from_path(_UR5_XML.as_posix())
    ur5e_model.model = prefix
    for i, g in enumerate(ur5e_model.worldbody.find_all("geom")):
        g.name = f"geom_{i}"
    if gripper_type is None:
        return ur5e_model, None, None
    else:
        return attach_gripper(prefix, gripper_type, ur5e_model)


def attach_gripper(
    prefix: str, gripper_type: str, ur5e_model: mjcf.RootElement
) -> tuple[mjcf.RootElement, np.ndarray, np.ndarray]:
    """Load a gripper and attach it to the ur5e arm."""
    gripper_attachment_site = ur5e_model.find("site", _GRIPPER_ATTACHMENT_SITE_NAME)
    if gripper_attachment_site is None:
        raise ValueError(
            f"Unable to find a site named {_GRIPPER_ATTACHMENT_SITE_NAME} "
            f"in {_UR5_XML.as_posix()}"
        )
    gripper, qpos, ctrl = load_gripper("gripper", gripper_type)
    gripper_attachment_site.attach(gripper)
    return ur5e_model, qpos, ctrl


def load_gripper(
    prefix: str, gripper_type: str
) -> tuple[mjcf.RootElement, np.ndarray, np.ndarray]:
    """Load gripper model and return its qpos and ctrl."""
    if gripper_type == "2f140":
        gripper_model = mjcf.from_path(_2F140_XML.as_posix())
    elif gripper_type == "abhl":
        gripper_model = mjcf.from_path(_ABHL_XML.as_posix())
    elif gripper_type == "abhr":
        gripper_model = mjcf.from_path(_ABHR_XML.as_posix())
    else:
        raise ValueError(
            f"Only gripper types '2f140', 'abhl','abhr' are supported at this time,"
            f"got {gripper_type}"
        )
    gripper_model.model = prefix
    gripper_key = gripper_model.find("key", "home")
    gripper_qpos = gripper_key.qpos
    gripper_ctrl = gripper_key.ctrl
    gripper_key.remove()
    for i, g in enumerate(gripper_model.worldbody.find_all("geom")):
        g.name = f"geom_{i}"
    return gripper_model, gripper_qpos, gripper_ctrl


def attach_arms_to_vention(
    save_file: bool,
    dir: str,
    filename: str,
    left_gripper_type: str | None,
    right_gripper_type: str | None = None,
) -> mujoco.MjModel:
    """Create a MuJoCo model with ur5es attached to each vention rail.

    Args:
        save_file: Whether or not the MuJoCo model should be saved to a file.
        dir: Directory for the saved MuJoCo model file.
        filename: Name of the saved MuJoCo model file.
        left_gripper_type: Type of gripper to use for left arm.
                           Select from: 2f140, none.
        right_gripper_type: Type of gripper to use for right arm.
                            Select from: 2f140, none.

    Returns:
        A MuJoCo model of two ur5es attached to the vention frame.
    """
    geodude_model = mjcf.from_path(_VENTION_XML.as_posix())
    for i, g in enumerate(geodude_model.worldbody.find_all("geom")):
        g.name = f"geom_{i}"

    left_arm_attachment_site = geodude_model.find(
        "site", _LEFT_ARM_ATTACHMENT_SITE_NAME
    )
    if left_arm_attachment_site is None:
        raise ValueError(
            f"Unable to find a site named {_LEFT_ARM_ATTACHMENT_SITE_NAME} "
            f"in {_VENTION_XML.as_posix()}"
        )
    left_ur5e, left_gripper_qpos, left_gripper_ctrl = load_ur5e_arm(
        "left_ur5e", left_gripper_type
    )
    left_arm_attachment_site.attach(left_ur5e)

    right_arm_attachment_site = geodude_model.find(
        "site", _RIGHT_ARM_ATTACHMENT_SITE_NAME
    )
    if right_arm_attachment_site is None:
        raise ValueError(
            f"Unable to find a site named {_RIGHT_ARM_ATTACHMENT_SITE_NAME} "
            f"in {_VENTION_XML.as_posix()}"
        )
    right_ur5e, right_gripper_qpos, right_gripper_ctrl = load_ur5e_arm(
        "right_ur5e", right_gripper_type
    )
    right_arm_attachment_site.attach(right_ur5e)

    # The ur5e model has a "home" keyframe defined.
    # Remove the keyframes for each arm and redefine a bimanual "home" keyframe.
    keys_to_search = [
        "left_ur5e/home",
        "right_ur5e/home",
    ]

    gripper_qposes = [
        left_gripper_qpos,
        right_gripper_qpos,
    ]

    gripper_ctrls = [
        left_gripper_ctrl,
        right_gripper_ctrl,
    ]

    gripper_types = [
        left_gripper_type,
        right_gripper_type,
    ]

    res_qpos = np.array([])
    res_ctrl = np.array([])
    for i, key_name in enumerate(keys_to_search):
        key = geodude_model.find("key", key_name)
        if key is not None:
            res_qpos = np.concatenate([res_qpos, key.qpos])
            res_ctrl = np.concatenate([res_ctrl, key.ctrl])
            key.remove()
        else:
            print(f"Keyframe {key_name} not found.")

        # add gripper qpos and ctrl to the result
        if gripper_types[i] is not None:
            res_qpos = np.concatenate([res_qpos, gripper_qposes[i]])
            res_ctrl = np.concatenate([res_ctrl, gripper_ctrls[i]])

    geodude_model.keyframe.add(
        "key",
        name="home",
        qpos=res_qpos,
        ctrl=res_ctrl,
    )

    if save_file:
        mjcf.export_with_assets(
            mjcf_model=geodude_model, out_dir=dir, out_file_name=filename
        )
    return mujoco.MjModel.from_xml_string(
        geodude_model.to_xml_string(), geodude_model.get_assets()
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a MuJoCo model of two UR5e arms on a vention base."
    )
    parser.add_argument(
        "-s",
        "--save-mjcf",
        action="store_true",
        default=False,
        help="Save MJCF to a file.",
    )
    parser.add_argument(
        "-d",
        "--directory",
        type=Path,
        default="geodude",
        help="Directory for the saved MJCF.",
    )
    parser.add_argument(
        "-f",
        "--filename",
        type=str,
        default="geodude.xml",
        help="Name of the MJCF file.",
    )
    parser.add_argument(
        "-l",
        "--left-gripper-type",
        type=str,
        choices=["2f140", "abhl"],
        default=None,
        help="Type of gripper to use for left arm. Leave out flag for no gripper.",
    )
    parser.add_argument(
        "-r",
        "--right-gripper-type",
        type=str,
        choices=["2f140", "abhr"],
        default=None,
        help="Type of gripper to use for right arm. Leave out flag for no gripper.",
    )
    args = parser.parse_args()
    if args.left_gripper_type is not None:
        args.left_gripper_type = args.left_gripper_type.lower()
    if args.right_gripper_type is not None:
        args.right_gripper_type = args.right_gripper_type.lower()
    attach_arms_to_vention(
        args.save_mjcf,
        args.directory,
        args.filename,
        args.left_gripper_type,
        args.right_gripper_type,
    )
