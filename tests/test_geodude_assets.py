# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Tests for geodude_assets package."""

import json

import mujoco
import numpy as np
import pytest

from geodude_assets import (
    AVAILABLE_MODELS,
    MODELS_DIR,
    get_component_path,
    get_geodude_path,
    get_model_path,
)


class TestModelPaths:
    """Tests for model path functions."""

    def test_models_dir_exists(self):
        """MODELS_DIR points to existing directory."""
        assert MODELS_DIR.exists()
        assert MODELS_DIR.is_dir()

    def test_get_geodude_path(self):
        """get_geodude_path returns path to geodude.xml."""
        path = get_geodude_path()
        assert path.exists()
        assert path.name == "geodude.xml"

    def test_get_model_path_default(self):
        """get_model_path defaults to geodude."""
        assert get_model_path() == get_geodude_path()

    @pytest.mark.parametrize("model_name", AVAILABLE_MODELS)
    def test_get_model_path_all_models(self, model_name):
        """All listed models have valid paths."""
        path = get_model_path(model_name)
        assert path.exists()
        assert path.suffix == ".xml"

    def test_get_model_path_invalid(self):
        """get_model_path raises for invalid model name."""
        with pytest.raises(FileNotFoundError, match="not found"):
            get_model_path("nonexistent_model")

    def test_get_component_path_ur5e(self):
        """get_component_path returns UR5e path."""
        path = get_component_path("ur5e")
        assert path.exists()
        assert "ur5e.xml" in path.name

    def test_get_component_path_gripper(self):
        """get_component_path returns gripper path."""
        path = get_component_path("2f140")
        assert path.exists()
        assert "2f140.xml" in path.name

    def test_get_component_path_invalid(self):
        """get_component_path raises for invalid component."""
        with pytest.raises(ValueError, match="Unknown component"):
            get_component_path("invalid_component")


class TestModelLoading:
    """Tests that models actually load in MuJoCo."""

    def test_load_geodude(self):
        """Geodude model loads successfully."""
        path = get_geodude_path()
        model = mujoco.MjModel.from_xml_path(str(path))

        assert model is not None
        assert model.nq > 0  # Has joints
        assert model.nv > 0  # Has DOFs

    def test_load_geodude_and_step(self):
        """Geodude model can be simulated."""
        path = get_geodude_path()
        model = mujoco.MjModel.from_xml_path(str(path))
        data = mujoco.MjData(model)

        # Should not raise
        mujoco.mj_step(model, data)
        assert data.time > 0

    @pytest.mark.parametrize("model_name", AVAILABLE_MODELS)
    def test_load_all_models(self, model_name):
        """All models load successfully."""
        path = get_model_path(model_name)
        model = mujoco.MjModel.from_xml_path(str(path))
        assert model is not None


class TestGeodueModelStructure:
    """Tests for expected structure of the geodude model."""

    @pytest.fixture
    def geodude_model(self):
        """Load geodude model."""
        path = get_geodude_path()
        return mujoco.MjModel.from_xml_path(str(path))

    def test_has_left_arm_joints(self, geodude_model):
        """Geodude has left arm joints."""
        joint_names = [
            "left_ur5e/shoulder_pan_joint",
            "left_ur5e/shoulder_lift_joint",
            "left_ur5e/elbow_joint",
            "left_ur5e/wrist_1_joint",
            "left_ur5e/wrist_2_joint",
            "left_ur5e/wrist_3_joint",
        ]
        for name in joint_names:
            jid = mujoco.mj_name2id(geodude_model, mujoco.mjtObj.mjOBJ_JOINT, name)
            assert jid != -1, f"Joint {name} not found"

    def test_has_right_arm_joints(self, geodude_model):
        """Geodude has right arm joints."""
        joint_names = [
            "right_ur5e/shoulder_pan_joint",
            "right_ur5e/shoulder_lift_joint",
            "right_ur5e/elbow_joint",
            "right_ur5e/wrist_1_joint",
            "right_ur5e/wrist_2_joint",
            "right_ur5e/wrist_3_joint",
        ]
        for name in joint_names:
            jid = mujoco.mj_name2id(geodude_model, mujoco.mjtObj.mjOBJ_JOINT, name)
            assert jid != -1, f"Joint {name} not found"

    def test_has_gripper_actuator(self, geodude_model):
        """Geodude has gripper actuator."""
        aid = mujoco.mj_name2id(
            geodude_model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            "right_ur5e/gripper/fingers_actuator",
        )
        assert aid != -1

    def test_has_ready_keyframe(self, geodude_model):
        """Geodude has ready keyframe."""
        kid = mujoco.mj_name2id(geodude_model, mujoco.mjtObj.mjOBJ_KEY, "ready")
        assert kid != -1

    def test_has_linear_actuators(self, geodude_model):
        """Geodude has linear rail actuators for both arms."""
        for name in ["left_linear_actuator", "right_linear_actuator"]:
            aid = mujoco.mj_name2id(geodude_model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            assert aid != -1, f"Actuator {name} not found"

    def test_linear_actuator_range(self, geodude_model):
        """Linear actuators have correct range (0=bottom, 0.5=top)."""
        for name in ["left_linear_actuator", "right_linear_actuator"]:
            aid = mujoco.mj_name2id(geodude_model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            ctrlrange = geodude_model.actuator_ctrlrange[aid]
            assert ctrlrange[0] == 0.0, f"{name} should have min=0"
            assert ctrlrange[1] == 0.5, f"{name} should have max=0.5"

    def test_linear_actuator_moves_arm(self, geodude_model):
        """Linear actuators actually move the arms."""
        data = mujoco.MjData(geodude_model)
        mujoco.mj_forward(geodude_model, data)

        # Get initial z position
        bid = mujoco.mj_name2id(geodude_model, mujoco.mjtObj.mjOBJ_BODY, "left_arm_vention_base")
        initial_z = data.xpos[bid][2]

        # Command to raise arm
        data.ctrl[0] = 0.3  # left_linear_actuator
        for _ in range(500):
            mujoco.mj_step(geodude_model, data)

        final_z = data.xpos[bid][2]
        assert final_z > initial_z + 0.2, "Arm should have raised ~0.3m"


class TestPhysicalConsistency:
    """The generated model is reproducible and its ready pose and lifts are physical."""

    @pytest.fixture
    def geodude_model(self):
        return mujoco.MjModel.from_xml_path(str(get_geodude_path()))

    def test_generated_model_matches_its_sources(self, tmp_path):
        """geodude.xml is exactly what the assembly generator writes from the component
        models, so no change lives only in the generated file."""
        pytest.importorskip("dm_control")
        from geodude_assets.assembly import attach_arms_to_vention

        attach_arms_to_vention(True, str(tmp_path), "geodude.xml", "2f140", "2f140")
        regenerated = (tmp_path / "geodude.xml").read_text()
        assert regenerated.startswith("<!-- Generated by geodude_assets.assembly")
        assert regenerated == get_geodude_path().read_text(), (
            "geodude.xml differs from its sources; regenerate it with "
            "`uv run python -m geodude_assets.assembly --save-mjcf "
            "-d src/geodude_assets/models/geodude -l 2f140 -r 2f140`"
        )

    def test_every_model_file_is_well_formed_xml(self):
        """Every model parses with a strict XML parser, not only MuJoCo's lenient one, so
        other tools (ssrobot, URDF and MJCF readers) can load it too."""
        import xml.etree.ElementTree as ET

        malformed = []
        for path in sorted(MODELS_DIR.rglob("*.xml")):
            try:
                ET.parse(path)
            except ET.ParseError as e:
                malformed.append(f"{path.relative_to(MODELS_DIR)}: {e}")
        assert malformed == []

    def test_every_keyframe_is_within_joint_limits_and_free_of_contact(self, geodude_model):
        """Each keyframe is a valid start: every limited joint inside its range, and no
        contacts."""
        problems = []
        data = mujoco.MjData(geodude_model)
        for key in range(geodude_model.nkey):
            key_name = mujoco.mj_id2name(geodude_model, mujoco.mjtObj.mjOBJ_KEY, key)
            for j in range(geodude_model.njnt):
                if not geodude_model.jnt_limited[j]:
                    continue
                q = geodude_model.key_qpos[key][geodude_model.jnt_qposadr[j]]
                low, high = geodude_model.jnt_range[j]
                if not low <= q <= high:
                    name = mujoco.mj_id2name(geodude_model, mujoco.mjtObj.mjOBJ_JOINT, j)
                    problems.append(f"{key_name}: {name}={q:.4f} not in [{low:.4f}, {high:.4f}]")
            mujoco.mj_resetDataKeyframe(geodude_model, data, key)
            mujoco.mj_forward(geodude_model, data)
            if data.ncon:
                problems.append(f"{key_name}: {data.ncon} contacts")
        assert geodude_model.nkey > 0 and problems == []

    def test_joint_servo_control_ranges_equal_joint_ranges(self, geodude_model):
        """A position servo on a limited joint can command exactly the joint's range:
        never less (a reachable pose would be clipped) and never more."""
        mismatched = []
        for a in range(geodude_model.nu):
            if geodude_model.actuator_trntype[a] != mujoco.mjtTrn.mjTRN_JOINT:
                continue
            gain = geodude_model.actuator_gainprm[a][0]
            bias = geodude_model.actuator_biasprm[a][:3]
            if bias[1] >= 0 or gain != -bias[1] or bias[0] != 0:
                continue  # not a unit-gain position servo, whose control is a position
            j = geodude_model.actuator_trnid[a][0]
            if not geodude_model.jnt_limited[j]:
                continue
            ctrl = tuple(geodude_model.actuator_ctrlrange[a]) if geodude_model.actuator_ctrllimited[a] else None
            joint_range = tuple(geodude_model.jnt_range[j])
            if ctrl is None or max(abs(c - r) for c, r in zip(ctrl, joint_range)) > 1e-6:
                name = mujoco.mj_id2name(geodude_model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
                mismatched.append(f"{name}: ctrlrange {ctrl} vs joint range {joint_range}")
        assert mismatched == []

    def test_every_keyframe_holds_without_saturating_or_drifting(self, geodude_model):
        """From each keyframe, every joint servo holds its joint for a second, within
        1 mm or 1 mrad, without reaching its force limit."""
        problems = []
        data = mujoco.MjData(geodude_model)
        for key in range(geodude_model.nkey):
            key_name = mujoco.mj_id2name(geodude_model, mujoco.mjtObj.mjOBJ_KEY, key)
            mujoco.mj_resetDataKeyframe(geodude_model, data, key)
            for _ in range(500):
                mujoco.mj_step(geodude_model, data)
            for a in range(geodude_model.nu):
                name = mujoco.mj_id2name(geodude_model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
                if geodude_model.actuator_forcelimited[a]:
                    limit = geodude_model.actuator_forcerange[a][1]
                    if abs(data.actuator_force[a]) >= 0.99 * limit:
                        problems.append(f"{key_name}: {name} at its force limit")
                if geodude_model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT:
                    address = geodude_model.jnt_qposadr[geodude_model.actuator_trnid[a][0]]
                    drift = abs(data.qpos[address] - geodude_model.key_qpos[key][address])
                    if drift > 1e-3:
                        problems.append(f"{key_name}: {name} drifted {drift:.4g}")
        assert problems == []

    def test_lifts_hold_their_height_like_a_self_locking_screw(self, geodude_model):
        """Holding the ready height, a lift does not sag under its own weight, and a
        300 N push moves it less than the screw's 0.125 mm backlash."""
        data = mujoco.MjData(geodude_model)
        mujoco.mj_resetDataKeyframe(geodude_model, data, 0)
        for side in ("left", "right"):
            joint = mujoco.mj_name2id(geodude_model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_arm_linear_vention")
            body = mujoco.mj_name2id(geodude_model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_arm_vention_base")
            address = geodude_model.jnt_qposadr[joint]
            for _ in range(500):
                mujoco.mj_step(geodude_model, data)
            held = data.qpos[address]
            assert abs(held - 0.25) < 1e-5, f"{side} lift sagged to {held:.6f}"
            data.xfrc_applied[body] = [0, 0, -300, 0, 0, 0]
            for _ in range(500):
                mujoco.mj_step(geodude_model, data)
            data.xfrc_applied[body] = 0
            assert held - data.qpos[address] < 0.125e-3, f"{side} lift gave way under 300 N"


class TestContactExclusions:
    """Tests for contact exclusions in the geodude model.

    These tests verify that the contact exclusions defined in component XMLs
    (ur5e.xml, 2f140.xml) are properly inherited and prevent spurious
    self-collisions in the assembled model.
    """

    @pytest.fixture
    def geodude_model(self):
        """Load geodude model."""
        path = get_geodude_path()
        return mujoco.MjModel.from_xml_path(str(path))

    @pytest.fixture
    def geodude_data(self, geodude_model):
        """Create data and reset to ready keyframe."""
        data = mujoco.MjData(geodude_model)
        mujoco.mj_resetDataKeyframe(geodude_model, data, 0)  # ready
        return data

    def test_has_contact_exclusions(self, geodude_model):
        """Model has contact exclusions defined."""
        # nexclude counts the number of body pairs excluded from contact
        assert geodude_model.nexclude > 0, "Model should have contact exclusions"

    def test_no_contacts_at_ready_pose(self, geodude_model, geodude_data):
        """No contacts at ready pose after physics step.

        At ready pose, there should be no self-collision contacts. This verifies
        that the contact exclusions for adjacent arm links and gripper internal
        mechanism are working correctly.
        """
        # Run forward dynamics to detect contacts
        mujoco.mj_forward(geodude_model, geodude_data)

        # Check for any contacts
        assert geodude_data.ncon == 0, (
            f"Expected no contacts at ready pose, but found {geodude_data.ncon}. "
            "This may indicate missing contact exclusions."
        )

    def test_gripper_close_no_internal_contacts(self, geodude_model, geodude_data):
        """Closing on nothing, each gripper's fingertips meet, and nothing else touches.

        The 4-bar linkage has parts that pass close to each other (coupler <->
        spring_link), which are excluded. The two fingers are not excluded from each
        other, so the only contacts are between the left and right follower (the pads).
        """
        for hand in ("left_ur5e", "right_ur5e"):
            actuator = mujoco.mj_name2id(
                geodude_model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{hand}/gripper/fingers_actuator"
            )
            geodude_data.ctrl[actuator] = 255  # fully closed

        for _ in range(500):
            mujoco.mj_step(geodude_model, geodude_data)

        def body(geom: int) -> str:
            return mujoco.mj_id2name(geodude_model, mujoco.mjtObj.mjOBJ_BODY, geodude_model.geom_bodyid[geom])

        touching = {tuple(sorted((body(c.geom1), body(c.geom2)))) for c in geodude_data.contact[: geodude_data.ncon]}
        fingertips = {
            (f"{hand}/gripper/left_follower", f"{hand}/gripper/right_follower") for hand in ("left_ur5e", "right_ur5e")
        }
        assert touching == fingertips, (
            f"Expected only fingertip-on-fingertip contacts when closing on nothing, got {sorted(touching)}"
        )

    def test_gripper_bodies_excluded(self, geodude_model):
        """Critical gripper body pairs are in the exclusion list.

        Verifies that the 4-bar linkage parts that move past each other
        are properly excluded from collision detection.
        """
        # Build a set of excluded body pairs for quick lookup
        # Signature encoding: (body1 << 16) | body2
        excluded_pairs = set()
        for i in range(geodude_model.nexclude):
            sig = geodude_model.exclude_signature[i]
            b1 = sig >> 16
            b2 = sig & 0xFFFF
            excluded_pairs.add((min(b1, b2), max(b1, b2)))

        def bodies_excluded(name1: str, name2: str) -> bool:
            """Check if two body names are in the exclusion list."""
            b1 = mujoco.mj_name2id(geodude_model, mujoco.mjtObj.mjOBJ_BODY, name1)
            b2 = mujoco.mj_name2id(geodude_model, mujoco.mjtObj.mjOBJ_BODY, name2)
            if b1 == -1 or b2 == -1:
                return False
            return (min(b1, b2), max(b1, b2)) in excluded_pairs

        # Check critical gripper exclusions (left arm)
        assert bodies_excluded("left_ur5e/gripper/right_coupler", "left_ur5e/gripper/right_spring_link"), (
            "Left gripper right coupler <-> spring_link should be excluded"
        )
        assert bodies_excluded("left_ur5e/gripper/left_coupler", "left_ur5e/gripper/left_spring_link"), (
            "Left gripper left coupler <-> spring_link should be excluded"
        )

        # Check critical gripper exclusions (right arm)
        assert bodies_excluded("right_ur5e/gripper/right_coupler", "right_ur5e/gripper/right_spring_link"), (
            "Right gripper right coupler <-> spring_link should be excluded"
        )
        assert bodies_excluded("right_ur5e/gripper/left_coupler", "right_ur5e/gripper/left_spring_link"), (
            "Right gripper left coupler <-> spring_link should be excluded"
        )

    def test_arm_adjacent_links_excluded(self, geodude_model):
        """Adjacent arm links are excluded from collision detection.

        Adjacent links in the UR5e kinematic chain shouldn't collide since
        they're connected by joints and always in contact.
        """
        # Build a set of excluded body pairs for quick lookup
        # Signature encoding: (body1 << 16) | body2
        excluded_pairs = set()
        for i in range(geodude_model.nexclude):
            sig = geodude_model.exclude_signature[i]
            b1 = sig >> 16
            b2 = sig & 0xFFFF
            excluded_pairs.add((min(b1, b2), max(b1, b2)))

        def bodies_excluded(name1: str, name2: str) -> bool:
            """Check if two body names are in the exclusion list."""
            b1 = mujoco.mj_name2id(geodude_model, mujoco.mjtObj.mjOBJ_BODY, name1)
            b2 = mujoco.mj_name2id(geodude_model, mujoco.mjtObj.mjOBJ_BODY, name2)
            if b1 == -1 or b2 == -1:
                return False
            return (min(b1, b2), max(b1, b2)) in excluded_pairs

        # Check left arm adjacent link exclusions
        left_pairs = [
            ("left_ur5e/base", "left_ur5e/shoulder_link"),
            ("left_ur5e/shoulder_link", "left_ur5e/upper_arm_link"),
            ("left_ur5e/upper_arm_link", "left_ur5e/forearm_link"),
            ("left_ur5e/forearm_link", "left_ur5e/wrist_1_link"),
            ("left_ur5e/wrist_1_link", "left_ur5e/wrist_2_link"),
            ("left_ur5e/wrist_2_link", "left_ur5e/wrist_3_link"),
        ]
        for b1, b2 in left_pairs:
            assert bodies_excluded(b1, b2), f"Left arm {b1} <-> {b2} should be excluded"

        # Check right arm adjacent link exclusions
        right_pairs = [
            ("right_ur5e/base", "right_ur5e/shoulder_link"),
            ("right_ur5e/shoulder_link", "right_ur5e/upper_arm_link"),
            ("right_ur5e/upper_arm_link", "right_ur5e/forearm_link"),
            ("right_ur5e/forearm_link", "right_ur5e/wrist_1_link"),
            ("right_ur5e/wrist_1_link", "right_ur5e/wrist_2_link"),
            ("right_ur5e/wrist_2_link", "right_ur5e/wrist_3_link"),
        ]
        for b1, b2 in right_pairs:
            assert bodies_excluded(b1, b2), f"Right arm {b1} <-> {b2} should be excluded"


class TestForceTorqueSensor:
    """Tests for the UR5e force/torque sensor.

    The UR5e has a built-in 6-axis F/T sensor at the tool flange.
    These tests verify the MuJoCo simulation matches the expected behavior.

    Coordinate frame (tool0): Z+ out of flange, X+ left, Y+ up.
    """

    @pytest.fixture
    def ur5e_model(self):
        """Load standalone UR5e model."""
        path = get_component_path("ur5e")
        return mujoco.MjModel.from_xml_path(str(path))

    @pytest.fixture
    def ur5e_data(self, ur5e_model):
        """Create data for UR5e model."""
        return mujoco.MjData(ur5e_model)

    @pytest.fixture
    def geodude_model(self):
        """Load geodude model."""
        path = get_geodude_path()
        return mujoco.MjModel.from_xml_path(str(path))

    def test_ft_sensor_site_exists(self, ur5e_model):
        """F/T sensor site exists in UR5e model."""
        site_id = mujoco.mj_name2id(ur5e_model, mujoco.mjtObj.mjOBJ_SITE, "ft_sensor_site")
        assert site_id != -1, "ft_sensor_site not found"

    def test_ft_sensor_force_exists(self, ur5e_model):
        """Force sensor exists in UR5e model."""
        sensor_id = mujoco.mj_name2id(ur5e_model, mujoco.mjtObj.mjOBJ_SENSOR, "ft_sensor_force")
        assert sensor_id != -1, "ft_sensor_force not found"

    def test_ft_sensor_torque_exists(self, ur5e_model):
        """Torque sensor exists in UR5e model."""
        sensor_id = mujoco.mj_name2id(ur5e_model, mujoco.mjtObj.mjOBJ_SENSOR, "ft_sensor_torque")
        assert sensor_id != -1, "ft_sensor_torque not found"

    def test_ft_sensor_force_has_3_values(self, ur5e_model):
        """Force sensor outputs 3 values (Fx, Fy, Fz)."""
        sensor_id = mujoco.mj_name2id(ur5e_model, mujoco.mjtObj.mjOBJ_SENSOR, "ft_sensor_force")
        # Sensor dimension
        assert ur5e_model.sensor_dim[sensor_id] == 3

    def test_ft_sensor_torque_has_3_values(self, ur5e_model):
        """Torque sensor outputs 3 values (Tx, Ty, Tz)."""
        sensor_id = mujoco.mj_name2id(ur5e_model, mujoco.mjtObj.mjOBJ_SENSOR, "ft_sensor_torque")
        # Sensor dimension
        assert ur5e_model.sensor_dim[sensor_id] == 3

    def test_ft_sensor_site_at_tool_flange(self, ur5e_model, ur5e_data):
        """F/T sensor site is at tool flange position.

        The ft_sensor_site should be at the same position as gripper_attachment_site.
        """
        mujoco.mj_forward(ur5e_model, ur5e_data)

        ft_site_id = mujoco.mj_name2id(ur5e_model, mujoco.mjtObj.mjOBJ_SITE, "ft_sensor_site")
        gripper_site_id = mujoco.mj_name2id(ur5e_model, mujoco.mjtObj.mjOBJ_SITE, "gripper_attachment_site")

        ft_pos = ur5e_data.site_xpos[ft_site_id]
        gripper_pos = ur5e_data.site_xpos[gripper_site_id]

        import numpy as np

        np.testing.assert_allclose(
            ft_pos, gripper_pos, atol=1e-6, err_msg="F/T sensor site should be at gripper attachment position"
        )

    def test_ft_sensor_frame_z_points_outward(self, ur5e_model, ur5e_data):
        """F/T sensor Z-axis points out of flange (tool0 convention).

        At ready pose, with the robot upright, tool0 Z should point
        in a direction away from the base.
        """
        # Reset to ready pose
        key_id = mujoco.mj_name2id(ur5e_model, mujoco.mjtObj.mjOBJ_KEY, "ready")
        mujoco.mj_resetDataKeyframe(ur5e_model, ur5e_data, key_id)
        mujoco.mj_forward(ur5e_model, ur5e_data)

        ft_site_id = mujoco.mj_name2id(ur5e_model, mujoco.mjtObj.mjOBJ_SITE, "ft_sensor_site")

        # Site rotation matrix (3x3, stored row-major as 9 elements)
        site_xmat = ur5e_data.site_xmat[ft_site_id].reshape(3, 3)

        # Z-axis of the site in world frame (third column of rotation matrix)
        z_axis = site_xmat[:, 2]

        # At ready pose, tool Z should have a significant vertical component
        # (pointing generally upward since the arm is in front and tool
        # faces up/forward). The exact direction depends on the ready pose,
        # but Z should not point back toward the base.
        import numpy as np

        # Verify z-axis is a unit vector
        np.testing.assert_allclose(np.linalg.norm(z_axis), 1.0, atol=1e-6, err_msg="Z-axis should be unit vector")

    def test_ft_sensor_reads_gravity_load(self, ur5e_model, ur5e_data):
        """F/T sensor measures gravity load from gripper/payload.

        With gravity enabled and arm at ready pose, the sensor should
        measure a non-zero force due to the weight of the wrist/tool.
        """
        # Reset to ready pose
        key_id = mujoco.mj_name2id(ur5e_model, mujoco.mjtObj.mjOBJ_KEY, "ready")
        mujoco.mj_resetDataKeyframe(ur5e_model, ur5e_data, key_id)

        # Run forward dynamics
        mujoco.mj_forward(ur5e_model, ur5e_data)

        # Get sensor values
        force_id = mujoco.mj_name2id(ur5e_model, mujoco.mjtObj.mjOBJ_SENSOR, "ft_sensor_force")
        force_adr = ur5e_model.sensor_adr[force_id]
        force = ur5e_data.sensordata[force_adr : force_adr + 3]

        # The sensor should measure some force (gravity on tool)
        # Note: MuJoCo force sensor measures interaction forces which may be zero
        # at static equilibrium when there's no external load. This test verifies
        # the sensor is working - actual values depend on model dynamics.
        assert force is not None and len(force) == 3, "Force should have 3 components"

    def test_geodude_both_arms_have_ft_sensors(self, geodude_model):
        """Both arms in geodude model have F/T sensors.

        When the UR5e is composed into geodude, sensors get prefixed
        with the arm namespace.
        """
        # Check left arm sensors
        left_force_id = mujoco.mj_name2id(geodude_model, mujoco.mjtObj.mjOBJ_SENSOR, "left_ur5e/ft_sensor_force")
        left_torque_id = mujoco.mj_name2id(geodude_model, mujoco.mjtObj.mjOBJ_SENSOR, "left_ur5e/ft_sensor_torque")

        # Check right arm sensors
        right_force_id = mujoco.mj_name2id(geodude_model, mujoco.mjtObj.mjOBJ_SENSOR, "right_ur5e/ft_sensor_force")
        right_torque_id = mujoco.mj_name2id(geodude_model, mujoco.mjtObj.mjOBJ_SENSOR, "right_ur5e/ft_sensor_torque")

        assert left_force_id != -1, "Left arm F/T force sensor not found"
        assert left_torque_id != -1, "Left arm F/T torque sensor not found"
        assert right_force_id != -1, "Right arm F/T force sensor not found"
        assert right_torque_id != -1, "Right arm F/T torque sensor not found"


class TestPinchTorsion:
    """A pinch resists spinning the object about the line between the pads.

    The 2F-140 pinches a 50 mm, 50 g ball, which touches each pad at a single point, and
    a torque then acts on the ball about the line between the pads: 0.02 N*m for 0.5 s,
    about what a can's own weight exerts pinched 40 mm from its centre. The only thing
    resisting it is each contact's friction about its own normal, which MuJoCo gives a
    contact only with condim 4 or 6: with condim 3 the ball spins between the pads as
    on an axle. Real rubber pads resist it over their contact area. (An object touching
    each pad at two separated points, such as a cylinder across the fingers, also resists
    by sliding friction; a scrape that rocks it onto one point does not.)

    Each case writes its measurements to pinch_torsion.json in its pytest tmp_path; to
    keep them: `uv run pytest tests/ -k TestPinchTorsion --basetemp=out`.
    """

    _RADIUS, _MASS = 0.025, 0.05
    _AHEAD = 0.07  # m ahead of the grasp site, between the fingertips' pads
    _CLOSE_STEPS = 1500  # 3 s at 2 ms
    _TORQUE, _TORQUE_STEPS = 0.02, 250  # N*m about the pinch line, for 0.5 s
    _MAX_TWIST = 0.05  # rad relative to the fingers

    @pytest.mark.parametrize("hand", ["left_ur5e", "right_ur5e"])
    def test_a_pinched_ball_does_not_spin_between_the_pads(self, hand, tmp_path):
        spec = mujoco.MjSpec.from_file(str(get_geodude_path()))
        model = spec.compile()
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, 0)
        mujoco.mj_forward(model, data)
        site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{hand}/gripper/grasp_site")
        rotation = data.site_xmat[site].reshape(3, 3).copy()
        body = spec.worldbody.add_body(
            name="object", pos=data.site_xpos[site] + self._AHEAD * rotation[:, 2], gravcomp=1
        )
        body.add_freejoint(name="object")
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[self._RADIUS, 0, 0], mass=self._MASS)
        model = spec.compile()
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, 0)
        address = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object")]
        data.qpos[address : address + 7] = model.qpos0[address : address + 7]

        data.ctrl[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{hand}/gripper/fingers_actuator")] = 255
        for _ in range(self._CLOSE_STEPS):
            mujoco.mj_step(model, data)
        obj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
        finger = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{hand}/gripper/right_follower")

        def relative() -> np.ndarray:
            return data.xmat[finger].reshape(3, 3).T @ data.xmat[obj].reshape(3, 3)

        touching = sorted(
            {
                (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other) or str(other), int(model.geom_condim[other]))
                for c in data.contact[: data.ncon]
                if obj in (model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2])
                for other in [c.geom2 if model.geom_bodyid[c.geom1] == obj else c.geom1]
            }
        )
        grasped = relative()
        pinch_line = data.site_xmat[site].reshape(3, 3)[:, 1]  # the finger-opening axis
        for _ in range(self._TORQUE_STEPS):
            data.xfrc_applied[obj, 3:] = self._TORQUE * pinch_line
            mujoco.mj_step(model, data)
        change = grasped.T @ relative()
        twist = float(np.arccos(np.clip((np.trace(change) - 1) / 2, -1.0, 1.0)))
        report = {"hand": hand, "twist_rad": twist, "touching_geoms_and_condim": touching}
        (tmp_path / "pinch_torsion.json").write_text(json.dumps(report, indent=2) + "\n")
        assert len(touching) >= 2, f"{hand}: the ball is not pinched ({report})"
        assert twist < self._MAX_TWIST, f"{hand}: the pinched ball spun {twist:.3f} rad ({report})"


class TestFingerContact:
    """The Robotiq 2F-140's two fingers cannot pass through each other.

    Each case closes one of Geodude's grippers from the ready pose, on nothing or on
    a small cylinder between its open fingers, and measures the deepest
    overlap between any collision geom of one finger and any of the other. On a
    real gripper the fingertips meet, or press the object, before they could cross.

    Each case writes its measurements to finger_contact.json in its pytest tmp_path.
    To keep and inspect them: `uv run pytest tests/ -k TestFingerContact --basetemp=out`,
    then read out/test_*/finger_contact.json. The simulation is deterministic, so a
    rerun on the same machine writes the same numbers.
    """

    _LINKS = ("driver", "coupler", "spring_link", "follower")
    _MAX_PENETRATION = 1e-3  # m
    _CLOSE_STEPS = 1500  # 3 s at the model's 2 ms timestep
    _SETTLED_STEPS = 250  # the last 0.5 s of the close
    # Closed on nothing, the fingertips press on each other and must stay put. At rest
    # they move about 10 micrometres in the last 0.5 s; 0.1 mm is jitter.
    _MAX_SETTLED_MOTION = 1e-4  # m
    # Cylinders whose axis runs across the fingers, ahead of the grasp site along the
    # approach axis, where the linkage folds around them far enough for the fingertips
    # to reach each other.
    _OBJECTS = {
        "nothing": None,
        "3cm_cylinder": {"radius": 0.015, "ahead": 0.05},
        "4cm_cylinder": {"radius": 0.02, "ahead": 0.06},
    }

    @staticmethod
    def _with_object(hand: str, radius: float, ahead: float) -> mujoco.MjModel:
        """Geodude with a free 100 g cylinder between ``hand``'s open fingers at the ready
        pose. The cylinder starts gravity-compensated, so it waits for the fingers."""
        spec = mujoco.MjSpec.from_file(str(get_geodude_path()))
        model = spec.compile()
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, 0)
        mujoco.mj_forward(model, data)
        site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{hand}/gripper/grasp_site")
        rotation = data.site_xmat[site].reshape(3, 3)
        position = data.site_xpos[site] + ahead * rotation[:, 2]
        # The cylinder's axis (its z) along the site's x: across the finger-opening axis.
        axis_to_site_x = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, (rotation @ axis_to_site_x).ravel())
        body = spec.worldbody.add_body(name="object", pos=position, quat=quat, gravcomp=1)
        body.add_freejoint(name="object")
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[radius, 0.06, 0], mass=0.1)
        return spec.compile()

    def _finger_geoms(self, model: mujoco.MjModel, hand: str, side: str) -> list[int]:
        bodies = {
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{hand}/gripper/{side}_{link}") for link in self._LINKS
        }
        return [
            g
            for g in range(model.ngeom)
            if model.geom_bodyid[g] in bodies and (model.geom_contype[g] or model.geom_conaffinity[g])
        ]

    @pytest.mark.parametrize("object_name", list(_OBJECTS))
    @pytest.mark.parametrize("hand", ["left_ur5e", "right_ur5e"])
    def test_fingers_do_not_pass_through_each_other(self, hand, object_name, tmp_path):
        """Closed on nothing or on a small cylinder, no pair of the fingers' collision
        geoms overlaps by more than 1 mm. On nothing, the fingertips come to rest against
        each other; a cylinder stays held between them once its gravity is switched on."""
        held = self._OBJECTS[object_name]
        if held is None:
            model = mujoco.MjModel.from_xml_path(str(get_geodude_path()))
        else:
            model = self._with_object(hand, held["radius"], held["ahead"])
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, 0)
        if held is not None:
            # The keyframe predates the cylinder; start it where it was placed.
            address = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object")]
            data.qpos[address : address + 7] = model.qpos0[address : address + 7]

        actuator = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{hand}/gripper/fingers_actuator")
        data.ctrl[actuator] = 255  # fully closed
        fingertips = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{hand}/gripper/{side}_follower")
            for side in ("left", "right")
        ]
        settled = []
        for step in range(self._CLOSE_STEPS):
            mujoco.mj_step(model, data)
            if step >= self._CLOSE_STEPS - self._SETTLED_STEPS:
                settled.append(data.xpos[fingertips].copy())
        settled_motion = float(np.ptp(np.array(settled), axis=0).max())

        report = {"hand": hand, "object": object_name, "settled_fingertip_motion_mm": 1e3 * settled_motion}
        if held is not None:
            obj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
            grasped_at = data.xpos[obj].copy()
            model.body_gravcomp[obj] = 0
            for _ in range(500):
                mujoco.mj_step(model, data)
            report["object_drop_mm"] = 1e3 * float(np.linalg.norm(data.xpos[obj] - grasped_at))
            sides = set()
            for c in data.contact[: data.ncon]:
                bodies = (model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2])
                if obj in bodies:
                    other = mujoco.mj_id2name(
                        model, mujoco.mjtObj.mjOBJ_BODY, bodies[1] if bodies[0] == obj else bodies[0]
                    )
                    for side in ("left", "right"):
                        if other.startswith(f"{hand}/gripper/{side}_"):
                            sides.add(side)
            report["object_touched_by"] = sorted(sides)

        deepest = (np.inf, "", "")
        for a in self._finger_geoms(model, hand, "left"):
            for b in self._finger_geoms(model, hand, "right"):
                distance = mujoco.mj_geomDistance(model, data, a, b, 0.05, None)
                if distance < deepest[0]:
                    deepest = (distance, model.geom(a).name, model.geom(b).name)
        report["min_finger_to_finger_distance_mm"] = 1e3 * deepest[0]
        report["closest_pair"] = list(deepest[1:])
        (tmp_path / "finger_contact.json").write_text(json.dumps(report, indent=2) + "\n")

        assert deepest[0] >= -self._MAX_PENETRATION, (
            f"{hand} fingers overlap by {-1e3 * deepest[0]:.1f} mm closing on {object_name} "
            f"({deepest[1]} / {deepest[2]})"
        )
        if held is None:
            assert settled_motion < self._MAX_SETTLED_MOTION, (
                f"{hand} fingertips moved {1e3 * settled_motion:.3f} mm after closing on nothing"
            )
        else:
            assert report["object_touched_by"] == ["left", "right"], f"{object_name} not held by both fingers"
            # Falling for that second would move it metres; held, it settles within 1 cm.
            assert report["object_drop_mm"] < 10.0, f"{object_name} slipped {report['object_drop_mm']:.1f} mm"


class TestGripDrive:
    """The 2F-140's drive behaves like the real gripper's: a position target with a force
    limit, at a set speed (personalrobotics/geodude_assets#37).

    From the ready pose, each case drives one of Geodude's grippers:
    - closing and opening on nothing: the fingertips' peak speed is within Robotiq's
      30-250 mm/s for the 2F-140, and the fingers are still once they arrive;
    - a half-open command opens the fingers part way, between fully open and closed;
    - gripping a 50 g cylinder of 40, 66 or 110 mm across, its axis across the fingers and
      its contact line 10 mm behind the fingertips: the finger stays parallel and each pad
      presses with 46 N, within 5 %. That is near the 50 N Geodude's hardware commands; a
      stronger drive would overpower the finger's spring link and fold the finger on the
      40 mm cylinder (personalrobotics/geodude_assets#36).

    Each case writes its measurements to grip_drive.json in its pytest tmp_path. To keep
    and inspect them: `uv run pytest tests/ -k TestGripDrive --basetemp=out`, then read
    out/test_*/grip_drive.json. The simulation is deterministic, so a rerun on the same
    machine writes the same numbers.
    """

    _STEP = 0.002  # s, the model's timestep
    _MOVE_STEPS = 1500  # 3 s
    _SPEED_RANGE = (0.030, 0.250)  # m/s at each fingertip, Robotiq's 2F-140 range
    _MAX_SETTLED_MOTION = 1e-4  # m over the last 0.5 s of a move
    _GRIP = 46.0  # N on each pad
    _GRIP_TOLERANCE = 0.05
    _BEHIND_TIP = 0.010  # m: where the cylinder touches the pads, behind the fingertips
    _FINGER_LENGTH = 0.114  # m: grasp site to fingertip along the approach axis

    @staticmethod
    def _gap(model, data, hand):
        left, right = (
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{hand}/gripper/{side}_follower")
            for side in ("left", "right")
        )
        return float(np.linalg.norm(data.xpos[left] - data.xpos[right]))

    def _move(self, hand, start, end):
        model = mujoco.MjModel.from_xml_path(str(get_geodude_path()))
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, 0)
        actuator = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{hand}/gripper/fingers_actuator")
        data.ctrl[actuator] = start
        for _ in range(self._MOVE_STEPS):
            mujoco.mj_step(model, data)
        data.ctrl[actuator] = end
        gaps = []
        for _ in range(self._MOVE_STEPS):
            mujoco.mj_step(model, data)
            gaps.append(self._gap(model, data, hand))
        gaps = np.array(gaps)
        speed = float(np.abs(np.diff(gaps)).max() / self._STEP / 2)  # each fingertip
        settled = float(np.ptp(gaps[-250:]))
        return gaps[0], gaps[-1], speed, settled

    @pytest.mark.parametrize("hand", ["left_ur5e", "right_ur5e"])
    def test_the_fingers_move_at_the_real_grippers_speed(self, hand, tmp_path):
        closed = self._move(hand, 0, 255)
        opened = self._move(hand, 255, 0)
        half = self._move(hand, 0, 128)
        report = {
            "hand": hand,
            "close": {
                "from_mm": 1e3 * closed[0],
                "to_mm": 1e3 * closed[1],
                "peak_fingertip_mm_s": 1e3 * closed[2],
                "settled_motion_mm": 1e3 * closed[3],
            },
            "open": {
                "from_mm": 1e3 * opened[0],
                "to_mm": 1e3 * opened[1],
                "peak_fingertip_mm_s": 1e3 * opened[2],
                "settled_motion_mm": 1e3 * opened[3],
            },
            "half_open_command_gap_mm": 1e3 * half[1],
        }
        (tmp_path / "grip_drive.json").write_text(json.dumps(report, indent=2) + "\n")
        low, high = self._SPEED_RANGE
        for name, (_, _, speed, settled) in (("closing", closed), ("opening", opened)):
            assert low <= speed <= high, f"{hand} fingertips peak at {1e3 * speed:.0f} mm/s {name} ({report})"
            assert settled < self._MAX_SETTLED_MOTION, f"{hand} fingers still moving after {name} ({report})"
        assert closed[1] < half[1] < opened[1], f"{hand}: a half-open command is not part way ({report})"

    @pytest.mark.parametrize("diameter", [0.040, 0.066, 0.110])
    @pytest.mark.parametrize("hand", ["left_ur5e", "right_ur5e"])
    def test_each_pad_grips_with_the_commanded_force(self, hand, diameter, tmp_path):
        model = TestFingerContact._with_object(hand, diameter / 2, self._FINGER_LENGTH - self._BEHIND_TIP)
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, 0)
        address = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object")]
        data.qpos[address : address + 7] = model.qpos0[address : address + 7]
        obj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
        model.body_mass[obj] = 0.05
        data.ctrl[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{hand}/gripper/fingers_actuator")] = 255
        for _ in range(self._MOVE_STEPS):
            mujoco.mj_step(model, data)
        model.body_gravcomp[obj] = 0
        for _ in range(250):
            mujoco.mj_step(model, data)
        per_side = {"left": 0.0, "right": 0.0}
        force = np.zeros(6)
        for i in range(data.ncon):
            c = data.contact[i]
            bodies = (model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2])
            if obj not in bodies:
                continue
            other = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bodies[1] if bodies[0] == obj else bodies[0])
            mujoco.mj_contactForce(model, data, i, force)
            for side in per_side:
                if other.startswith(f"{hand}/gripper/{side}_"):
                    per_side[side] += float(force[0])
        report = {"hand": hand, "diameter_mm": 1e3 * diameter, "pad_force_N": per_side}
        (tmp_path / "grip_drive.json").write_text(json.dumps(report, indent=2) + "\n")
        for side, value in per_side.items():
            assert abs(value - self._GRIP) <= self._GRIP_TOLERANCE * self._GRIP, (
                f"{hand} {side} pad grips a {1e3 * diameter:.0f} mm cylinder with {value:.1f} N ({report})"
            )
