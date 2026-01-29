"""Tests for geodude_assets package."""

import mujoco
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

    def test_has_home_keyframe(self, geodude_model):
        """Geodude has home keyframe."""
        kid = mujoco.mj_name2id(geodude_model, mujoco.mjtObj.mjOBJ_KEY, "home")
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
        bid = mujoco.mj_name2id(
            geodude_model, mujoco.mjtObj.mjOBJ_BODY, "left_arm_vention_base"
        )
        initial_z = data.xpos[bid][2]

        # Command to raise arm
        data.ctrl[0] = 0.3  # left_linear_actuator
        for _ in range(500):
            mujoco.mj_step(geodude_model, data)

        final_z = data.xpos[bid][2]
        assert final_z > initial_z + 0.2, "Arm should have raised ~0.3m"


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
        """Create data and reset to home keyframe."""
        data = mujoco.MjData(geodude_model)
        mujoco.mj_resetDataKeyframe(geodude_model, data, 0)  # home
        return data

    def test_has_contact_exclusions(self, geodude_model):
        """Model has contact exclusions defined."""
        # nexclude counts the number of body pairs excluded from contact
        assert geodude_model.nexclude > 0, "Model should have contact exclusions"

    def test_no_contacts_at_home_pose(self, geodude_model, geodude_data):
        """No contacts at home pose after physics step.

        At home pose, there should be no self-collision contacts. This verifies
        that the contact exclusions for adjacent arm links and gripper internal
        mechanism are working correctly.
        """
        # Run forward dynamics to detect contacts
        mujoco.mj_forward(geodude_model, geodude_data)

        # Check for any contacts
        assert geodude_data.ncon == 0, (
            f"Expected no contacts at home pose, but found {geodude_data.ncon}. "
            "This may indicate missing contact exclusions."
        )

    def test_gripper_close_no_internal_contacts(self, geodude_model, geodude_data):
        """Gripper can close without internal collision contacts.

        The gripper 4-bar linkage mechanism has parts that pass close to each
        other (coupler <-> spring_link). This test verifies those exclusions
        are working by closing the gripper and checking for contacts.
        """
        # Get gripper actuator indices
        left_gripper_id = mujoco.mj_name2id(
            geodude_model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            "left_ur5e/gripper/fingers_actuator",
        )
        right_gripper_id = mujoco.mj_name2id(
            geodude_model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            "right_ur5e/gripper/fingers_actuator",
        )

        # Close both grippers (255 = fully closed)
        geodude_data.ctrl[left_gripper_id] = 255
        geodude_data.ctrl[right_gripper_id] = 255

        # Step simulation to let gripper close
        for _ in range(200):
            mujoco.mj_step(geodude_model, geodude_data)

        # Check for contacts - there should be none since grippers are closing
        # on empty space and internal mechanism is excluded
        assert geodude_data.ncon == 0, (
            f"Expected no contacts when closing gripper, "
            f"but found {geodude_data.ncon}. "
            "This may indicate missing gripper internal contact exclusions "
            "(coupler <-> spring_link, follower <-> follower)."
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
        assert bodies_excluded(
            "left_ur5e/gripper/right_coupler", "left_ur5e/gripper/right_spring_link"
        ), "Left gripper right coupler <-> spring_link should be excluded"
        assert bodies_excluded(
            "left_ur5e/gripper/left_coupler", "left_ur5e/gripper/left_spring_link"
        ), "Left gripper left coupler <-> spring_link should be excluded"

        # Check critical gripper exclusions (right arm)
        assert bodies_excluded(
            "right_ur5e/gripper/right_coupler", "right_ur5e/gripper/right_spring_link"
        ), "Right gripper right coupler <-> spring_link should be excluded"
        assert bodies_excluded(
            "right_ur5e/gripper/left_coupler", "right_ur5e/gripper/left_spring_link"
        ), "Right gripper left coupler <-> spring_link should be excluded"

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
            assert bodies_excluded(b1, b2), (
                f"Right arm {b1} <-> {b2} should be excluded"
            )
