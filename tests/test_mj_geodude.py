"""Tests for mj_geodude package."""

import mujoco
import pytest

from mj_geodude import (
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
            aid = mujoco.mj_name2id(
                geodude_model, mujoco.mjtObj.mjOBJ_ACTUATOR, name
            )
            assert aid != -1, f"Actuator {name} not found"

    def test_linear_actuator_range(self, geodude_model):
        """Linear actuators have correct range (0=bottom, 0.5=top)."""
        for name in ["left_linear_actuator", "right_linear_actuator"]:
            aid = mujoco.mj_name2id(
                geodude_model, mujoco.mjtObj.mjOBJ_ACTUATOR, name
            )
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
