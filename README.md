# geodude_assets

MuJoCo models for the Geodude bimanual robot.

## Installation

```bash
pip install geodude_assets
```

Or from source:
```bash
git clone https://github.com/personalrobotics/geodude_assets.git
cd geodude_assets
pip install -e .
```

## Quick Start

```bash
# View the robot in MuJoCo's viewer
uv run python -m mujoco.viewer --mjcf="$(uv run python -c 'from geodude_assets import get_model_path; print(get_model_path())')"
```

This opens the MuJoCo viewer with the robot.

## Usage

### Load the Geodude model

```python
import mujoco
from geodude_assets import get_model_path

# Load the full geodude model (two UR5e arms on Vention base)
model = mujoco.MjModel.from_xml_path(str(get_model_path("geodude")))
data = mujoco.MjData(model)

# Step simulation
mujoco.mj_step(model, data)
```

### Available models

```python
from geodude_assets import get_model_path, AVAILABLE_MODELS

print(AVAILABLE_MODELS)
# ['geodude', 'universal_robots_ur5e', 'robotiq_2f140', 'vention', 'abh_left_small', 'abh_right_small']

# Load individual components
ur5e_path = get_model_path("universal_robots_ur5e")
gripper_path = get_model_path("robotiq_2f140")
```

### Build custom robot configurations

To build custom robot configurations (e.g., different gripper combinations), install with the assembly extra:

```bash
pip install geodude_assets[assembly]
```

Then use the assembly module:

```python
from geodude_assets.assembly import attach_arms_to_vention

# Build a model with Robotiq gripper on right arm, Ability Hand on left
model = attach_arms_to_vention(
    save_file=False,
    dir=".",
    filename="custom_geodude.xml",
    left_gripper_type="abhl",
    right_gripper_type="2f140",
)
```

Or use the CLI:

```bash
python -m geodude_assets.assembly --save-mjcf --left-gripper-type abhl --right-gripper-type 2f140
```

## Robot Configuration

The Geodude robot consists of:
- **Vention frame** with vertical linear rails (enclosed lead screw actuators)
- **Two UR5e arms** mounted on the rails
- **Robotiq 2F-140 grippers** on both arms

### Actuators

| Actuator | Joint | Range | Description |
|----------|-------|-------|-------------|
| `left_linear_actuator` | `left_arm_linear_vention` | 0-0.5m | Left arm vertical position (0=bottom, 0.5=top) |
| `right_linear_actuator` | `right_arm_linear_vention` | 0-0.5m | Right arm vertical position |
| `left_ur5e/shoulder_pan` | ... | ±π | Left arm joint 1 |
| `left_ur5e/shoulder_lift` | ... | ±π | Left arm joint 2 |
| ... | ... | ... | ... |
| `left_ur5e/gripper/fingers_actuator` | | 0-255 | Left gripper (0=open, 255=closed) |
| `right_ur5e/gripper/fingers_actuator` | | 0-255 | Right gripper |

## Models

| Model | Description |
|-------|-------------|
| `geodude` | Full robot: two UR5e arms on Vention base with Robotiq 2F-140 grippers |
| `universal_robots_ur5e` | Single UR5e arm |
| `robotiq_2f140` | Robotiq 2F-140 parallel jaw gripper |
| `vention` | Vention aluminum frame base with linear rails |
| `abh_left_small` | Psyonic Ability Hand (left, small) |
| `abh_right_small` | Psyonic Ability Hand (right, small) |

## Model Assembly

The `geodude` model is programmatically generated from component models using `dm_control.mjcf`. This allows:

- Individual components to be inspected/tested independently
- Easy swapping of end effectors (grippers, hands)
- Custom robot configurations

To regenerate the geodude model:

```bash
pip install geodude_assets[assembly]
python -m geodude_assets.assembly --save-mjcf -d src/geodude_assets/models/geodude -l 2f140 -r 2f140
```

## Development

```bash
pip install -e ".[dev]"

# Lint
ruff check .
ruff format .

# Test
pytest
```

## License

MIT
