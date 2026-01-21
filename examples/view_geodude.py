#!/usr/bin/env mjpython
"""View the geodude robot in MuJoCo's viewer.

Usage:
    uv run mjpython examples/view_geodude.py
"""

import mujoco
import mujoco.viewer

from geodude_assets import get_model_path

model = mujoco.MjModel.from_xml_path(str(get_model_path()))
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, 0)  # home pose

mujoco.viewer.launch(model, data)
