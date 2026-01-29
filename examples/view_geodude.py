#!/usr/bin/env python
"""View the geodude robot in MuJoCo's viewer.

Usage:
    .venv-viewer/bin/mjpython examples/view_geodude.py  # Best: front camera, home pose
    uv run python examples/view_geodude.py              # Fallback: blocking viewer
"""

import sys

import mujoco
import mujoco.viewer

from geodude_assets import get_model_path

model = mujoco.MjModel.from_xml_path(str(get_model_path()))
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, 0)  # home pose

# Try launch_passive with camera control, fall back to blocking viewer
try:
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Set camera: front view looking at workspace
        viewer.cam.azimuth = -90
        viewer.cam.elevation = -25
        viewer.cam.lookat[:] = [0, -0.4, 1.1]
        viewer.cam.distance = 3.0

        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()

        # Print final camera parameters
        cam = viewer.cam
        print(f"\n--- Final Camera Parameters ---", flush=True)
        print(f"azimuth: {cam.azimuth}", flush=True)
        print(f"elevation: {cam.elevation}", flush=True)
        print(f"distance: {cam.distance}", flush=True)
        print(f"lookat: [{cam.lookat[0]}, {cam.lookat[1]}, {cam.lookat[2]}]", flush=True)
        print(f"-------------------------------\n", flush=True)
except Exception as e:
    if "mjpython" in str(e).lower():
        print("For front camera view, run with mjpython:", flush=True)
        print("  .venv-viewer/bin/mjpython examples/view_geodude.py", flush=True)
    mujoco.viewer.launch(model, data)
