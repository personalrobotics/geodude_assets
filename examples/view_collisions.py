#!/usr/bin/env python
"""Interactive collision viewer with visual feedback.

Use the MuJoCo viewer's joint controls to move the robot.
Colliding bodies turn RED. Contact points are shown as dots.

Usage:
    uv run python examples/view_collisions.py

    # Or with mjpython for non-blocking viewer (requires framework Python):
    mjpython examples/view_collisions.py

Viewer tips:
    - Press Tab to show/hide the control panel
    - Use joint sliders to move the robot
    - Press '0' to toggle collision geometry view
    - Press 'C' to toggle contact points
    - Press 'R' to reset to home
"""

import sys

import mujoco
import mujoco.viewer
import numpy as np

from geodude_assets import get_model_path


def main():
    print("Loading model...", flush=True)
    model = mujoco.MjModel.from_xml_path(str(get_model_path()))
    data = mujoco.MjData(model)

    # Disable gravity so robot doesn't collapse when stepping physics
    model.opt.gravity[:] = 0

    # Save original colors for all geoms
    original_rgba = model.geom_rgba.copy()

    # Color for colliding geoms
    COLLISION_COLOR = np.array([1.0, 0.0, 0.0, 1.0])  # Red

    # Start at home
    home_key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, home_key)

    print("\nCollision Viewer", flush=True)
    print("=" * 40, flush=True)
    print("Use joint sliders to move the robot", flush=True)
    print("Colliding bodies turn RED", flush=True)
    print("=" * 40, flush=True)

    def reset_colors():
        """Reset all geom colors to original."""
        model.geom_rgba[:] = original_rgba

    last_collision_msg = [None]  # Use list to allow mutation in nested function

    def highlight_collisions():
        """Color colliding geoms red and print collision info."""
        reset_colors()

        colliding_geoms = set()
        collision_pairs = []
        for i in range(data.ncon):
            contact = data.contact[i]
            colliding_geoms.add(contact.geom1)
            colliding_geoms.add(contact.geom2)

            # Get body names for this contact
            body1 = model.geom_bodyid[contact.geom1]
            body2 = model.geom_bodyid[contact.geom2]
            body1_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body1)
            body2_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body2)
            collision_pairs.append((body1_name, body2_name, contact.dist))

        for geom_id in colliding_geoms:
            model.geom_rgba[geom_id] = COLLISION_COLOR

        # Print collision info if changed
        if collision_pairs:
            msg = f"Collisions ({len(collision_pairs)}): " + ", ".join(
                f"{b1} <-> {b2} (d={d:.4f})" for b1, b2, d in collision_pairs[:3]
            )
            if len(collision_pairs) > 3:
                msg += f" ... +{len(collision_pairs) - 3} more"
        else:
            msg = None

        if msg != last_collision_msg[0]:
            if msg:
                print(msg, flush=True)
            elif last_collision_msg[0] is not None:
                print("No collisions", flush=True)
            last_collision_msg[0] = msg

        return len(colliding_geoms) > 0

    def print_collision_geoms():
        """Print all collision-enabled geoms."""
        print("\nCollision-enabled geoms:")
        print("-" * 60)
        for i in range(model.ngeom):
            contype = model.geom_contype[i]
            conaffinity = model.geom_conaffinity[i]
            if contype > 0 or conaffinity > 0:
                name = (
                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or f"geom_{i}"
                )
                body_id = model.geom_bodyid[i]
                body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
                geom_type = model.geom_type[i]
                type_names = [
                    "plane",
                    "hfield",
                    "sphere",
                    "capsule",
                    "ellipsoid",
                    "cylinder",
                    "box",
                    "mesh",
                ]
                type_name = (
                    type_names[geom_type]
                    if geom_type < len(type_names)
                    else f"type_{geom_type}"
                )
                print(f"  {name} ({type_name}) on body '{body_name}'")
        print("-" * 60)
        collision_count = sum(
            1
            for i in range(model.ngeom)
            if model.geom_contype[i] > 0 or model.geom_conaffinity[i] > 0
        )
        print(f"Total collision geoms: {collision_count}")
        print()

    # Container for viewer reference (needed for key_callback closure)
    viewer_ref = [None]

    def key_callback(keycode):
        if keycode == ord("R") or keycode == ord("r"):
            mujoco.mj_resetDataKeyframe(model, data, home_key)
        elif keycode == ord("P") or keycode == ord("p"):
            print_collision_geoms()
        elif keycode == ord("V") or keycode == ord("v"):
            # Print camera parameters for copying to model
            if viewer_ref[0] is not None:
                cam = viewer_ref[0].cam
                print("\n--- Camera Parameters ---", flush=True)
                print(f"azimuth: {cam.azimuth}", flush=True)
                print(f"elevation: {cam.elevation}", flush=True)
                print(f"distance: {cam.distance}", flush=True)
                lk = cam.lookat
                print(f"lookat: ({lk[0]}, {lk[1]}, {lk[2]})", flush=True)
                print("-------------------------\n", flush=True)

    # Try launch_passive first (requires mjpython on macOS)
    # Fall back to blocking launch() if that fails
    use_passive = True
    if sys.platform == "darwin":
        # Check if we're running under mjpython by testing launch_passive
        try:
            # Quick test - this will fail immediately if not mjpython
            viewer = mujoco.viewer.launch_passive(model, data)
            viewer.close()
        except Exception:
            use_passive = False
            print("\nNote: Using blocking viewer (close window to exit).", flush=True)
            print(
                "For non-blocking viewer, use: mjpython examples/view_collisions.py",
                flush=True,
            )
            print(
                "(Requires framework Python, e.g., from Homebrew or python.org)\n",
                flush=True,
            )

    if use_passive:
        print("\nKeys (non-blocking mode):", flush=True)
        print("  Tab  - show/hide control panel", flush=True)
        print("  0    - toggle collision geometry", flush=True)
        print("  C    - toggle contact points", flush=True)
        print("  R    - reset to home", flush=True)
        print("  V    - print camera parameters", flush=True)
        print(flush=True)
        with mujoco.viewer.launch_passive(
            model, data, key_callback=key_callback
        ) as viewer:
            viewer_ref[0] = viewer  # Store for key_callback access

            # Enable contact visualization
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True

            # Camera defaults come from model (statistic/visual elements)

            while viewer.is_running():
                # Step physics so actuators respond to control sliders
                mujoco.mj_step(model, data)
                highlight_collisions()
                viewer.sync()

            # Print camera params on close
            cam = viewer.cam
            print("\n--- Final Camera Parameters ---", flush=True)
            print(f"azimuth: {cam.azimuth}", flush=True)
            print(f"elevation: {cam.elevation}", flush=True)
            print(f"distance: {cam.distance}", flush=True)
            lk = cam.lookat
            print(f"lookat: ({lk[0]}, {lk[1]}, {lk[2]})", flush=True)
            print("-------------------------------\n", flush=True)
    else:
        # Blocking viewer - custom key_callback not supported
        # Built-in shortcuts: Backspace=reset, C=contacts, 0=collisions
        print("\nBuilt-in keys (blocking mode):", flush=True)
        print("  Backspace - reset simulation", flush=True)
        print("  C         - toggle contact points", flush=True)
        print("  0         - toggle collision geometry", flush=True)
        print("  Tab       - show/hide control panel", flush=True)
        print(flush=True)
        mujoco.viewer.launch(model, data)

    # Restore colors on exit
    reset_colors()
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
