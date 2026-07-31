"""Multi-robot control loop for combined_mobile_keyboard_control.py -m."""
from __future__ import annotations
import numpy as np


def run(c):
    carb = c.carb
    c.open_stage(c.MULTI_STAGE_PATH)
    for _ in range(20):
        c.simulation_app.update()
    stage = c.get_current_stage()
    c.configure_mobile_joint_usd_limits(stage)
    c.configure_wheel_joint_usd_drives(stage)
    c.configure_arm_joint_usd_drives(stage, c.ROBOT_CONFIGS["v4_onlyarm"])
    from pxr import Usd, UsdGeom, UsdPhysics

    def root_below(scope):
        found = [str(p.GetPath()) for p in c.iter_world_prims(stage)
                 if str(p.GetPath()).startswith(scope + "/")
                 and p.HasAPI(UsdPhysics.ArticulationRootAPI)]
        if len(found) == 1:
            return found[0]
        fallback = scope + "/base_mobile"
        prim = stage.GetPrimAtPath(fallback)
        if prim and prim.IsValid():
            if not prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                UsdPhysics.ArticulationRootAPI.Apply(prim)
                print(f"[WARN] Applied missing ArticulationRootAPI to {fallback}")
            return fallback
        raise RuntimeError(f"Expected one articulation below {scope}; found {found}")

    def prim_below(scope, name):
        found = [p for p in c.iter_world_prims(stage)
                 if str(p.GetPath()).startswith(scope + "/") and p.GetName() == name]
        if len(found) != 1:
            raise RuntimeError(f"Expected one {name} below {scope}; found {[str(p.GetPath()) for p in found]}")
        return found[0]

    world = c.World()
    states = {}
    for slot, spec in c.MULTI_ROBOT_SPECS.items():
        root = root_below(spec["root_scope"])
        c.configure_robot_gravity(stage, root, disable=False)
        robot = c.Articulation(root, name=f"multi_robot_{slot}")
        world.scene.add(robot)
        states[slot] = dict(spec, slot=slot, root_path=root, robot=robot)
    world.reset()
    for s in states.values():
        c.wait_for_articulation_ready(world, s["robot"])

    for s in states.values():
        robot = s["robot"]
        ctl = robot.get_articulation_controller()
        wheels = c.resolve_wheel_joint_indices(robot)
        if len(wheels) != 4:
            raise RuntimeError(f"{s['name']} requires 4 wheel DOFs; found {len(wheels)}")
        mobile = c.resolve_mobile_joint_indices(robot)
        s.update(controller=ctl, wheel_indices=wheels, mobile_indices=mobile,
                 dof_names=c.get_dof_names(robot), mode=("drive" if s["full_control"] or not mobile else "ladder_task"), active_arm="right",
                 arm_target="right", paint_enabled={"right": False, "left": False},
                 mobile_targets=(robot.get_joint_positions()[mobile].copy()
                                 if mobile else np.array([], dtype=np.float64)),
                 mobile_sign=np.ones(len(mobile)), mobile_scale=np.ones(len(mobile)),
                 mobile_enabled=np.ones(len(mobile), dtype=bool),
                 ladder_kin=c.LadderKinematics(c.LADDER_KINEMATICS_PATH))
        cfg = None
        arm_force, arm_kp, arm_kd = c.ARM_JOINT_MAX_FORCE, c.KP, c.KD
        if s["full_control"]:
            cfg = dict(c.ROBOT_CONFIGS[s["robot_key"]])
            cfg["stage_path"] = c.MULTI_STAGE_PATH
            arm_force = float(cfg.get("arm_max_force", arm_force))
            arm_kp, arm_kd = float(cfg.get("arm_kp", arm_kp)), float(cfg.get("arm_kd", arm_kd))
        s["cfg"] = cfg
        efforts = np.ones(robot.num_dof) * arm_force
        efforts[wheels] = c.WHEEL_MAX_FORCE
        if mobile:
            efforts[mobile] = c.MOBILE_JOINT_MAX_FORCE
            efforts[mobile[0]] = c.MOBILE_JOINT1_MAX_FORCE
        if hasattr(robot, "set_max_efforts"):
            robot.set_max_efforts(efforts)
        if hasattr(ctl, "set_gains"):
            kp, kd = np.ones(robot.num_dof) * arm_kp, np.ones(robot.num_dof) * arm_kd
            kp[wheels], kd[wheels] = 0.0, c.WHEEL_DRIVE_DAMPING
            if mobile:
                kp[mobile], kd[mobile] = c.MOBILE_JOINT_STIFFNESS, c.MOBILE_JOINT_DAMPING
                kp[mobile[0]], kd[mobile[0]] = c.MOBILE_JOINT1_STIFFNESS, c.MOBILE_JOINT1_DAMPING
            ctl.set_gains(kp, kd)
        if not s["full_control"]:
            s.update(arms={}, ee_prims={}, paint=None)
            continue
        rkin = c.LulaKinematicsSolver(robot_description_path=cfg["right_desc_yaml"], urdf_path=cfg["urdf_path"])
        lkin = c.LulaKinematicsSolver(robot_description_path=cfg["left_desc_yaml"], urdf_path=cfg["urdf_path"])
        c.set_ik_iters(rkin, 80); c.set_ik_iters(lkin, 80)
        arms = {
            "right": c.ArmState("right", rkin, c.ArticulationKinematicsSolver(robot, rkin, cfg["right_ee_frame"]), cfg["right_ee_frame"]),
            "left": c.ArmState("left", lkin, c.ArticulationKinematicsSolver(robot, lkin, cfg["left_ee_frame"]), cfg["left_ee_frame"]),
        }
        for arm in arms.values():
            pos, rot = arm.ik.compute_end_effector_pose()
            arm.target_pos, arm.target_quat = np.asarray(pos), c.as_wxyz_quat(rot)
            arm.start_target_pos = arm.target_pos.copy()
        saved = c.load_kinematics_info(cfg["kinematics_info_path"])
        c.apply_saved_kinematics_to_arms(arms, saved)
        s["mobile_sign"], s["mobile_scale"], s["mobile_enabled"], _ = c.load_mobile_calibration(saved, len(mobile))
        s["arms"] = arms
        s["ee_prims"] = {side: prim_below(s["root_scope"], cfg[f"{side}_ee_frame"]) for side in ("right", "left")}
        s["paint"] = c.PaintSprayVisualizer(
            stage,
            s["root_path"],
            scope_path=f"/World/PaintSimulation_{s['slot']}",
            nozzle_axis=cfg.get("paint_nozzle_axis", (0.0, 1.0, 0.0)),
        )

    selected_slot, selected = 1, states[1]
    status = c.ControlStatusWindow(f"1: {selected['name']}", c.MULTI_STAGE_PATH)
    status.set_slots(
        "1: humanoid | 2: v4 onlyarm | 3: only_mobile | 4: EMPTY | 5: EMPTY"
    )
    help_ui = c.ControlHelpWindow()
    target_selector = c.PaintTargetAreaSelector(stage, status.set_event)
    fob = c.FOBControlBridge()
    fob.start()
    status.set_fob_status("connecting /dev/ttyUSB1")
    arduino_switch = c.ArduinoSwitchSource()
    arduino_switch.start()
    status.set_arduino_status("connecting /dev/ttyUSB0 @ 9600")
    for state in states.values():
        paint_visualizer = state.get("paint")
        if paint_visualizer is not None:
            paint_visualizer.coverage_callback = target_selector.record_paint_patch
            paint_visualizer.coverage_clear_callback = target_selector.clear_coverage
    pressed, last_pressed = set(), set()
    should_quit = False
    iface = carb.input.acquire_input_interface()
    import omni.appwindow
    keyboard = omni.appwindow.get_default_app_window().get_keyboard()

    def keyboard_event(event, *_):
        nonlocal should_quit
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            pressed.add(event.input)
            should_quit |= event.input == carb.input.KeyboardInput.ESCAPE
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            pressed.discard(event.input)
    iface.subscribe_to_keyboard_events(keyboard, keyboard_event)
    down = lambda key: key in pressed

    def refresh_base(s):
        t, qraw = s["robot"].get_world_pose()
        q = c.to_wxyz_from_xyzw(qraw) if c.FORCE_BASE_WXYZ_SWAP else np.asarray(qraw)
        for arm in s["arms"].values():
            arm.kin.set_robot_base_pose(t, q)

    def help_text(s):
        print("\n" + "=" * 72)
        print(f"[MULTI] Selected robot {s['slot']}: {s['name']}")
        print("  1: humanoid | 2: v4 onlyarm | 3: only_mobile | 4: EMPTY (camera) | 5: EMPTY (target area)")
        print("  [DRIVE] W/S: forward/reverse | A/D: left/right | Space: brake")
        if s["full_control"]:
            print("  B: DRIVE/ARM | V: ARM/MOBILE")
            print("  P: ARM/FOB pose control (/dev/ttyUSB1)")
            print("  [ARM] TAB: cycle RIGHT/LEFT/BOTH | arrows: X/Y | Shift+Up/Down: Z")
            print("  [ARM] I/K: pitch | J/H: yaw | U/O: roll | R: reset")
            print("  [PAINT] Z: rectangle | hold X: spray | C: clear")
            print("  [V MOBILE] Q/A W/S E/D R/F T/G Y/H U/J I/K: joint1..joint8 +/-")
            print("  [V MOBILE] Arrows: body8 task control (X reversed, Shift=Z, Left/Right=joint1)")
        elif s["mobile_indices"]:
            print("  [LADDER] Up/Down: body8 X forward/back | Shift+Up/Down: body8 Z up/down")
            print("  [LADDER] Left/Right: joint1 turn | B: LADDER/DRIVE")
        else:
            print("  only_mobile asset has wheel DOFs only; DRIVE mode is available.")
        print(f"  ESC: quit | MODE={s['mode'].upper()} | PRIM={s['root_path']}")
        print("=" * 72)

    def set_mode(s, mode):
        if not s["full_control"] and mode not in ("drive", "ladder_task"):
            return
        if s["mode"] == "drive" and mode != "drive":
            c.safe_apply_action(s["controller"], c.make_wheel_drive_action(s["wheel_indices"], 0, 0))
        s["mode"] = mode
        if mode == "arm":
            for arm in s["arms"].values():
                pos, rot = arm.ik.compute_end_effector_pose()
                arm.target_pos, arm.target_quat = np.asarray(pos), c.as_wxyz_quat(rot)
                arm.start_target_pos = arm.target_pos.copy()
        elif mode in ("ladder_task", "mobile"):
            s["mobile_targets"][:] = s["robot"].get_joint_positions()[s["mobile_indices"]]
            s["ladder_kin"].reset_orientation_reference(s["mobile_targets"])
        status.set_event(f"Robot {s['slot']} {s['name']} / {mode.upper()}")

        help_ui.update_context(f"{s['slot']}: {s['name']}", mode, wheels=True, multi=True)
    def update_status(s, command):
        q, qd = s["robot"].get_joint_positions(), s["robot"].get_joint_velocities()
        if q is None:
            return
        deg = np.rad2deg(np.asarray(q))
        arm_ids = [i for i, n in enumerate(s["dof_names"])
                   if n not in c.MOBILE_JOINT_NAMES and n not in c.WHEEL_JOINT_NAMES]
        def rows(ids, values):
            return "\n".join(f"{s['dof_names'][i]:<28} {values[i]:+9.2f}" for i in ids) if ids else "not available"
        status.set_fob_status(fob.status)
        status.set_arduino_status(arduino_switch.status)
        status.set_robot(f"{s['slot']}: {s['name']}")
        status.update(mode=f"MULTI / ROBOT {s['slot']} / {s['mode'].upper()}",
                      target=s["name"], error="-", command=command,
                      paint=(f"{s['active_arm'].upper()} rectangle={'ON' if s['paint_enabled'][s['active_arm']] else 'OFF'}"
                             f" / spray={'ON' if (down(carb.input.KeyboardInput.X) or (s['mode'] == 'fob' and arduino_switch.active)) else 'OFF'}"
                             f" / {target_selector.coverage_status}"
                             if s["full_control"] else "not available"),
                      arm_joints=rows(arm_ids, deg), ladder_joints=rows(s["mobile_indices"], deg),
                      wheel_joints=rows(s["wheel_indices"], np.asarray(qd)) if qd is not None else "not available")

    help_text(selected)
    help_ui.update_context(f"1: {selected['name']}", selected["mode"], wheels=True, multi=True)
    dt, frame = 1 / 60, 0
    nums = [
        carb.input.KeyboardInput.KEY_1, carb.input.KeyboardInput.KEY_2,
        carb.input.KeyboardInput.KEY_3, carb.input.KeyboardInput.KEY_4,
        carb.input.KeyboardInput.KEY_5,
    ]
    mobile_keys = {n: getattr(carb.input.KeyboardInput, n) for pair in c.MOBILE_JOINT_KEY_BINDINGS for n in pair}
    while c.simulation_app.is_running() and not should_quit:
        world.step(render=True); frame += 1
        new = pressed - last_pressed; last_pressed = set(pressed)
        switched = False
        for slot, key in enumerate(nums, 1):
            if key not in new or slot == selected_slot:
                continue
            if selected is not None:
                c.safe_apply_action(selected["controller"], c.make_wheel_drive_action(selected["wheel_indices"], 0, 0))
            selected_slot = slot
            if slot == 5:
                selected = None
                target_selector.begin_selection()
                status.show_empty(5, "TARGET AREA", "Mode 5: TARGET AREA")
                help_ui.update(
                    "5: TARGET AREA", "TARGET AREA",
                    [
                        "Point at PaintWall_A or PaintWall_B: live camera ray",
                        "F: select first/second diagonal corner",
                        "1/2/3: select a robot | 4: free camera",
                        "ESC: quit",
                    ],
                )
                print("[MULTI] Mode 5: TARGET AREA")
            elif slot == 4:
                selected = None
                target_selector.end_selection()
                status.show_empty(4, "FREE CAMERA", "Free camera mode: robot controls disabled")
                print("[MULTI] Free camera mode (4): all robot controls disabled")
                help_ui.show_free_camera("1: humanoid | 2: v4 onlyarm | 3: only_mobile | 4: EMPTY | 5: EMPTY")
            else:
                selected = states[slot]
                target_selector.end_selection()
                status.set_event(f"Selected robot {slot}: {selected['name']}")
                help_text(selected)
                help_ui.update_context(f"{slot}: {selected['name']}", selected["mode"], wheels=True, multi=True)
            switched = True
            break
        if target_selector.active:
            if carb.input.KeyboardInput.F in new:
                target_selector.select_hover_point()
            target_selector.update()
        if switched or selected is None:
            continue
        if not selected["full_control"] and selected["mobile_indices"] and carb.input.KeyboardInput.B in new:
            set_mode(selected, "drive" if selected["mode"] == "ladder_task" else "ladder_task")
        if selected["full_control"]:
            if carb.input.KeyboardInput.B in new:
                set_mode(selected, "arm" if selected["mode"] == "drive" else "drive")
            if carb.input.KeyboardInput.V in new:
                set_mode(selected, "mobile" if selected["mode"] == "arm" else "arm")
        mode, ctl = selected["mode"], selected["controller"]
        if mode == "drive":
            fwd = down(carb.input.KeyboardInput.W) - down(carb.input.KeyboardInput.S)
            turn = down(carb.input.KeyboardInput.A) - down(carb.input.KeyboardInput.D)
            if down(carb.input.KeyboardInput.SPACE): fwd = turn = 0
            c.safe_apply_action(ctl, c.make_wheel_drive_action(selected["wheel_indices"], fwd, turn))
            if frame % 10 == 0: update_status(selected, f"forward={fwd:+.0f} turn={turn:+.0f}")
            continue
        if mode == "ladder_task":
            shift = down(carb.input.KeyboardInput.LEFT_SHIFT) or down(carb.input.KeyboardInput.RIGHT_SHIFT)
            forward = 0.0 if shift else down(carb.input.KeyboardInput.DOWN) - down(carb.input.KeyboardInput.UP)
            lift = down(carb.input.KeyboardInput.UP) - down(carb.input.KeyboardInput.DOWN) if shift else 0.0
            turn = down(carb.input.KeyboardInput.LEFT) - down(carb.input.KeyboardInput.RIGHT)
            count = min(len(selected["mobile_targets"]), len(selected["ladder_kin"].joints))
            if count:
                delta = selected["ladder_kin"].command_delta(selected["mobile_targets"][:count], forward, lift, turn, dt)
                delta *= selected["mobile_sign"][:count] * selected["mobile_scale"][:count]
                delta *= selected["mobile_enabled"][:count]
                selected["mobile_targets"][:count] += delta
                selected["mobile_targets"][:] = np.clip(selected["mobile_targets"], c.MOBILE_JOINT_LOWER_RAD, c.MOBILE_JOINT_UPPER_RAD)
                c.safe_apply_action(ctl, c.make_mobile_joint_action(selected["mobile_indices"], selected["mobile_targets"]))
            if frame % 10 == 0:
                update_status(selected, f"body8 forward={forward:+.0f} lift={lift:+.0f} turn={turn:+.0f}")
            continue
        if mode == "mobile":
            shift = down(carb.input.KeyboardInput.LEFT_SHIFT) or down(carb.input.KeyboardInput.RIGHT_SHIFT)
            forward = 0.0 if shift else down(carb.input.KeyboardInput.DOWN) - down(carb.input.KeyboardInput.UP)
            lift = down(carb.input.KeyboardInput.UP) - down(carb.input.KeyboardInput.DOWN) if shift else 0.0
            turn = down(carb.input.KeyboardInput.LEFT) - down(carb.input.KeyboardInput.RIGHT)
            count = min(len(selected["mobile_targets"]), len(selected["ladder_kin"].joints))
            if count:
                task_delta = selected["ladder_kin"].command_delta(selected["mobile_targets"][:count], forward, lift, turn, dt)
                task_delta *= selected["mobile_sign"][:count] * selected["mobile_scale"][:count]
                task_delta *= selected["mobile_enabled"][:count]
                selected["mobile_targets"][:count] += task_delta
            # Preserve the original eight per-joint letter bindings.
            for i, (plus, minus) in enumerate(c.MOBILE_JOINT_KEY_BINDINGS[:len(selected["mobile_targets"])]):
                if selected["mobile_enabled"][i]:
                    direction = down(mobile_keys[plus]) - down(mobile_keys[minus])
                    selected["mobile_targets"][i] += direction * c.MOBILE_JOINT_SPEED_RPS * dt * c.MOBILE_JOINT_COMMAND_SCALE[i] * selected["mobile_sign"][i] * selected["mobile_scale"][i]
            selected["mobile_targets"][:] = np.clip(selected["mobile_targets"], c.MOBILE_JOINT_LOWER_RAD, c.MOBILE_JOINT_UPPER_RAD)
            c.safe_apply_action(ctl, c.make_mobile_joint_action(selected["mobile_indices"], selected["mobile_targets"]))
            if frame % 10 == 0: update_status(selected, "body8 task + individual joint control")
            continue
        arms = selected["arms"]
        if carb.input.KeyboardInput.TAB in new:
            order = ["right", "left", "both"]
            selected["arm_target"] = order[(order.index(selected["arm_target"]) + 1) % 3]
            if selected["arm_target"] != "both": selected["active_arm"] = selected["arm_target"]
            status.set_event(f"Arm target -> {selected['arm_target'].upper()}")
        active = selected["active_arm"]
        if carb.input.KeyboardInput.P in new:
            if selected["mode"] == "fob":
                set_mode(selected, "arm")
                status.set_event("FOB mode -> ARM")
            elif selected["mode"] == "arm":
                arm = arms[active]
                cur_pos, cur_rot = arm.ik.compute_end_effector_pose()
                arm.target_pos = np.asarray(cur_pos, dtype=np.float64)
                arm.target_quat = c.as_wxyz_quat(cur_rot)
                arm.start_target_pos = arm.target_pos.copy()
                fob_key = f"{selected['slot']}:{active}"
                ok, fob_status = fob.capture_reference(
                    fob_key, arm.target_pos, arm.target_quat
                )
                status.set_fob_status(fob_status)
                if ok:
                    selected["arm_target"] = active
                    set_mode(selected, "fob")
                    status.set_event(f"FOB controls robot {selected['slot']} {active.upper()} arm")
                else:
                    status.set_event(f"FOB {fob_status}")
        controlled = [arms["right"], arms["left"]] if selected["arm_target"] == "both" else [arms[selected["arm_target"]]]
        if carb.input.KeyboardInput.Z in new:
            selected["paint_enabled"][active] = not selected["paint_enabled"][active]
        if carb.input.KeyboardInput.C in new: selected["paint"].clear()
        refresh_base(selected)
        shift = down(carb.input.KeyboardInput.LEFT_SHIFT) or down(carb.input.KeyboardInput.RIGHT_SHIFT)
        desired = np.zeros(3)
        desired[2 if shift else 0] = c.EE_SPEED_MPS * dt * (down(carb.input.KeyboardInput.UP) - down(carb.input.KeyboardInput.DOWN))
        desired[1] = c.EE_SPEED_MPS * dt * (down(carb.input.KeyboardInput.LEFT) - down(carb.input.KeyboardInput.RIGHT))
        desired *= c.KEYBOARD_BASE_AXIS_SIGN
        for arm in controlled:
            sign = np.asarray(selected["cfg"].get("arm_input_axis_sign", {}).get(arm.name, [1, 1, 1]))
            arm.target_pos += desired * sign
            arm.target_pos = np.minimum(np.maximum(arm.target_pos, arm.start_target_pos - c.MAX_TARGET_OFFSET_FROM_START),
                                        arm.start_target_pos + c.MAX_TARGET_OFFSET_FROM_START)
        if selected["mode"] == "fob":
            fob_key = f"{selected['slot']}:{active}"
            fob_position, fob_quaternion, fob_status = fob.target(fob_key)
            status.set_fob_status(fob_status)
            if fob_position is not None:
                arm = arms[active]
                arm.target_pos = np.minimum(
                    np.maximum(fob_position, arm.start_target_pos - c.MAX_TARGET_OFFSET_FROM_START),
                    arm.start_target_pos + c.MAX_TARGET_OFFSET_FROM_START,
                )
                arm.target_quat = fob_quaternion
            else:
                status.set_event(f"FOB {fob_status}")
        if carb.input.KeyboardInput.R in new:
            for arm in controlled:
                pos, rot = arm.ik.compute_end_effector_pose()
                arm.target_pos, arm.target_quat = np.asarray(pos), c.as_wxyz_quat(rot)
                arm.start_target_pos = arm.target_pos.copy()
            if selected["mode"] == "fob":
                ok, fob_status = fob.capture_reference(
                    f"{selected['slot']}:{active}", arms[active].target_pos, arms[active].target_quat
                )
                status.set_fob_status(fob_status)
        da = c.ROT_SPEED_RPS * dt if selected["mode"] == "arm" else 0.0
        rotations = [([1,0,0], (down(carb.input.KeyboardInput.U)-down(carb.input.KeyboardInput.O))*da),
                     ([0,1,0], (down(carb.input.KeyboardInput.I)-down(carb.input.KeyboardInput.K))*da),
                     ([0,0,1], (down(carb.input.KeyboardInput.J)-down(carb.input.KeyboardInput.H))*da)]
        for arm in controlled:
            for axis, angle in rotations:
                if angle: arm.target_quat = c.quat_norm(c.quat_mul(c.quat_from_axis_angle(axis, angle), arm.target_quat))
            action, ok = arm.ik.compute_inverse_kinematics(arm.target_pos, arm.target_quat)
            c.safe_apply_action(ctl, action if ok else None)
        c.safe_apply_action(ctl, c.make_mobile_joint_action(selected["mobile_indices"], selected["mobile_targets"]))
        matrix = UsdGeom.Xformable(selected["ee_prims"][active]).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        spray_active = down(carb.input.KeyboardInput.X) or (selected["mode"] == "fob" and arduino_switch.active)
        selected["paint"].update(active, matrix, selected["paint_enabled"][active], spray_active)
        if frame % 10 == 0: update_status(selected, f"arm target={selected['arm_target'].upper()}")
    fob.close()
    arduino_switch.close()
    for s in states.values():
        c.safe_apply_action(s["controller"], c.make_wheel_drive_action(s["wheel_indices"], 0, 0))
    c.simulation_app.close()
