"""Replay MimicGen demos, re-render cameras, record the 6D wrist wrench, and resample to LIFT's 10 Hz.

MimicGen demos are 20 Hz OSC_POSE *delta* actions. Dropping every other delta would lose half the
motion, so the 10 Hz data uses LIFT's single-arm convention instead: the action at 10 Hz frame k is
the absolute end-effector pose reached at the next 10 Hz frame plus the latest gripper command.
LIFT's IPhoneZeroStateAndRealRelativeActions then converts it to a pose relative to the current state.

Per 10 Hz frame k (simulation step t = 2k):
    images   agentview / wrist RGB rendered at --size, observed before the action at step t
    state    eef position (3), axis-angle (3), gripper finger width (1) at step t
    actions  eef pose at step t + 2 (clipped to the last step), gripper command of step t + 1
    wrench   robosuite force_ee / torque_ee sensor (gripper-mount frame, no gravity compensation),
             averaged over the causal window of steps t - 1 and t

Replay modes (--mode):
    forced  (default) before every step, reset the simulator to the stored state of that step, then
            execute the stored action for one step and record the result. Errors cannot accumulate,
            so every demo reproduces its stored trajectory, and the wrench still comes from a real
            controller step (mj_step), not a static evaluation.
    open    reset once to the initial state and execute all actions open loop. Contact-sensitive
            tasks drift (Threading succeeds in about a third of demos); kept for comparison.
The end-effector drift against the stored trajectory and task success are recorded per demo.

Run in the MimicGen environment (Python 3.8):
    MUJOCO_GL=egl /mnt/ssd1/guningquan/miniforge3/envs/mimicgen/bin/python \
        preprocess_data/mimicgen_replay.py \
        --hdf5 /mnt/ssd2/guningquan/Dataset/mimicgen/core/threading_d0.hdf5 \
        --out /mnt/ssd2/guningquan/Dataset/mimicgen/replay/threading_d0_10hz \
        --task "pick up the needle and thread it through the hole" --gpu 4
"""

import argparse
import json
import multiprocessing
import pathlib

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

CAMERAS = {"agentview": "agentview", "wrist": "robot0_eye_in_hand"}
STRIDE = 2  # 20 Hz -> 10 Hz


def _make_env(hdf5, size, gpu):
    import mimicgen  # noqa: F401  (registers MimicGen task variants)
    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.file_utils as FileUtils

    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=hdf5)
    env_meta["env_kwargs"]["render_gpu_device_id"] = gpu
    return EnvUtils.create_env_for_data_processing(
        env_meta=env_meta,
        camera_names=list(CAMERAS.values()),
        camera_height=size,
        camera_width=size,
        reward_shaping=False,
    )


def _record(env, obs):
    robot = env.env.robots[0]
    pose = np.concatenate([obs["robot0_eef_pos"], Rotation.from_quat(obs["robot0_eef_quat"]).as_rotvec()])
    width = obs["robot0_gripper_qpos"][0] - obs["robot0_gripper_qpos"][1]
    return {
        # robomimic's EnvRobosuite already flips MuJoCo's upside-down renders to the usual orientation.
        **{name: obs[f"{cam}_image"].copy() for name, cam in CAMERAS.items()},
        "pose": pose,
        "width": width,
        "wrench": np.concatenate([robot.ee_force, robot.ee_torque]),
    }


def _replay(job):
    hdf5, demo, out, task, size, gpu, mode = job
    global _ENV
    if "_ENV" not in globals():
        _ENV = _make_env(hdf5, size, gpu)
    env = _ENV
    with h5py.File(hdf5, "r") as f:
        g = f[f"data/{demo}"]
        states, actions = g["states"][:], g["actions"][:]
        ref_pos = g["obs/robot0_eef_pos"][:]
        model = g.attrs["model_file"]

    env.reset()
    obs = env.reset_to({"model": model, "states": states[0]})
    steps = [_record(env, obs)]
    drift = 0.0
    for t, action in enumerate(actions):
        if mode == "forced" and t > 0:
            env.reset_to({"states": states[t]})
        obs, _, _, _ = env.step(action)
        steps.append(_record(env, obs))
        if t + 1 < len(ref_pos):
            drift = max(drift, float(np.linalg.norm(obs["robot0_eef_pos"] - ref_pos[t + 1])))
    if mode == "forced":
        # The forced trajectory follows the stored demo, so judge success on its stored final state.
        env.reset_to({"states": states[-1]})
    success = bool(env.is_success()["task"])

    n_steps = len(steps)  # T + 1 observations for T actions
    pose = np.array([s["pose"] for s in steps])
    width = np.array([s["width"] for s in steps])
    wrench = np.array([s["wrench"] for s in steps])
    grip_cmd = actions[:, -1]
    frames = np.arange(0, n_steps, STRIDE)
    target = np.minimum(frames + STRIDE, n_steps - 1)
    episode = {
        "agentview": np.stack([steps[t]["agentview"] for t in frames]),
        "wrist": np.stack([steps[t]["wrist"] for t in frames]),
        "state": np.concatenate([pose[frames], width[frames, None]], axis=1).astype(np.float32),
        "actions": np.concatenate(
            [pose[target], grip_cmd[np.minimum(frames + 1, len(actions) - 1), None]], axis=1
        ).astype(np.float32),
        "wrench": np.stack([wrench[max(t - 1, 0) : t + 1].mean(0) for t in frames]).astype(np.float32),
        "task": np.array(task),
    }
    index = int(demo.split("_")[1])
    np.savez_compressed(out / f"demo_{index}.npz", **episode)
    # Full-rate wrench and gripper command for plotting.
    np.savez_compressed(out / "raw" / f"demo_{index}.npz", wrench=wrench, grip_cmd=grip_cmd, width=width)
    return {"demo": index, "steps_20hz": n_steps, "frames_10hz": len(frames), "drift_m": drift, "success": success}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hdf5", required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--task", required=True, help="language instruction stored with every frame")
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--gpu", type=int, default=0, help="EGL rendering device")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--max-demos", type=int, default=None)
    parser.add_argument("--mode", choices=["forced", "open"], default="forced")
    args = parser.parse_args()

    (args.out / "raw").mkdir(parents=True, exist_ok=True)
    with h5py.File(args.hdf5, "r") as f:
        demos = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
    if args.max_demos is not None:
        demos = demos[: args.max_demos]
    todo = [d for d in demos if not (args.out / f"demo_{int(d.split('_')[1])}.npz").exists()]
    print(f"{len(demos)} demos, {len(demos) - len(todo)} already replayed", flush=True)

    jobs = [(args.hdf5, d, args.out, args.task, args.size, args.gpu, args.mode) for d in todo]
    results = []
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(args.workers) as pool:
        for r in pool.imap_unordered(_replay, jobs):
            results.append(r)
            if len(results) % 50 == 0 or len(results) == len(jobs):
                ok = sum(x["success"] for x in results)
                print(f"{len(results)}/{len(jobs)} replayed, success {ok}/{len(results)}", flush=True)

    log = args.out / "replay_log.jsonl"
    with open(log, "a") as fh:
        for r in sorted(results, key=lambda x: x["demo"]):
            fh.write(json.dumps(r) + "\n")
    if results:
        drift = np.array([r["drift_m"] for r in results])
        print(f"success {sum(r['success'] for r in results)}/{len(results)}, "
              f"eef drift vs stored obs: median {np.median(drift) * 1000:.2f} mm, max {drift.max() * 1000:.2f} mm")


if __name__ == "__main__":
    main()
