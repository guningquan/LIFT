"""Write replayed MimicGen episodes (from mimicgen_replay.py) as a LeRobot v2.1 dataset for LIFT.

Each input .npz holds one 10 Hz episode with:
    agentview, wrist   uint8 [T, H, W, 3]   third-person and wrist RGB
    state              float32 [T, 7]       eef position (3), axis-angle (3), gripper width (1)
    actions            float32 [T, 7]       absolute eef target for the next 10 Hz frame, same layout,
                                            with the gripper command (-1 open, 1 close) last
    wrench             float32 [T, 6]       wrist force (N) and torque (N*m), sensor frame
    task               str

Feature names follow LIFT's single-arm convention (state / actions / left_wrench / left_wrist_img),
so LeRobotSingleiPhoneFlexivDataConfig-style transforms apply unchanged.

Usage (LIFT environment):
    uv run python preprocess_data/mimicgen_replay_to_lerobot.py \
        --replay-dir /mnt/ssd2/guningquan/Dataset/mimicgen/replay/threading_d0_10hz \
        --repo-id mimicgen_threading_d0_10hz_lerobot \
        --root /mnt/ssd2/guningquan/Dataset/mimicgen_threading_d0_10hz_lerobot
"""

import argparse
import json
import pathlib
import shutil

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import tqdm


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--replay-dir", type=pathlib.Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--root", type=pathlib.Path, required=True, help="output dataset directory")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--image-writer-threads", type=int, default=4)
    parser.add_argument("--image-writer-processes", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--only-success", action="store_true", help="keep demos marked successful in replay_log.jsonl"
    )
    args = parser.parse_args()

    files = sorted(args.replay_dir.glob("demo_*.npz"), key=lambda p: int(p.stem.split("_")[1]))
    if args.only_success:
        log = [json.loads(line) for line in (args.replay_dir / "replay_log.jsonl").read_text().splitlines()]
        keep = {r["demo"] for r in log if r["success"]}
        dropped = [p.stem for p in files if int(p.stem.split("_")[1]) not in keep]
        files = [p for p in files if int(p.stem.split("_")[1]) in keep]
        print(f"--only-success: dropped {len(dropped)} demos {dropped}")
    if args.max_episodes is not None:
        files = files[: args.max_episodes]
    if not files:
        raise FileNotFoundError(f"no demo_*.npz in {args.replay_dir}")
    if args.root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.root} exists; pass --overwrite to replace it")
        shutil.rmtree(args.root)

    with np.load(files[0]) as first:
        height, width = first["wrist"].shape[1:3]
    image = {"dtype": "image", "shape": (height, width, 3), "names": ["height", "width", "channel"]}
    features = {
        "left_wrist_img": image,
        "agentview_img": image,
        "state": {"dtype": "float32", "shape": (7,), "names": ["x", "y", "z", "rx", "ry", "rz", "gripper_width"]},
        "actions": {"dtype": "float32", "shape": (7,), "names": ["x", "y", "z", "rx", "ry", "rz", "gripper_cmd"]},
        "left_wrench": {"dtype": "float32", "shape": (6,), "names": ["fx", "fy", "fz", "tx", "ty", "tz"]},
    }
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        root=args.root,
        robot_type="robosuite_panda",
        features=features,
        use_videos=False,
        image_writer_threads=args.image_writer_threads,
        image_writer_processes=args.image_writer_processes,
    )

    for path in tqdm.tqdm(files, desc="episodes"):
        # NpzFile decompresses an array on every key access, so load each array once.
        with np.load(path) as npz:
            ep = {key: npz[key] for key in npz.files}
        task = str(ep["task"])
        for t in range(len(ep["state"])):
            dataset.add_frame(
                {
                    "left_wrist_img": ep["wrist"][t],
                    "agentview_img": ep["agentview"][t],
                    "state": ep["state"][t],
                    "actions": ep["actions"][t],
                    "left_wrench": ep["wrench"][t],
                    "task": task,
                }
            )
        dataset.save_episode()
    print(f"wrote {dataset.num_episodes} episodes / {dataset.num_frames} frames to {args.root}")


if __name__ == "__main__":
    main()
