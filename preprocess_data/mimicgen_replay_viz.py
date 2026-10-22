"""Plot the replayed wrist wrench and write side-by-side camera videos (.webm) for a few demos.

Run in the MimicGen environment:
    /mnt/ssd1/guningquan/miniforge3/envs/mimicgen/bin/python preprocess_data/mimicgen_replay_viz.py \
        --replay-dir /mnt/ssd2/guningquan/Dataset/mimicgen/replay/threading_d0_10hz --demos 0 2 3
"""

import argparse
import pathlib

import imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot(raw, ep, path, title):
    wrench, grip_cmd, width = raw["wrench"], raw["grip_cmd"], raw["width"]
    t20 = np.arange(len(wrench)) / 20.0
    t10 = np.arange(len(ep["wrench"])) / 10.0
    fig, ax = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    for i, n in enumerate("xyz"):
        ax[0].plot(t20, wrench[:, i], label=f"F{n}")
    ax[0].plot(t20, np.linalg.norm(wrench[:, :3], axis=1), "k", lw=1.5, label="|F|")
    ax[0].set_ylabel("force (N)")
    for i, n in enumerate("xyz"):
        ax[1].plot(t20, wrench[:, 3 + i], label=f"T{n}")
    ax[1].set_ylabel("torque (N·m)")
    ax[2].plot(t20, np.linalg.norm(wrench[:, :3], axis=1), color="0.7", label="|F| 20 Hz")
    ax[2].step(t10, np.linalg.norm(ep["wrench"][:, :3], axis=1), where="post", color="C3", label="|F| 10 Hz (stored)")
    ax[2].set_ylabel("force (N)")
    ax[3].step(t20[:-1], grip_cmd, where="post", label="gripper cmd (1 = close)")
    ax[3].plot(t20, width / 0.08, label="finger width / 0.08 m")
    ax[3].set_ylabel("gripper")
    ax[3].set_xlabel("time (s)")
    for a in ax:
        a.legend(loc="upper left", fontsize=8, ncol=4)
        a.grid(alpha=0.3)
    ax[0].set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--replay-dir", type=pathlib.Path, required=True)
    parser.add_argument("--demos", type=int, nargs="+", default=[0])
    args = parser.parse_args()
    out = args.replay_dir / "viz"
    out.mkdir(exist_ok=True)
    for d in args.demos:
        raw = np.load(args.replay_dir / "raw" / f"demo_{d}.npz")
        ep = np.load(args.replay_dir / f"demo_{d}.npz")
        plot(raw, ep, out / f"demo_{d}_wrench.png", f"demo {d}: wrist wrench (sensor frame) during replay")
        frames = np.concatenate([ep["agentview"], ep["wrist"]], axis=2)  # side by side
        imageio.mimwrite(out / f"demo_{d}.webm", frames, fps=10, codec="libvpx-vp9", quality=8, macro_block_size=1)
        print(f"demo {d}: {out / f'demo_{d}_wrench.png'}  {out / f'demo_{d}.webm'}")


if __name__ == "__main__":
    main()
