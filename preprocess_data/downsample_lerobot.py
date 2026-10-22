"""Temporally downsample a LeRobot v2.1 dataset with embedded images into a new dataset.

Every `stride`-th frame is kept (frames 0, stride, 2*stride, ...). Images, state,
action, and other per-frame features are taken from the kept frame. Features
listed in --mean-keys (by default the 6D `left_wrench`, when present) are instead
averaged over the causal window [t - stride + 1, t] ending at each kept frame, an
anti-aliasing filter that uses no future readings, so the force stays causal for
LIFT's latency-masked force memory.

timestamp, frame_index, and index are regenerated for the new rate; meta/info.json,
episodes.jsonl, and episodes_stats.jsonl are rewritten. The source is not modified.

Example (30 Hz -> 10 Hz):
    uv run python preprocess_data/downsample_lerobot.py \
        --src /mnt/ssd2/guningquan/Dataset/board_insertion_right_force_lerobot \
        --dst /mnt/ssd2/guningquan/Dataset/board_insertion_right_force_10hz_lerobot \
        --stride 3
"""

import argparse
import json
import math
import multiprocessing
import pathlib
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

_REGENERATED = ("timestamp", "frame_index", "index")


def _read_jsonl(path: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_jsonl(path: pathlib.Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _vector_column(table: pa.Table, name: str) -> np.ndarray:
    """Return a numeric column as [rows, dim] (dim=1 for scalar columns)."""
    column = table[name].combine_chunks()
    if pa.types.is_fixed_size_list(column.type) or pa.types.is_list(column.type):
        values = np.asarray(column.flatten())
        return values.reshape(len(column), -1)
    return np.asarray(column).reshape(len(column), 1)


def _stats(x: np.ndarray) -> dict:
    x = x.astype(np.float64)
    return {
        "min": x.min(0).tolist(),
        "max": x.max(0).tolist(),
        "mean": x.mean(0).tolist(),
        "std": x.std(0).tolist(),
        "count": [len(x)],
    }


def _causal_window_mean(x: np.ndarray, keep: np.ndarray, stride: int) -> np.ndarray:
    cumsum = np.concatenate([np.zeros((1, x.shape[1])), np.cumsum(x.astype(np.float64), axis=0)])
    start = np.maximum(keep - stride + 1, 0)
    return ((cumsum[keep + 1] - cumsum[start]) / (keep + 1 - start)[:, None]).astype(x.dtype)


def _downsample_episode(job: dict) -> tuple[int, dict]:
    src, dst, rel = pathlib.Path(job["src"]), pathlib.Path(job["dst"]), job["rel"]
    stride, fps, index_offset = job["stride"], job["fps"], job["index_offset"]
    # pre_buffer=False is much faster on dictionary-encoded image columns.
    table = pq.read_table(src / rel, pre_buffer=False)
    keep = np.arange(0, table.num_rows, stride)
    out = table.take(pa.array(keep))

    for key in job["mean_keys"]:
        values = _vector_column(table, key)
        averaged = _causal_window_mean(values, keep, stride)
        field = table.schema.field(key)
        column = pa.FixedSizeListArray.from_arrays(
            pa.array(averaged.reshape(-1), field.type.value_type), values.shape[1]
        )
        out = out.set_column(out.schema.get_field_index(key), field, column)

    n = len(keep)
    regenerated = {
        "timestamp": pa.array((np.arange(n) / fps).astype(np.float32), out.schema.field("timestamp").type),
        "frame_index": pa.array(np.arange(n), out.schema.field("frame_index").type),
        "index": pa.array(np.arange(n) + index_offset, out.schema.field("index").type),
    }
    for name, column in regenerated.items():
        out = out.set_column(out.schema.get_field_index(name), out.schema.field(name), column)
    out = out.replace_schema_metadata(table.schema.metadata)

    (dst / rel).parent.mkdir(parents=True, exist_ok=True)
    # Unique PNG bytes gain nothing from dictionary encoding; plain pages read much faster.
    numeric = [name for name in out.column_names if not pa.types.is_struct(out.schema.field(name).type)]
    pq.write_table(out, dst / rel, row_group_size=100, use_dictionary=numeric)

    stats = {name: _stats(_vector_column(out, name)) for name in numeric}
    return n, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=pathlib.Path, required=True, help="source LeRobot dataset root")
    parser.add_argument("--dst", type=pathlib.Path, required=True, help="new dataset root (must not exist)")
    parser.add_argument("--stride", type=int, default=3, help="keep every N-th frame; new fps = fps / N")
    parser.add_argument(
        "--mean-keys",
        nargs="*",
        default=None,
        help="vector features averaged over the causal window instead of sampled (default: left_wrench if present)",
    )
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--max-episodes", type=int, default=None, help="convert only the first N (debugging)")
    parser.add_argument("--overwrite", action="store_true", help="delete --dst first if it exists")
    args = parser.parse_args()

    if args.src.resolve() == args.dst.resolve():
        raise ValueError("--dst must differ from --src; the source dataset is never modified")
    info = json.loads((args.src / "meta" / "info.json").read_text())
    if info.get("total_videos", 0):
        raise ValueError("video-backed datasets are not supported; images must be embedded in the parquet files")
    src_fps = info["fps"]
    new_fps = src_fps / args.stride
    if args.stride < 1 or not math.isclose(new_fps, round(new_fps)):
        raise ValueError(f"fps {src_fps} is not divisible by stride {args.stride}")
    new_fps = round(new_fps)
    mean_keys = args.mean_keys
    if mean_keys is None:
        mean_keys = [key for key in ("left_wrench",) if key in info["features"]]
    for key in mean_keys:
        if key not in info["features"] or info["features"][key]["dtype"] not in ("float32", "float64"):
            raise ValueError(f"--mean-keys entry {key!r} must be a float feature of {args.src}")

    if args.dst.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.dst} exists; pass --overwrite to replace it")
        shutil.rmtree(args.dst)
    (args.dst / "meta").mkdir(parents=True)

    episodes = _read_jsonl(args.src / "meta" / "episodes.jsonl")
    if args.max_episodes is not None:
        episodes = episodes[: args.max_episodes]
    src_stats = {row["episode_index"]: row for row in _read_jsonl(args.src / "meta" / "episodes_stats.jsonl")}

    jobs, offset = [], 0
    for ep in episodes:
        idx = ep["episode_index"]
        jobs.append(
            {
                "src": str(args.src),
                "dst": str(args.dst),
                "rel": info["data_path"].format(episode_chunk=idx // info["chunks_size"], episode_index=idx),
                "stride": args.stride,
                "fps": new_fps,
                "index_offset": offset,
                "mean_keys": mean_keys,
            }
        )
        offset += math.ceil(ep["length"] / args.stride)

    new_episodes, new_stats = [], []
    with multiprocessing.Pool(args.workers) as pool:
        for ep, (length, stats) in zip(episodes, pool.imap(_downsample_episode, jobs), strict=True):
            idx = ep["episode_index"]
            new_episodes.append({**ep, "length": length})
            merged = dict(src_stats[idx]["stats"])
            # Image statistics are per-channel colour stats over sampled frames; subsampling keeps them valid.
            for name, value in merged.items():
                if name not in stats:
                    merged[name] = {**value, "count": [min(value["count"][0], length)]}
            merged.update(stats)
            new_stats.append({"episode_index": idx, "stats": merged})
            print(f"episode {idx:3d}: {ep['length']} -> {length} frames", flush=True)

    total_frames = sum(ep["length"] for ep in new_episodes)
    if total_frames != offset:
        raise RuntimeError(f"frame count mismatch: wrote {total_frames}, expected {offset}")
    info.update(fps=new_fps, total_frames=total_frames, total_episodes=len(new_episodes))
    info["splits"] = {"train": f"0:{len(new_episodes)}"}
    (args.dst / "meta" / "info.json").write_text(json.dumps(info, indent=4))
    _write_jsonl(args.dst / "meta" / "episodes.jsonl", new_episodes)
    _write_jsonl(args.dst / "meta" / "episodes_stats.jsonl", new_stats)
    shutil.copy(args.src / "meta" / "tasks.jsonl", args.dst / "meta" / "tasks.jsonl")
    for extra in ("synthetic_wrench.json",):
        if (args.src / "meta" / extra).exists():
            shutil.copy(args.src / "meta" / extra, args.dst / "meta" / extra)
    provenance = {
        "source": str(args.src),
        "source_fps": src_fps,
        "fps": new_fps,
        "stride": args.stride,
        "kept_frames": "0, stride, 2*stride, ... of each source episode",
        "mean_keys": mean_keys,
        "mean_window": "causal [t - stride + 1, t]",
    }
    (args.dst / "meta" / "downsample.json").write_text(json.dumps(provenance, indent=2))
    print(f"wrote {len(new_episodes)} episodes / {total_frames} frames at {new_fps} Hz to {args.dst}")


if __name__ == "__main__":
    main()
