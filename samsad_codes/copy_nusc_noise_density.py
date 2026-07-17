import os
import shutil
from pathlib import Path


def get_sensor_type(path):
    parts = path.parts
    for p in parts:
        if "LIDAR" in p:
            return "lidar"
        if "CAM" in p:
            return "camera"
    return "other"


def copy_file_with_fallback(src_sim, src_orig, dst, stats, rel_path):
    sensor_type = get_sensor_type(rel_path)

    if src_sim.exists():
        shutil.copy2(src_sim, dst)
        stats[sensor_type]["sim"] += 1
    elif src_orig.exists():
        shutil.copy2(src_orig, dst)
        stats[sensor_type]["orig"] += 1
    else:
        stats[sensor_type]["missing"] += 1
        print(f"[WARNING] Missing file: {rel_path}")


def merge_sensor_folder(folder_name, orig_root, sim_root, out_root, stats):
    orig_dir = Path(orig_root) / folder_name
    sim_dir = Path(sim_root) / folder_name
    out_dir = Path(out_root) / folder_name

    for root, _, files in os.walk(orig_dir):
        rel_path = Path(root).relative_to(orig_dir)

        for file in files:
            orig_file = Path(root) / file
            sim_file = sim_dir / rel_path / file
            out_file = out_dir / rel_path / file

            out_file.parent.mkdir(parents=True, exist_ok=True)

            copy_file_with_fallback(
                sim_file, orig_file, out_file, stats, rel_path
            )


def print_stats(stats):
    print("\n===== DATA SOURCE REPORT =====")

    for sensor in ["lidar", "camera", "other"]:
        s = stats[sensor]
        total = s["sim"] + s["orig"]

        if total == 0:
            continue

        sim_pct = 100 * s["sim"] / total
        orig_pct = 100 * s["orig"] / total

        print(f"\n[{sensor.upper()}]")
        print(f"Simulated: {s['sim']} ({sim_pct:.2f}%)")
        print(f"Original : {s['orig']} ({orig_pct:.2f}%)")
        print(f"Missing  : {s['missing']}")


def validate_sim_usage(stats, min_sim_ratio=0.05):
    total_sim = sum(stats[s]["sim"] for s in stats)
    total = sum(stats[s]["sim"] + stats[s]["orig"] for s in stats)

    if total == 0:
        raise RuntimeError("No files processed.")

    ratio = total_sim / total

    if ratio < min_sim_ratio:
        raise RuntimeError(
            f"Simulated data usage too low ({ratio:.4f}). "
            "Likely wrong sim_root path."
        )


def copy_folder(src, dst):
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def create_nuscenes(orig_root, sim_root, out_root, version = "v1.0-trainval"):
    orig_root = Path(orig_root)
    sim_root = Path(sim_root)
    out_root = Path(out_root)

    stats = {
        "lidar": {"sim": 0, "orig": 0, "missing": 0},
        "camera": {"sim": 0, "orig": 0, "missing": 0},
        "other": {"sim": 0, "orig": 0, "missing": 0},
    }

    out_root.mkdir(parents=True, exist_ok=True)

    print("Merging samples...")
    merge_sensor_folder("samples", orig_root, sim_root, out_root, stats)

    print("Merging sweeps...")
    merge_sensor_folder("sweeps", orig_root, sim_root, out_root, stats)

    print("Copying maps...")
    copy_folder(orig_root / "maps", out_root / "maps")

    print(f"Copying metadata ({version})...")
    copy_folder(orig_root / version, out_root / version)

    print_stats(stats)

    # Safety check
    validate_sim_usage(stats, min_sim_ratio=0.01)

    print("\nDone.")


if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sensor", default="", help="Sensor type")
    parser.add_argument("--fault", default="", help="Fault type")
    parser.add_argument("--sub_fault", default="beam_drop_30", help="Sub-fault type")
    parser.add_argument("--orig_root", default="/home/user/Documents/dev_exps/sensor-fusion/OpenPCDet/data/v1.0-trainval", help="Original root directory")
    parser.add_argument("--sim_root", default="/home/user/Documents/dev_exps/sensor-fusion/OpenPCDet/data/v1.0-trainval-sim", help="Simulated root directory")
    parser.add_argument("--out_root", default="/home/user/Documents/dev_exps/sensor-fusion/OpenPCDet/data/nuscenes_density", help="Output root directory")

    args = parser.parse_args()

    sensor = args.sensor
    fault = args.fault
    sub_fault = args.sub_fault
    orig_root = args.orig_root
    sim_root = f"{args.sim_root}/{sensor}/{fault}/{sub_fault}"
    out_root = args.out_root
    os.makedirs(out_root, exist_ok=True)

    create_nuscenes(
        orig_root=orig_root,
        sim_root=sim_root,
        out_root=out_root,
        version="v1.0-trainval"
    )