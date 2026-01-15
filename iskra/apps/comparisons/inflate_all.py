import itertools
import subprocess
from pathlib import Path


def main():
    mesh_dir = Path().home() / "Dropbox" / "Data" / "iskra-data" / "mcf"
    results_root_dir = Path().home() / "experiments" / "iskra" / "mcf"
    results_root_dir.mkdir(exist_ok=True, parents=True)

    mesh_paths = list(mesh_dir.glob("*.obj"))
    devices = ["cpu", "cuda"]
    dtypes = ["float64"]
    methods = ["theseus"]  # "iskra", "alec",
    script_module = "iskra.apps.comparisons.inflate"
    t_value = 0.001

    combinations = list(itertools.product(mesh_paths, devices, dtypes, methods))
    print(f"Running {len(combinations)} total configurations...\n")

    for i, (mesh_path, device, dtype, method) in enumerate(combinations, 1):
        out_dir = results_root_dir / Path(mesh_path).stem / f"{method}_{device}_{dtype}"
        out_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"[{i}/{len(combinations)}] {mesh_path.name}, device={device}, dtype={dtype}, method={method}"
        )
        cmd = [
            "python",
            "-m",
            script_module,
            str(mesh_path),
            "--results_dir",
            str(out_dir),
            "--t",
            str(t_value),
            "--dtype",
            dtype,
            "--device",
            device,
            "--method",
            method,
        ]

        try:
            result = subprocess.run(cmd, check=True, capture_output=False, text=True)
            print("  ✓ Success\n")
        except subprocess.CalledProcessError as e:
            print(f"  ✗ Failed: {e.stderr}\n")

    print("All runs complete!")


if __name__ == "__main__":
    main()
