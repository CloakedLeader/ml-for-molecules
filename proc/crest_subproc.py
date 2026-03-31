import subprocess
from pathlib import Path

def run_crest_search(filename: str, tmp_dir: Path) -> list[tuple[float, float, float]]:
    cmd = ["crest", filename, "--gfn2", "--quick"]

    subprocess.run(cmd, cwd=tmp_dir, check=True)

    best_xyz = tmp_dir / "crest_best.xyz"
    with open(best_xyz, "r") as f:
        lines = f.readlines()

    coords = []
    for line in lines[2:]:
        parts = line.split()
        x, y, z = map(float, parts[1:4])
        coords.append((x,y,z))

    return coords