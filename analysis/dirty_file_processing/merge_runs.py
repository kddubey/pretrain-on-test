"""
Merge runs
"""

from pathlib import Path
import shutil

run_dir = "runs"
destination_dir = Path("accuracies")

dataset_dirs: set[Path] = set()
run_dirs = Path(run_dir).glob("*/")
for run_dir in run_dirs:
    for file in run_dir.rglob("*"):
        if file.is_file() and file.parent != run_dir:
            dataset_dirs.add(file.parent)

destination_dir.mkdir()
for dataset_dir in dataset_dirs:
    dst = destination_dir / Path(*dataset_dir.parts[3:])
    shutil.copytree(dataset_dir, dst, dirs_exist_ok=True)
