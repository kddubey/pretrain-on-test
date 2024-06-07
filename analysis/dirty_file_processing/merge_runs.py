from pathlib import Path
import shutil

from tap import tapify
from tqdm.auto import tqdm


def merge(runs_dir: str, destination_dir: str):
    """
    Merge accuracy data collected across many runs into a single directory which can be
    used for analysis.

    Parameters
    ----------
    runs_dir : str
        Directory containing `run-{timestamp}` directories, which contain the accuracy
        data.
    destination_dir : str
        Directory which will contain all accuracy data from all runs.
    """
    dataset_dirs: set[Path] = set()
    run_dirs = Path(runs_dir).glob("*/")
    for run_dir in run_dirs:
        for file in run_dir.rglob("*"):
            if file.is_file() and file.parent != run_dir:
                dataset_dirs.add(file.parent)

    destination_dir_path = Path(destination_dir)
    destination_dir_path.mkdir(exist_ok=True)
    for dataset_dir in tqdm(dataset_dirs, desc="Copying directories"):
        dst = destination_dir_path / Path(*dataset_dir.parts[3:])
        shutil.copytree(dataset_dir, dst, dirs_exist_ok=True)


if __name__ == "__main__":
    tapify(merge)
