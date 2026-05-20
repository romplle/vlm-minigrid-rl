from pathlib import Path
import sys


def bootstrap() -> None:
    project_root = Path(__file__).resolve().parents[1]
    for path in (project_root / "src", project_root / "nanoVLM", project_root):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
