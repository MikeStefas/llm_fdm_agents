import os
import shutil

HARNESS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
SHARED_DIR = os.path.join(HARNESS_ROOT, "shared")
DATA_DIR = os.path.join(HARNESS_ROOT, "data")

os.makedirs(SHARED_DIR, exist_ok=True)


def reset_shared_directory(shared_dir: str | None = None) -> None:
    target_dir = shared_dir or SHARED_DIR
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)
        return

    for item in os.listdir(target_dir):
        item_path = os.path.join(target_dir, item)
        try:
            if os.path.islink(item_path) or os.path.isfile(item_path):
                os.unlink(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
        except Exception as e:
            print(f"Warning: Failed to remove '{item_path}' during shared dir reset: {e}")


__all__ = [
    "HARNESS_ROOT",
    "SHARED_DIR",
    "DATA_DIR",
    "reset_shared_directory",
]
