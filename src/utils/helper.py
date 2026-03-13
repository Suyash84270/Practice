import os
import json
import yaml
import shutil
from typing import Any, Dict, List


# =========================================================
# FILE & DIRECTORY UTILITIES
# =========================================================

def create_dir(path: str) -> None:
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def create_dirs(paths: List[str]) -> None:
    """Create multiple directories."""
    for path in paths:
        create_dir(path)


def remove_dir(path: str) -> None:
    """Delete directory safely."""
    if os.path.exists(path):
        shutil.rmtree(path)


# =========================================================
# YAML UTILITIES
# =========================================================

def read_yaml(file_path: str) -> Dict[str, Any]:
    """Read YAML file."""
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


def write_yaml(file_path: str, data: Dict[str, Any]) -> None:
    """Write data to YAML file."""
    with open(file_path, "w") as file:
        yaml.dump(data, file)


# =========================================================
# JSON UTILITIES
# =========================================================

def read_json(file_path: str) -> Dict[str, Any]:
    """Read JSON file."""
    with open(file_path, "r") as file:
        return json.load(file)


def write_json(file_path: str, data: Dict[str, Any]) -> None:
    """Write data to JSON file."""
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)


# =========================================================
# FILE UTILITIES
# =========================================================

def get_files(folder_path: str, extensions: List[str] = None) -> List[str]:
    """
    Return all files in folder optionally filtered by extensions.
    Example extensions = ['.jpg', '.png']
    """
    files = []

    for file in os.listdir(folder_path):
        full_path = os.path.join(folder_path, file)

        if os.path.isfile(full_path):
            if extensions:
                if os.path.splitext(file)[1].lower() in extensions:
                    files.append(full_path)
            else:
                files.append(full_path)

    return sorted(files)


def file_exists(file_path: str) -> bool:
    """Check if file exists."""
    return os.path.exists(file_path)


# =========================================================
# PATH UTILITIES
# =========================================================

def get_filename(file_path: str) -> str:
    """Return filename without extension."""
    return os.path.splitext(os.path.basename(file_path))[0]


def get_extension(file_path: str) -> str:
    """Return file extension."""
    return os.path.splitext(file_path)[1]


# =========================================================
# DICTIONARY UTILITIES
# =========================================================

def flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """
    Flatten nested dictionary.

    Example:
    {"a": {"b": 1}} -> {"a.b": 1}
    """
    items = []

    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k

        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))

    return dict(items)