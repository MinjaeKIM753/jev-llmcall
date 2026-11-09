from __future__ import annotations

from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[1]


def repo_root() -> Path:
    """Directory that contains policy/ and catalogue/. This package lives inside it."""
    policy = _ROOT / "policy" / "policy.example.yaml"
    if not policy.is_file():
        raise FileNotFoundError(f"Policy file not found: {policy}")
    return _ROOT


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not contain a mapping.")
    return data


def load_bundle(root: Path | None = None) -> tuple[dict, dict, dict]:
    base = root or repo_root()
    policy = load_yaml(base / "policy" / "policy.example.yaml")
    tasks = load_yaml(base / "policy" / "tasks.yaml")
    catalogue = load_yaml(base / "catalogue" / "models.example.yaml")
    return policy, tasks, catalogue
