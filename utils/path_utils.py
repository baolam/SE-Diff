from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def repo_path(*parts: str) -> Path:
    return REPO_ROOT.joinpath(*parts)


def ensure_exists(path_str: str, description: str) -> Path:
    path = Path(path_str).expanduser()
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {description}: {path}. "
            "Please place the required prerequisite file in the expected location "
            "or pass an explicit path by CLI/config."
        )
    return path
