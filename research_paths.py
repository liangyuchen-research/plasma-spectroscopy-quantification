"""Portable input and output locations for the preserved research notebooks."""

from pathlib import Path
import os

REPOSITORY_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("PLASMA_DATA_DIR", REPOSITORY_ROOT / "data"))
EXTERNAL_ROOT = Path(os.environ.get("PLASMA_EXTERNAL_DATA_DIR", DATA_ROOT / "external"))
ARTIFACT_ROOT = Path(os.environ.get("PLASMA_ARTIFACT_DIR", REPOSITORY_ROOT / "artifacts"))
OUTPUT_ROOT = Path(os.environ.get("PLASMA_OUTPUT_DIR", REPOSITORY_ROOT / "outputs"))


def _input_file(root: Path, relative: str) -> Path:
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(
            f"Required research input is missing: {path}. "
            "See docs/reproduction.md for the data inventory and external input slots."
        )
    return path


def data_file(relative: str) -> Path:
    """Return a supplied CSV dataset or fail with an actionable error."""
    return _input_file(DATA_ROOT, relative)


def external_file(name: str) -> Path:
    """Resolve an additional acquisition absent from the supplied snapshot."""
    return _input_file(EXTERNAL_ROOT, name)


def checkpoint_file(name: str) -> str:
    """Return an archived checkpoint path for legacy TensorFlow/Keras APIs."""
    return str(_input_file(ARTIFACT_ROOT / "checkpoints", name))


def history_file(name: str) -> Path:
    """Return an archived NumPy training-history array."""
    return _input_file(ARTIFACT_ROOT / "training_histories", name)


def output_file(name: str, experiment: str) -> Path:
    """Keep newly generated results separate from supplied data and artifacts."""
    path = OUTPUT_ROOT / experiment / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
