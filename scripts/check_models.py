"""Check notebook model setup and checkpoint inference without running training."""

import argparse
import ast
import contextlib
import io
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def check_notebook(path):
    """Run setup statements only, stopping before the first model.fit call."""
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "\n\n".join(
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"
    )
    tree = ast.parse(source)
    setup = []
    for statement in tree.body:
        if any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "fit"
            for node in ast.walk(statement)
        ):
            break
        setup.append(statement)
    namespace = {"__name__": "notebook_model_check"}
    previous = Path.cwd()
    try:
        os.chdir(ROOT)
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(ast.Module(body=setup, type_ignores=[]), str(path), "exec"), namespace)
        models = [
            value
            for name, value in namespace.items()
            if name.endswith("_Model") and isinstance(value, tf.keras.Model)
        ]
        if not models:
            raise AssertionError("No model was constructed before the training cell.")
        for model in models:
            inputs = [tf.zeros((2, *tensor.shape.as_list()[1:])) for tensor in model.inputs]
            output = model(inputs[0] if len(inputs) == 1 else inputs, training=False).numpy()
            assert output.shape == (2, 1) and np.isfinite(output).all()
            update = model.train_on_batch(
                inputs[0] if len(inputs) == 1 else inputs, tf.ones((2, 1))
            )
            assert np.isfinite(np.asarray(update)).all()
    finally:
        os.chdir(previous)
        plt.close("all")
        tf.keras.backend.clear_session()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--notebooks",
        action="store_true",
        help="Also execute each notebook up to its training call",
    )
    args = parser.parse_args()
    failures = []
    for path in sorted((ROOT / "artifacts/checkpoints").glob("*.h5")):
        try:
            model = tf.keras.models.load_model(path, compile=False)
            inputs = [tf.zeros((2, *tensor.shape.as_list()[1:])) for tensor in model.inputs]
            result = model(inputs[0] if len(inputs) == 1 else inputs, training=False).numpy()
            assert result.shape == (2, 1) and np.isfinite(result).all()
            print(f"PASS checkpoint: {path.name}")
        except Exception as error:
            failures.append(f"{path.name}: {type(error).__name__}: {error}")
        finally:
            tf.keras.backend.clear_session()
    if args.notebooks:
        for path in sorted((ROOT / "notebooks").rglob("*.ipynb")):
            try:
                check_notebook(path)
                print(f"PASS notebook setup: {path.relative_to(ROOT).as_posix()}")
            except Exception as error:
                failures.append(
                    f"{path.relative_to(ROOT).as_posix()}: {type(error).__name__}: {error}"
                )
    for failure in failures:
        print(f"FAIL: {failure}")
    if failures:
        raise SystemExit(1)
    print(
        "Model setup and finite-output checks passed. Full training and scientific accuracy were not evaluated."
    )


if __name__ == "__main__":
    main()
