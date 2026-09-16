"""Validate repository structure and preserved input hashes without training."""

from pathlib import Path
import ast
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[1]
TEXT_EXTENSIONS = {".py", ".ipynb", ".md", ".txt", ".json", ".csv"}
CJK = re.compile(r"[\u3400-\u9fff]")


def main() -> None:
    failures = []
    code_cells = 0
    files = 0
    for path in sorted(ROOT.rglob("*")):
        relative = path.relative_to(ROOT)
        if any(part in {".git", ".venv", "__pycache__", "outputs"} for part in relative.parts):
            continue
        if not path.is_file():
            continue
        files += 1
        if CJK.search(str(relative)):
            failures.append(f"Non-English filename: {relative}")
        if path.suffix not in TEXT_EXTENSIONS and not path.name.startswith("."):
            continue
        text = path.read_text(encoding="utf-8-sig")
        if CJK.search(text):
            failures.append(f"Untranslated text: {relative}")
        try:
            if path.suffix == ".py":
                ast.parse(text)
            elif path.suffix == ".ipynb":
                notebook = json.loads(text)
                assert notebook["nbformat"] == 4
                assert isinstance(notebook["cells"], list)
                for index, cell in enumerate(notebook["cells"]):
                    assert cell["cell_type"] in {"code", "markdown", "raw"}
                    if cell["cell_type"] == "code":
                        ast.parse("".join(cell["source"]))
                        assert cell["outputs"] == [], f"Output in cell {index}"
                        assert cell["execution_count"] is None
                        code_cells += 1
        except (SyntaxError, ValueError, KeyError, AssertionError) as error:
            failures.append(f"{relative}: {error}")

    copied_inputs = 0
    ledger = json.loads((ROOT / "docs/provenance.json").read_text(encoding="utf-8"))
    for record in ledger:
        path = ROOT / record["repository_path"]
        if path.suffix in {".csv", ".npy", ".h5"}:
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != record["source_sha256"]:
                failures.append(f"Original-input hash mismatch: {record['repository_path']}")
            copied_inputs += 1

    for failure in failures:
        print(f"FAIL: {failure}")
    if failures:
        raise SystemExit(1)
    print(
        f"PASS: {files} files, {code_cells} notebook code cells, {copied_inputs} preserved numerical inputs."
    )
    print("Syntax and data-integrity checks passed. Full model training was not executed.")


if __name__ == "__main__":
    main()
