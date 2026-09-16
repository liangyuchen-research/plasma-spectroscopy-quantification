"""Inspect dataset schemas and row counts using only the Python standard library."""

from pathlib import Path
import csv
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research_paths import data_file


def main() -> None:
    inventory = json.loads((ROOT / "docs/data-inventory.json").read_text(encoding="utf-8"))
    for entry in inventory:
        path = data_file(entry["path"].removeprefix("data/"))
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream)
            columns = next(reader)
            rows = 0
            for row in reader:
                if len(row) != len(columns):
                    raise ValueError(f"Unexpected width at row {rows + 2}: {entry['path']}")
                rows += 1
        assert rows == entry["rows"], entry["path"]
        assert len(columns) == entry["columns"], entry["path"]
        if "training_data_" in path.name or "testing_data_" in path.name:
            wavelength_columns = columns[18:1880]
            assert len(wavelength_columns) == 1862
            wavelengths = [float(value) for value in wavelength_columns]
            assert all(a < b for a, b in zip(wavelengths, wavelengths[1:]))
            assert {"Cu", "Ni", "Pb", "Zn", "Conductivity", "Ontime"}.issubset(columns)
        print(f"{entry['path']}: {rows} rows x {len(columns)} columns")
    print("PASS: dataset inventory, row widths, and standard spectral-column assumptions.")


if __name__ == "__main__":
    main()
