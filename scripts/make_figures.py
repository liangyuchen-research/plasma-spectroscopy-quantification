"""Generate the README figures from the included data and the archived Transformer checkpoint.

Outputs are written to ``docs/figures/``. The script uses only files that ship
with the repository: the recollected Cu spectra, the ``Cu_Transfer.h5``
checkpoint and its training history. Nothing is retrained.

    python scripts/make_figures.py            # all figures
    python scripts/make_figures.py --no-model # spectra and history only (no TensorFlow)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research_paths import checkpoint_file, data_file, history_file  # noqa: E402

FIG_DIR = ROOT / "docs" / "figures"
OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
CU_LINES = {"Cu I 324.75": 324.754, "Cu I 327.40": 327.396}

# Wavelength windows zeroed before modelling (Pb, Zn and Ni emission regions),
# copied from the wastewater notebooks so the checkpoint sees the same input.
ZEROED_WINDOWS = [
    ("278.051", "286.721"), ("360.189", "375.312"), ("400.718", "410.412"),
    ("210.081", "220.118"), ("468.665", "485.098"), ("230.303", "240.078"),
    ("335.114", "360.016"),
]
CLIP = 60000.0


def style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9, "axes.labelsize": 9.5, "axes.titlesize": 9.5,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
        "axes.linewidth": 0.7, "lines.linewidth": 1.1, "lines.markersize": 4,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "axes.prop_cycle": mpl.cycler(color=OKABE_ITO),
        "legend.frameon": False,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
        "figure.dpi": 100,
    })


def load(name: str, columns: list[str] | None = None):
    """Return (wavelength, spectra, concentration, conductivity, columns)."""
    frame = pd.read_csv(data_file(f"spectra/{name}.csv"), dtype=float)
    cols = list(frame.columns)
    ref = columns or cols
    values = frame.to_numpy()
    wavelength = np.array([float(c) for c in cols[18:1880]])
    spectra = np.clip(values[:, 18:1880], None, CLIP)
    zero = np.concatenate([
        np.arange(ref.index(lo) - 18, ref.index(hi) - 18 + 1) for lo, hi in ZEROED_WINDOWS
    ])
    spectra[:, zero] = 0.0
    spectra /= CLIP
    return wavelength, spectra, values[:, cols.index("Cu")], values[:, cols.index("Conductivity")], cols


def save(fig, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.png", dpi=200)
    plt.close(fig)
    print(f"wrote docs/figures/{name}.png")


def mark_lines(ax, y_frac: float = 0.97) -> None:
    for label, wl in CU_LINES.items():
        ax.axvline(wl, color="0.35", lw=0.6, ls="--", zorder=0)
    ax.text(CU_LINES["Cu I 327.40"] + 1.2, y_frac, "Cu I\n324.75 / 327.40 nm",
            transform=ax.get_xaxis_transform(), ha="left", va="top", fontsize=8, color="0.25")


def figure_spectra() -> None:
    wl, spectra, cu, cond, _ = load("recollected_training")
    raw = pd.read_csv(data_file("spectra/recollected_training.csv"), dtype=float).to_numpy()[:, 18:1880]
    fig, (ax, axd) = plt.subplots(1, 2, figsize=(7.4, 2.9), constrained_layout=True,
                                  gridspec_kw={"width_ratios": [2, 1]})
    levels = sorted(set(cu))
    means = {level: raw[(cu == level) & (cond == 3500)].mean(axis=0) for level in levels}
    for color, level in zip(OKABE_ITO, levels):
        ax.plot(wl, means[level] / 1000, color=color, lw=0.7, label=f"{level:g} ppm Cu")
    ax.set_xlim(200, 517)
    ax.set_ylim(0, 72)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Intensity (10$^3$ counts)")
    ax.legend(loc="center right", title="3500 µS/cm, mean of 30 spectra", title_fontsize=7.5)
    for wl_line in CU_LINES.values():
        ax.axvline(wl_line, color="0.35", lw=0.6, ls="--", zorder=0)
    ax.annotate("Cu I 324.75 / 327.40 nm\n(on the OH band shoulder)", xy=(327.4, 36), xytext=(362, 58),
                fontsize=7.5, color="0.25", arrowprops={"arrowstyle": "-", "color": "0.4", "lw": 0.6})
    ax.text(300, 67.5, "saturated OH / N$_2$ bands", fontsize=7, color="0.4", ha="center")
    window = (wl >= 318) & (wl <= 334)
    for color, level in zip(OKABE_ITO[1:], levels[1:]):
        diff = (means[level] - means[0.0]) / 1000
        axd.plot(wl[window], diff[window], color=color, lw=1.2, label=f"{level:g} ppm")
    axd.axhline(0, color="0.5", lw=0.5)
    for wl_line in CU_LINES.values():
        axd.axvline(wl_line, color="0.35", lw=0.6, ls="--", zorder=0)
    axd.set_xlim(318, 334)
    axd.set_xlabel("Wavelength (nm)")
    axd.set_ylabel("Δ intensity vs 0 ppm (10$^3$ counts)")
    axd.legend(loc="upper right")
    for a, tag in ((ax, "(a)"), (axd, "(b)")):
        a.text(0.02, 0.97, tag, transform=a.transAxes, fontweight="bold", va="top")
    save(fig, "spectra_by_concentration")


def figure_history() -> None:
    hist = np.load(history_file("Cu_Transfer.npy"), allow_pickle=False)
    fig, ax = plt.subplots(figsize=(3.6, 2.7), constrained_layout=True)
    epochs = np.arange(1, hist.shape[1] + 1)
    ax.plot(epochs, hist[1], label="Training MAE", color=OKABE_ITO[0], lw=0.9)
    ax.plot(epochs, hist[3], label="Validation MAE", color=OKABE_ITO[1], lw=0.9)
    best = int(np.argmin(hist[3]))
    ax.plot(epochs[best], hist[3][best], "o", color=OKABE_ITO[1], ms=4)
    ax.annotate(f"best epoch {epochs[best]}\n{hist[3][best]:.2f} ppm", (epochs[best], hist[3][best]),
                xytext=(0.55, 0.6), textcoords="axes fraction", fontsize=8,
                arrowprops={"arrowstyle": "-", "color": "0.4", "lw": 0.6})
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean absolute error (ppm)")
    ax.legend(loc="upper right")
    save(fig, "training_history")


def batched_predict(model, spectra: np.ndarray, batch: int = 512) -> np.ndarray:
    return np.concatenate([
        model.predict(spectra[i:i + batch], verbose=0).ravel() for i in range(0, len(spectra), batch)
    ])


def figure_parity(model) -> dict:
    _, train_x, train_y, _, cols = load("recollected_training")
    _, test_x, test_y, test_cond, _ = load("recollected_testing", cols)
    train_p = batched_predict(model, train_x)
    test_p = batched_predict(model, test_x)
    mae = float(np.mean(np.abs(test_p - test_y)))
    mape = float(np.mean(np.abs(test_p - test_y) / test_y) * 100)
    rng = np.random.default_rng(0)

    fig, ax = plt.subplots(figsize=(3.7, 3.7), constrained_layout=True)
    lim = (-1, 23)
    ax.plot(lim, lim, ls="--", color="0.4", lw=0.8, label="1:1")
    ax.fill_between(lim, [0.9 * v for v in lim], [1.1 * v for v in lim], color="0.85", lw=0, label="±10 %")
    jitter = rng.normal(0, 0.12, len(train_y))
    ax.plot(train_y + jitter, train_p, "o", ms=3, mfc="none", mec=OKABE_ITO[0], mew=0.7, alpha=0.6,
            label=f"training levels (n = {len(train_y)})")
    ax.plot(test_y + rng.normal(0, 0.12, len(test_y)), test_p, "^", ms=4, color=OKABE_ITO[1],
            mec="white", mew=0.4, label=f"unseen levels (n = {len(test_y)})")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Reference Cu concentration (ppm)")
    ax.set_ylabel("Predicted Cu concentration (ppm)")
    per_level = "\n".join(
        f"{level:g} ppm → {test_p[test_y == level].mean():.1f} ± {test_p[test_y == level].std():.1f} ppm"
        for level in sorted(set(test_y))
    )
    ax.text(0.03, 0.97, f"unseen levels: MAE = {mae:.2f} ppm, MAPE = {mape:.1f} %\n{per_level}",
            transform=ax.transAxes, va="top", fontsize=8)
    ax.legend(loc="lower right", fontsize=7.5)
    ax.set_aspect("equal")
    save(fig, "transformer_parity")
    return {"mae": mae, "mape": mape, "n_test": int(len(test_y)), "n_train": int(len(train_y))}


def occlusion_importance(model, spectrum: np.ndarray, window_sizes) -> np.ndarray:
    """Sliding-window occlusion, as in the explainability notebooks (batched)."""
    base = float(model.predict(spectrum[None, :], verbose=0).ravel()[0])
    n = len(spectrum)
    total = np.zeros(n)
    for ws in window_sizes:
        positions = n // ws
        batch = np.repeat(spectrum[None, :], positions, axis=0)
        for t in range(positions):
            batch[t, ws * t: ws * (t + 1)] = 0.0
        preds = batched_predict(model, batch)
        weight = np.log2(max(window_sizes)) / np.log2(ws)
        each = np.zeros(n)
        for t in range(positions):
            each[ws * t: ws * (t + 1)] = -(preds[t] - base) * weight
        total += each
    return total / np.max(total)


def figure_occlusion(model) -> None:
    _, _, _, _, cols = load("recollected_training")
    wl, test_x, test_y, test_cond, _ = load("recollected_testing", cols)
    idx = int(np.flatnonzero((test_y == 15) & (test_cond == 3500))[0])
    spectrum = test_x[idx]
    window_sizes = [5 * (i + 1) for i in range(20)]
    importance = occlusion_importance(model, spectrum, window_sizes)
    pred = float(model.predict(spectrum[None, :], verbose=0).ravel()[0])

    fig, (ax_map, ax) = plt.subplots(2, 1, figsize=(7.4, 3.4), sharex=True, constrained_layout=True,
                                     gridspec_kw={"height_ratios": [1, 3.2]})
    ax_map.pcolormesh(wl, [0, 1], np.vstack([importance, importance]), cmap="viridis", shading="auto")
    ax_map.set_yticks([])
    ax_map.set_ylabel("occlusion\nimportance", fontsize=8)
    ax.plot(wl, spectrum, color="k", lw=0.7)
    ax.set_xlim(wl.min(), wl.max())
    ax.set_ylim(0, spectrum.max() * 1.08)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Normalised intensity")
    mark_lines(ax, 0.95)
    ax.text(0.99, 0.95, f"test spectrum, 15 ppm Cu, 3500 µS/cm\nprediction {pred:.1f} ppm",
            transform=ax.transAxes, ha="right", va="top", fontsize=8)
    ax.text(0.01, 0.95, "zeroed windows: Pb, Zn, Ni regions", transform=ax.transAxes, fontsize=7.5,
            color="0.4", va="top")
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=mpl.colors.Normalize(importance.min(), 1))
    cbar = fig.colorbar(sm, ax=[ax_map, ax], pad=0.01, fraction=0.03)
    cbar.set_label("relative importance", fontsize=8)
    cbar.ax.tick_params(labelsize=7.5)
    save(fig, "occlusion_importance")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-model", action="store_true", help="skip the figures that load the checkpoint")
    args = parser.parse_args()
    style()
    figure_spectra()
    figure_history()
    if args.no_model:
        return
    from tensorflow.keras.models import load_model  # noqa: E402  (imported late: slow)

    model = load_model(checkpoint_file("Cu_Transfer.h5"), compile=False)
    metrics = figure_parity(model)
    print("held-out metrics:", {k: round(v, 3) if isinstance(v, float) else v for k, v in metrics.items()})
    figure_occlusion(model)


if __name__ == "__main__":
    main()
