"""Double-peak PSTH figure for GRB006 first-stim responses.

Uses the archived local GRB006 trial timestamps plus the KS4 spike-time export
to identify double-peak units with the same unsmoothed 10 ms-bin settings used
for the GRB058 Dario figure.

Output:
    figures/grb006_double_peak_examples.pdf

Figure layout:
    2 rows x 3 columns showing the top-ranked GRB006 double-peak units,
    ranked by the smaller of the two peak heights above baseline.
"""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ephys.src.utils.utils_analysis import (
    classify_peak_count,
    compute_population_peth,
    compute_unit_selectivity,
)


SESSION = "20240821_121447"
TRIAL_TS_PATH = Path(
    "/Users/gabriel/data/GRB006/20240821_121447/pre_processed/trial_ts.pkl"
)
SPIKE_TIMES_PATHS = [
    Path(
        "/Users/gabriel/data/GRB006/20240821_121447/pre_processed/"
        "20240821_121447_ks4_spike_times.pkl"
    ),
    Path("/Users/gabriel/Downloads/Organized/Code/20240821_121447_ks4_spike_times.pkl"),
]
OUT_PATH = Path("/Users/gabriel/lib/ephys/figures/grb006_double_peak_examples.pdf")

PETH_KWARGS = dict(
    pre_seconds=0.1,
    post_seconds=0.15,
    binwidth_ms=10,
    t_rise=None,
    t_decay=None,
)
SELECTIVITY_KWARGS = dict(
    base_window=(-0.04, 0.0),
    resp_window=(0.06, 0.10),
    test="wilcoxon",
    correction="bonferroni",
    alpha=0.05,
)
PEAK_KWARGS = dict(
    search_window=(0.0, 0.15),
    baseline_window=(-0.04, 0.0),
    min_prominence_frac=0.25,
    min_distance_ms=20.0,
    binwidth_ms=10.0,
)

MIN_PEAK_HEIGHT_ABS = 20.0
N_PANELS = 6


def resolve_spike_times_path() -> Path:
    for path in SPIKE_TIMES_PATHS:
        if path.exists():
            return path
    searched = "\n".join(str(path) for path in SPIKE_TIMES_PATHS)
    raise FileNotFoundError(f"Could not find KS4 spike-time export in:\n{searched}")


def load_local_spike_times(spike_times_path: Path, sampling_rate: float = 30000.0):
    spike_df = pd.read_pickle(spike_times_path)
    unit_ids = spike_df["unit_id"].astype(int).tolist()
    spike_times = [
        np.asarray(times, dtype=float) / sampling_rate
        for times in spike_df["spike_times"].tolist()
    ]
    return unit_ids, spike_times


def baseline_mean(peth_trials, bin_centers):
    mask = (bin_centers >= -0.04) & (bin_centers < 0.0)
    return peth_trials.mean(axis=0)[mask].mean()


def plot_trace(ax, bin_centers, peth_trials, color):
    mean = peth_trials.mean(axis=0)
    sem = peth_trials.std(axis=0) / np.sqrt(peth_trials.shape[0])
    ax.plot(bin_centers, mean, color=color, linewidth=1.5)
    ax.fill_between(bin_centers, mean - sem, mean + sem, alpha=0.25, color=color)


def mark_peaks(ax, peak_row, color):
    for pt, ph in zip(peak_row["peak_times"], peak_row["peak_heights"]):
        ax.plot(pt, ph, "v", color=color, markersize=7, zorder=5)


def collect_double_peak_rows():
    trial_ts = pd.read_pickle(TRIAL_TS_PATH).reset_index(drop=True)
    first_stim = trial_ts["first_stim_ts"].to_numpy(dtype=float)
    first_stim = first_stim[np.isfinite(first_stim)]

    spike_times_path = resolve_spike_times_path()
    unit_ids, spike_times = load_local_spike_times(spike_times_path)

    peth, bin_edges, bin_centers = compute_population_peth(
        spike_times_per_unit=spike_times,
        alignment_times=first_stim,
        **PETH_KWARGS,
    )
    _, masks = compute_unit_selectivity(
        peth, bin_edges, unit_ids=unit_ids, **SELECTIVITY_KWARGS
    )

    exc_idx = np.where(masks["excited"])[0]
    exc_ids = [unit_ids[i] for i in exc_idx]
    exc_peth = peth[exc_idx]
    peaks_df = classify_peak_count(
        exc_peth, bin_centers, unit_ids=exc_ids, **PEAK_KWARGS
    )

    rows = []
    for _, peak_row in peaks_df.loc[peaks_df["n_peaks"] == 2].iterrows():
        uid = int(peak_row["unit"])
        i = exc_ids.index(uid)
        base = baseline_mean(exc_peth[i], bin_centers)
        heights_above = [float(h - base) for h in peak_row["peak_heights"]]
        if min(heights_above) < MIN_PEAK_HEIGHT_ABS:
            continue
        rows.append(
            dict(
                uid=uid,
                peth=exc_peth[i],
                n_trials=len(first_stim),
                peaks_df_row=peak_row,
                bin_centers=bin_centers,
                baseline=base,
                min_above=min(heights_above),
                max_above=max(heights_above),
                peak_times=peak_row["peak_times"],
            )
        )

    rows.sort(key=lambda row: (row["min_above"], row["max_above"]), reverse=True)
    return rows, len(unit_ids), len(exc_ids), len(first_stim), spike_times_path


def make_figure(rows):
    ncols = 3
    nrows = int(np.ceil(len(rows[:N_PANELS]) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10.5, 7), sharex=True, sharey=False)
    axes = np.atleast_1d(axes).ravel()

    for ax, row in zip(axes, rows[:N_PANELS]):
        bc = row["bin_centers"]
        plot_trace(ax, bc, row["peth"], "tab:blue")
        mark_peaks(ax, row["peaks_df_row"], "tab:blue")
        ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
        peak_ms = ", ".join(f"{int(round(1000 * t))}" for t in row["peak_times"])
        ax.set_title(
            f"GRB006 unit {row['uid']}\npeaks at {peak_ms} ms",
            fontsize=8,
        )
        ax.set_xlabel("Time from first stim onset (s)", fontsize=8)
        ax.set_ylabel("sp/s", fontsize=8)
        ax.tick_params(labelsize=7)

    for ax in axes[len(rows[:N_PANELS]) :]:
        ax.axis("off")

    fig.suptitle(
        "GRB006 double-peak V1 units aligned to first stim onset\n"
        "Unsmoothened 10 ms bins; triangles mark detected peaks",
        fontsize=10,
        y=0.98,
    )
    fig.tight_layout()
    return fig


def main():
    rows, n_units, n_excited, n_trials, spike_times_path = collect_double_peak_rows()
    if not rows:
        raise RuntimeError("No GRB006 double-peak units passed the filters.")

    print(f"Session: {SESSION}")
    print(f"Spike times: {spike_times_path}")
    print(f"Units loaded: {n_units}")
    print(f"First-stim events: {n_trials}")
    print(f"Excited units: {n_excited}")
    print(f"Double-peak units: {len(rows)}")
    print("\nRanked candidates:")
    for rank, row in enumerate(rows, start=1):
        peak_ms = [int(round(1000 * t)) for t in row["peak_times"]]
        print(
            f"  {rank}. unit {row['uid']}  peaks={peak_ms} ms  "
            f"min_above={row['min_above']:.1f} sp/s  "
            f"max_above={row['max_above']:.1f} sp/s"
        )

    fig = make_figure(rows)
    with PdfPages(OUT_PATH) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    print(f"\nFigure saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
