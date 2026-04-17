"""Build the single-page double-peak figure for the Dario email.

Story:
1. Anne and Gabriel first noticed the double-peak response shape in GRB006.
2. The same motif reappeared in GRB058, but in fewer units.
3. A simple onset+offset explanation motivated the GRB058 pulse-width test.
4. March 12 and March 19 are shown separately because their recorded 30 ms
   fractions differed and the outcomes should not be pooled.
"""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ephys.src.utils.utils_IO import (
    fetch_good_units,
    fetch_session_events,
    fetch_trial_metadata,
)
from ephys.src.utils.utils_analysis import (
    classify_peak_count,
    compute_population_peth,
    compute_unit_selectivity,
)


GRB006_SESSION = "20240821_121447"
GRB006_TRIAL_TS_PATH = Path(
    "/Users/gabriel/data/GRB006/20240821_121447/pre_processed/trial_ts.pkl"
)
GRB006_SPIKE_TIMES_PATHS = [
    Path(
        "/Users/gabriel/data/GRB006/20240821_121447/pre_processed/"
        "20240821_121447_ks4_spike_times.pkl"
    ),
    Path("/Users/gabriel/Downloads/Organized/Code/20240821_121447_ks4_spike_times.pkl"),
]
GRB006_SHOW_UNITS = [579, 694, 217]

GRB058_SUBJECT = "GRB058"
SESSION_ORDER = ["20260312_134952", "20260319_131303"]
SESSION_LABELS = {
    "20260312_134952": "GRB058  2026-03-12",
    "20260319_131303": "GRB058  2026-03-19",
}
INTENDED_4HZ_30MS_FRAC = {
    "20260312_134952": 0.25,
    "20260319_131303": 0.50,
}
SESSION_SHOW_UNITS = {
    "20260312_134952": [410, 651],
    "20260319_131303": [515],
}

OUT_PATH = Path("/Users/gabriel/lib/ephys/figures/dario_double_peak_story.pdf")

PETH_KWARGS = dict(
    pre_seconds=0.1,
    post_seconds=0.15,
    binwidth_ms=10,
    t_rise=None,
    t_decay=None,
)
PETH_SCALE_BACK = PETH_KWARGS["binwidth_ms"] / 1000.0
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


def resolve_grb006_spike_times_path() -> Path:
    for path in GRB006_SPIKE_TIMES_PATHS:
        if path.exists():
            return path
    searched = "\n".join(str(path) for path in GRB006_SPIKE_TIMES_PATHS)
    raise FileNotFoundError(
        f"Could not find GRB006 KS4 spike-time export in:\n{searched}"
    )


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


def plot_trace(ax, bin_centers, peth_trials, color, label):
    mean = peth_trials.mean(axis=0)
    sem = peth_trials.std(axis=0) / np.sqrt(peth_trials.shape[0])
    ax.plot(bin_centers, mean, color=color, linewidth=1.5, label=label)
    ax.fill_between(bin_centers, mean - sem, mean + sem, alpha=0.25, color=color)


def mark_peaks(ax, peak_row, color):
    for pt, ph in zip(peak_row["peak_times"], peak_row["peak_heights"]):
        ax.plot(pt, ph, "v", color=color, markersize=7, zorder=5)


def collect_grb006():
    spike_times_path = resolve_grb006_spike_times_path()
    trial_ts = pd.read_pickle(GRB006_TRIAL_TS_PATH).reset_index(drop=True)
    first_stim = trial_ts["first_stim_ts"].to_numpy(dtype=float)
    first_stim = first_stim[np.isfinite(first_stim)]

    unit_ids, spike_times = load_local_spike_times(spike_times_path)
    peth, bin_edges, bin_centers = compute_population_peth(
        spike_times_per_unit=spike_times,
        alignment_times=first_stim,
        **PETH_KWARGS,
    )
    peth *= PETH_SCALE_BACK
    _, masks = compute_unit_selectivity(
        peth, bin_edges, unit_ids=unit_ids, **SELECTIVITY_KWARGS
    )
    exc_idx = np.where(masks["excited"])[0]
    exc_ids = [unit_ids[i] for i in exc_idx]
    exc_peth = peth[exc_idx]
    peaks_df = classify_peak_count(
        exc_peth, bin_centers, unit_ids=exc_ids, **PEAK_KWARGS
    )

    rows = {}
    double_ids = []
    for _, peak_row in peaks_df.loc[peaks_df["n_peaks"] == 2].iterrows():
        uid = int(peak_row["unit"])
        double_ids.append(uid)
        rows[uid] = dict(
            uid=uid,
            peth=exc_peth[exc_ids.index(uid)],
            bin_centers=bin_centers,
            peaks_df_row=peak_row,
            peak_times_ms=[int(round(1000 * t)) for t in peak_row["peak_times"]],
        )

    return {
        "session": GRB006_SESSION,
        "n_units": len(unit_ids),
        "n_excited": len(exc_ids),
        "n_double": len(double_ids),
        "rows": rows,
        "spike_times_path": spike_times_path,
    }


def classify_first_stim_widths_by_trial(align_ev, trial_df):
    first15 = pd.DataFrame(
        {
            "stim_onset": np.asarray(align_ev["first_stim_ev_15ms"], dtype=float),
            "width_ms": 15,
        }
    )
    first30 = pd.DataFrame(
        {
            "stim_onset": np.asarray(align_ev["first_stim_ev_30ms"], dtype=float),
            "width_ms": 30,
        }
    )
    first = (
        pd.concat([first15, first30], ignore_index=True)
        .sort_values("stim_onset")
        .reset_index(drop=True)
    )

    trial_starts = trial_df["trial_start_ts"].to_numpy(dtype=float)
    trial_idx = (
        np.searchsorted(
            trial_starts, first["stim_onset"].to_numpy(dtype=float), side="right"
        )
        - 1
    )
    valid = (trial_idx >= 0) & (trial_idx < len(trial_df))
    first = first.loc[valid].copy()
    first["trial_idx"] = trial_idx[valid]
    first = first.drop_duplicates("trial_idx", keep="first")

    merged = trial_df.reset_index(drop=True).join(
        first.set_index("trial_idx")[["width_ms"]], how="left"
    )
    merged["has_classified_first_stim"] = merged["width_ms"].notna()
    return merged


def collect_grb058_session(session):
    st_per_unit = fetch_good_units(GRB058_SUBJECT, session)
    align_ev = fetch_session_events(GRB058_SUBJECT, session)
    trial_df = fetch_trial_metadata(GRB058_SUBJECT, session, align_ev)
    if trial_df is None:
        raise RuntimeError(
            f"Could not load trial metadata for {GRB058_SUBJECT} {session}"
        )

    unit_ids = list(st_per_unit.keys())
    spike_times = list(st_per_unit.values())
    peth_15, bin_edges, bin_centers = compute_population_peth(
        spike_times_per_unit=spike_times,
        alignment_times=align_ev["first_stim_ev_15ms"],
        **PETH_KWARGS,
    )
    peth_15 *= PETH_SCALE_BACK
    _, masks = compute_unit_selectivity(
        peth_15, bin_edges, unit_ids=unit_ids, **SELECTIVITY_KWARGS
    )
    exc_idx = np.where(masks["excited"])[0]
    exc_ids = [unit_ids[i] for i in exc_idx]
    exc_peth = peth_15[exc_idx]
    peaks_df_15 = classify_peak_count(
        exc_peth, bin_centers, unit_ids=exc_ids, **PEAK_KWARGS
    )

    double_ids = []
    for _, peak_row in peaks_df_15.loc[peaks_df_15["n_peaks"] == 2].iterrows():
        uid = int(peak_row["unit"])
        i = exc_ids.index(uid)
        base = baseline_mean(exc_peth[i], bin_centers)
        heights_above = [float(h - base) for h in peak_row["peak_heights"]]
        if min(heights_above) >= MIN_PEAK_HEIGHT_ABS:
            double_ids.append(uid)

    rows = {}
    if double_ids:
        peth_30, _, _ = compute_population_peth(
            spike_times_per_unit=[
                spike_times[unit_ids.index(uid)] for uid in double_ids
            ],
            alignment_times=align_ev["first_stim_ev_30ms"],
            **PETH_KWARGS,
        )
        peth_30 *= PETH_SCALE_BACK
        peaks_df_30 = classify_peak_count(
            peth_30, bin_centers, unit_ids=double_ids, **PEAK_KWARGS
        )

        for j, uid in enumerate(double_ids):
            rows[uid] = dict(
                uid=uid,
                peth_15=exc_peth[exc_ids.index(uid)],
                peth_30=peth_30[j],
                bin_centers=bin_centers,
                peak_row_15=peaks_df_15[peaks_df_15["unit"] == uid].iloc[0],
                peak_row_30=peaks_df_30[peaks_df_30["unit"] == uid].iloc[0],
            )

    classified_trial_df = classify_first_stim_widths_by_trial(align_ev, trial_df)
    rate4 = classified_trial_df[
        (classified_trial_df["has_classified_first_stim"])
        & (classified_trial_df["stim_rate_vision"] == 4)
    ]
    n_4hz_total = int((trial_df["stim_rate_vision"] == 4).sum())
    n_4hz_classified = len(rate4)
    n_4hz_15 = int((rate4["width_ms"] == 15).sum())
    n_4hz_30 = int((rate4["width_ms"] == 30).sum())
    frac_30 = n_4hz_30 / n_4hz_classified if n_4hz_classified else np.nan

    return {
        "label": SESSION_LABELS[session],
        "session": session,
        "n_units": len(unit_ids),
        "n_excited": len(exc_ids),
        "n_double": len(double_ids),
        "rows": rows,
        "n_tr_15": len(align_ev["first_stim_ev_15ms"]),
        "n_tr_30": len(align_ev["first_stim_ev_30ms"]),
        "n_4hz_total": n_4hz_total,
        "n_4hz_classified": n_4hz_classified,
        "n_4hz_15": n_4hz_15,
        "n_4hz_30": n_4hz_30,
        "frac_30": frac_30,
        "intended_frac_30": INTENDED_4HZ_30MS_FRAC[session],
    }


def plot_session_mix(ax, session_data):
    counts = [session_data["n_4hz_15"], session_data["n_4hz_30"]]
    ax.bar(["15 ms", "30 ms"], counts, color=["tab:blue", "tab:orange"], width=0.6)
    ax.set_title(
        f"{session_data['label']}\n4 Hz first-stim trial mix",
        fontsize=8,
    )
    ax.set_ylabel("Classified trials", fontsize=8)
    ax.tick_params(labelsize=7)


def plot_summary_table(ax, session_0312, session_0319):
    ax.axis("off")
    summary = (
        "GRB058 session summary\n\n"
        f"2026-03-12\n"
        f"  double-peak: {session_0312['n_double']}/{session_0312['n_units']} good units\n"
        f"  4 Hz long pulses: {session_0312['n_4hz_30']}/{session_0312['n_4hz_classified']} "
        f"({100 * session_0312['frac_30']:.1f}%)\n"
        f"  classified first pulses: {session_0312['n_tr_15']} 15 ms, {session_0312['n_tr_30']} 30 ms\n\n"
        f"2026-03-19\n"
        f"  double-peak: {session_0319['n_double']}/{session_0319['n_units']} good units\n"
        f"  4 Hz long pulses: {session_0319['n_4hz_30']}/{session_0319['n_4hz_classified']} "
        f"({100 * session_0319['frac_30']:.1f}%)\n"
        f"  classified first pulses: {session_0319['n_tr_15']} 15 ms, {session_0319['n_tr_30']} 30 ms"
    )
    ax.text(
        0.02,
        0.98,
        summary,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        linespacing=1.4,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "#f7f7f7", "alpha": 0.95},
    )


def make_figure(grb006, session_0312, session_0319):
    fig = plt.figure(figsize=(12.5, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.9, wspace=0.45)

    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    for ax, uid in zip(axes, GRB006_SHOW_UNITS):
        row = grb006["rows"][uid]
        plot_trace(ax, row["bin_centers"], row["peth"], "tab:blue", "15 ms")
        mark_peaks(ax, row["peaks_df_row"], "tab:blue")
        ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
        peak_ms = ", ".join(str(x) for x in row["peak_times_ms"])
        ax.set_title(f"GRB006 unit {uid}\npeaks at {peak_ms} ms", fontsize=8)
        ax.set_xlabel("Time from first stim onset (s)", fontsize=8)
        ax.set_ylabel("sp/s", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_ylim(bottom=0)

    session_0312_axes = [fig.add_subplot(gs[1, i]) for i in range(3)]
    for ax, uid in zip(session_0312_axes[:2], SESSION_SHOW_UNITS["20260312_134952"]):
        row = session_0312["rows"][uid]
        bc = row["bin_centers"]
        plot_trace(
            ax, bc, row["peth_15"], "tab:blue", f"15 ms (n={session_0312['n_tr_15']})"
        )
        mark_peaks(ax, row["peak_row_15"], "tab:blue")
        plot_trace(
            ax, bc, row["peth_30"], "tab:orange", f"30 ms (n={session_0312['n_tr_30']})"
        )
        mark_peaks(ax, row["peak_row_30"], "tab:orange")
        ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_title(f"{session_0312['label']}\nunit {uid}", fontsize=8)
        ax.set_xlabel("Time from first stim onset (s)", fontsize=8)
        ax.set_ylabel("sp/s", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=7, frameon=False, loc="upper left")
    plot_session_mix(session_0312_axes[2], session_0312)

    session_0319_axes = [fig.add_subplot(gs[2, i]) for i in range(3)]
    row = session_0319["rows"][SESSION_SHOW_UNITS["20260319_131303"][0]]
    bc = row["bin_centers"]
    plot_trace(
        session_0319_axes[0],
        bc,
        row["peth_15"],
        "tab:blue",
        f"15 ms (n={session_0319['n_tr_15']})",
    )
    mark_peaks(session_0319_axes[0], row["peak_row_15"], "tab:blue")
    plot_trace(
        session_0319_axes[0],
        bc,
        row["peth_30"],
        "tab:orange",
        f"30 ms (n={session_0319['n_tr_30']})",
    )
    mark_peaks(session_0319_axes[0], row["peak_row_30"], "tab:orange")
    session_0319_axes[0].axvline(0, color="gray", linestyle="--", linewidth=0.8)
    session_0319_axes[0].set_title(
        f"{session_0319['label']}\nunit {SESSION_SHOW_UNITS['20260319_131303'][0]}",
        fontsize=8,
    )
    session_0319_axes[0].set_xlabel("Time from first stim onset (s)", fontsize=8)
    session_0319_axes[0].set_ylabel("sp/s", fontsize=8)
    session_0319_axes[0].tick_params(labelsize=7)
    session_0319_axes[0].set_ylim(bottom=0)
    session_0319_axes[0].legend(fontsize=7, frameon=False, loc="upper left")
    plot_session_mix(session_0319_axes[1], session_0319)
    plot_summary_table(session_0319_axes[2], session_0312, session_0319)

    fig.text(
        0.06,
        0.94,
        (
            "A. Double peaks were first noticed in GRB006 "
            f"({grb006['n_double']}/{grb006['n_units']} good units)"
        ),
        fontsize=10,
        fontweight="bold",
    )
    fig.text(
        0.06,
        0.63,
        (
            "B. GRB058 2026-03-12: long-pulse manipulation session "
            f"({session_0312['n_double']}/{session_0312['n_units']} good units)"
        ),
        fontsize=10,
        fontweight="bold",
    )
    fig.text(
        0.06,
        0.315,
        (
            "C. GRB058 2026-03-19: follow-up session shown separately "
            f"({session_0319['n_double']}/{session_0319['n_units']} good units)"
        ),
        fontsize=10,
        fontweight="bold",
    )

    fig.suptitle(
        "Double-peaked flash responses in mouse VISp\n"
        "Unsmoothed 10 ms bins; triangles mark detected peaks",
        fontsize=12,
        y=0.985,
    )
    fig.subplots_adjust(top=0.88, bottom=0.08, left=0.07, right=0.98)
    return fig


def main():
    grb006 = collect_grb006()
    session_0312 = collect_grb058_session("20260312_134952")
    session_0319 = collect_grb058_session("20260319_131303")
    fig = make_figure(grb006, session_0312, session_0319)

    with PdfPages(OUT_PATH) as pdf:
        pdf.savefig(fig, bbox_inches="tight")

    print(f"Figure saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
