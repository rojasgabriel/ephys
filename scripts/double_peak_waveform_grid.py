"""Compact 2×3 scatter grid: waveform metrics of double-peak units.

Layout
------
         GRB006 20240821  |  GRB058 20260312  |  GRB058 20260319
Row 0:   FR vs spike_dur  |  FR vs spike_dur  |  FR vs spike_dur
Row 1:   depth vs spike_dur | depth vs spike_dur | depth vs spike_dur

All good units shown (not just excited). Double-peak units in orange,
all others in blue. FS/RS boundary line at 0.4 ms.

GRB006 event loading uses local trial_ts.pkl (DB events not available).
GRB058 sessions are fully DB-backed.

Output
------
    figures/double_peak/waveform_grid.pdf
    figures/double_peak/waveform_grid_pooled.pdf
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from labdata.schema import EphysRecording, SpikeSorting, UnitCount, UnitMetrics
from matplotlib.backends.backend_pdf import PdfPages

from ephys.src.utils.utils_IO import fetch_session_events
from ephys.src.utils.utils_analysis import (
    classify_peak_count,
    compute_population_peth,
    compute_unit_selectivity,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SESSIONS = [
    ("GRB006", "20240821_121447"),
    ("GRB058", "20260224_152424"),
]

# GRB006 local paths (DB spike times are stale; use KS4 local export instead)
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

OUT_PATH = Path("/Users/gabriel/lib/ephys/figures/double_peak/waveform_grid.pdf")
OUT_PATH_POOLED = Path(
    "/Users/gabriel/lib/ephys/figures/double_peak/waveform_grid_pooled.pdf"
)
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

UNIT_CRITERIA_ID = 1

PETH_KWARGS = dict(
    pre_seconds=0.1,
    post_seconds=0.15,
    binwidth_ms=10,
    t_rise=None,
    t_decay=None,
)
PEAK_KWARGS = dict(
    search_window=(0.0, 0.15),
    baseline_window=(-0.04, 0.0),
    min_prominence_frac=0.25,
    min_distance_ms=20.0,
    binwidth_ms=10.0,
)
SELECTIVITY_KWARGS = dict(
    base_window=(-0.04, 0.0),
    resp_window=(0.02, 0.10),
    test="wilcoxon",
    correction="fdr_bh",
    alpha=0.05,
)
NARROW_BROAD_MS = 0.4

COL_OTHER = "#4C72B0"
COL_DOUBLE = "#DD8452"

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _fetch_unit_table(subject: str, session: str) -> pd.DataFrame:
    sess_q = (
        SpikeSorting() & f'subject_name = "{subject}"' & f'session_name = "{session}"'
    ).proj()

    good_ids = (
        sess_q
        * (UnitCount.Unit & f"unit_criteria_id = {UNIT_CRITERIA_ID}" & "passes = 1")
    ).fetch("subject_name", "session_name", "unit_id", as_dict=True)

    df = pd.DataFrame(
        ((SpikeSorting.Unit & good_ids) * UnitMetrics).fetch(
            "unit_id",
            "spike_times",
            "depth",
            "spike_duration",
            "fw3m",
            "spike_amplitude",
            "firing_rate",
            as_dict=True,
        )
    )

    srate = float((EphysRecording.ProbeSetting() & sess_q).fetch("sampling_rate")[0])
    if subject == "GRB006":
        spk_map = _load_grb006_spike_times_map()
        df["spike_times_s"] = df["unit_id"].apply(
            lambda uid: spk_map.get(int(uid), np.array([]))
        )
    else:
        df["spike_times_s"] = df["spike_times"].apply(
            lambda st: np.asarray(st, dtype=float) / srate
        )

    med = df["spike_duration"].dropna()
    if len(med):
        df["spike_duration_ms"] = (
            df["spike_duration"] / srate * 1000.0
            if med.median() > 100
            else df["spike_duration"]
        )
    else:
        df["spike_duration_ms"] = np.nan

    return df.sort_values("depth", ascending=True).reset_index(drop=True)


def _load_grb006_spike_times_map(sampling_rate: float = 30000.0) -> dict:
    """Return {unit_id: spike_times_s} from the local KS4 pkl."""
    for p in GRB006_SPIKE_TIMES_PATHS:
        if p.exists():
            spk = pd.read_pickle(p)
            return {
                int(row["unit_id"]): np.asarray(row["spike_times"], dtype=float)
                / sampling_rate
                for _, row in spk.iterrows()
            }
    raise FileNotFoundError(
        "GRB006 KS4 spike-times pkl not found in:\n"
        + "\n".join(str(p) for p in GRB006_SPIKE_TIMES_PATHS)
    )


def _load_grb006_first_stim() -> np.ndarray:
    trial_ts = pd.read_pickle(GRB006_TRIAL_TS_PATH).reset_index(drop=True)
    first_stim = trial_ts["first_stim_ts"].to_numpy(dtype=float)
    return first_stim[np.isfinite(first_stim)]


def _get_first_stim(subject: str, session: str) -> np.ndarray:
    if subject == "GRB006":
        return _load_grb006_first_stim()
    align_ev = fetch_session_events(subject, session)
    return align_ev["first_stim_ev_15ms"]


def _classify_double_peak(df: pd.DataFrame, first_stim: np.ndarray) -> pd.DataFrame:
    """Classify double-peak units using FDR selectivity + prominence filter.

    Uses FDR correction with resp_window=(0.02, 0.10) to catch early-peaking
    units (first peak before 60 ms) that Bonferroni + narrow window misses.
    No absolute height floor — the selectivity gate already ensures the unit
    has a statistically significant response.
    """
    unit_ids = df["unit_id"].tolist()
    spike_times = df["spike_times_s"].tolist()

    if len(first_stim) == 0:
        df["is_double"] = False
        return df

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

    double_ids = set(int(u) for u in peaks_df.loc[peaks_df["n_peaks"] == 2, "unit"])

    df["is_double"] = df["unit_id"].isin(double_ids)
    if double_ids:
        print(f"    double-peak unit_ids: {sorted(double_ids)}")
    return df


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def make_grid(session_data):
    """session_data: list of dicts with keys subject, session, df."""
    ncols = len(session_data)
    fig, axes = plt.subplots(
        2,
        ncols,
        figsize=(4.5 * ncols, 8),
        constrained_layout=True,
    )

    for col, sd in enumerate(session_data):
        df = sd["df"]
        subj, sess = sd["subject"], sd["session"]

        other = df[~df["is_double"]]
        double = df[df["is_double"]]

        col_title = f"{subj}  {sess[:8]}\nn={len(df)}  dp={int(df['is_double'].sum())}"

        # --- Row 0: firing rate vs spike duration ---
        ax = axes[0, col]
        ax.scatter(
            other["spike_duration_ms"],
            other["firing_rate"],
            s=14,
            alpha=0.40,
            color=COL_OTHER,
            rasterized=True,
            label=f"other (n={len(other)})",
        )
        ax.scatter(
            double["spike_duration_ms"],
            double["firing_rate"],
            s=28,
            alpha=0.90,
            color=COL_DOUBLE,
            zorder=3,
            edgecolors="k",
            linewidths=0.3,
            label=f"double-peak (n={len(double)})",
        )
        ax.axvline(NARROW_BROAD_MS, color="k", lw=0.7, ls="--", alpha=0.6)
        ax.set_xlabel("Spike duration (ms)", fontsize=9)
        ax.set_ylabel("Firing rate (sp/s)", fontsize=9)
        ax.set_title(col_title, fontsize=9)
        ax.legend(frameon=False, fontsize=8, loc="upper right")
        ax.tick_params(labelsize=8)

        # --- Row 1: depth vs spike duration ---
        ax = axes[1, col]
        ax.scatter(
            other["spike_duration_ms"],
            other["depth"],
            s=14,
            alpha=0.40,
            color=COL_OTHER,
            rasterized=True,
            label=f"other (n={len(other)})",
        )
        ax.scatter(
            double["spike_duration_ms"],
            double["depth"],
            s=28,
            alpha=0.90,
            color=COL_DOUBLE,
            zorder=3,
            edgecolors="k",
            linewidths=0.3,
            label=f"double-peak (n={len(double)})",
        )
        ax.axvline(NARROW_BROAD_MS, color="k", lw=0.7, ls="--", alpha=0.6)
        ax.set_xlabel("Spike duration (ms)", fontsize=9)
        ax.set_ylabel("Depth (µm)", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.legend(frameon=False, fontsize=8, loc="upper right")

    # Row labels
    axes[0, 0].annotate(
        "firing rate\nvs waveform width",
        xy=(-0.35, 0.5),
        xycoords="axes fraction",
        ha="center",
        va="center",
        fontsize=9,
        rotation=90,
    )
    axes[1, 0].annotate(
        "depth\nvs waveform width",
        xy=(-0.35, 0.5),
        xycoords="axes fraction",
        ha="center",
        va="center",
        fontsize=9,
        rotation=90,
    )

    fig.suptitle(
        "Double-peak unit waveform profile  ·  all good units shown  ·  "
        f"FS/RS boundary = {NARROW_BROAD_MS} ms",
        fontsize=10,
    )
    return fig


# ---------------------------------------------------------------------------
# Pooled figure
# ---------------------------------------------------------------------------


def make_pooled(session_data):
    """1×2 figure pooling all sessions; same two scatter types."""
    df = pd.concat([sd["df"] for sd in session_data], ignore_index=True)
    other = df[~df["is_double"]]
    double = df[df["is_double"]]

    subjects = ", ".join(dict.fromkeys(sd["subject"] for sd in session_data))
    n_sessions = len(session_data)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), constrained_layout=True)

    scatter_kw_other = dict(s=14, alpha=0.35, color=COL_OTHER, rasterized=True)
    scatter_kw_double = dict(
        s=28, alpha=0.90, color=COL_DOUBLE, zorder=3, edgecolors="k", linewidths=0.3
    )

    # Panel 0: FR vs spike duration
    ax = axes[0]
    ax.scatter(
        other["spike_duration_ms"],
        other["firing_rate"],
        **scatter_kw_other,
        label=f"other (n={len(other)})",
    )
    ax.scatter(
        double["spike_duration_ms"],
        double["firing_rate"],
        **scatter_kw_double,
        label=f"double-peak (n={len(double)})",
    )
    ax.axvline(NARROW_BROAD_MS, color="k", lw=0.7, ls="--", alpha=0.6)
    ax.set_xlabel("Spike duration (ms)", fontsize=10)
    ax.set_ylabel("Firing rate (sp/s)", fontsize=10)
    ax.set_title("Firing rate vs waveform width", fontsize=10)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.tick_params(labelsize=9)

    # Panel 1: depth vs spike duration
    ax = axes[1]
    ax.scatter(
        other["spike_duration_ms"],
        other["depth"],
        **scatter_kw_other,
        label=f"other (n={len(other)})",
    )
    ax.scatter(
        double["spike_duration_ms"],
        double["depth"],
        **scatter_kw_double,
        label=f"double-peak (n={len(double)})",
    )
    ax.axvline(NARROW_BROAD_MS, color="k", lw=0.7, ls="--", alpha=0.6)
    ax.set_xlabel("Spike duration (ms)", fontsize=10)
    ax.set_ylabel("Depth (µm)", fontsize=10)
    ax.set_title("Depth vs waveform width", fontsize=10)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.tick_params(labelsize=9)

    fig.suptitle(
        f"Pooled  ·  {subjects}  ·  {n_sessions} sessions  ·  "
        f"all good units  ·  FS/RS boundary = {NARROW_BROAD_MS} ms",
        fontsize=10,
    )
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    session_data = []

    for subject, session in SESSIONS:
        print(f"\n{subject} / {session}")
        try:
            df = _fetch_unit_table(subject, session)
        except Exception as e:
            print(f"  ✗ unit table: {e}")
            continue
        print(f"  units: {len(df)}")

        try:
            first_stim = _get_first_stim(subject, session)
        except Exception as e:
            print(f"  ✗ events: {e}")
            continue
        print(f"  first_stim events: {len(first_stim)}")

        df = _classify_double_peak(df, first_stim)
        n_dp = int(df["is_double"].sum())
        print(f"  double-peak: {n_dp}")

        session_data.append(dict(subject=subject, session=session, df=df))

    if not session_data:
        raise RuntimeError("No sessions loaded.")

    fig = make_grid(session_data)
    with PdfPages(OUT_PATH) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved → {OUT_PATH}")

    fig_pool = make_pooled(session_data)
    with PdfPages(OUT_PATH_POOLED) as pdf:
        pdf.savefig(fig_pool, bbox_inches="tight")
    plt.close(fig_pool)
    print(f"Saved → {OUT_PATH_POOLED}")


if __name__ == "__main__":
    main()
