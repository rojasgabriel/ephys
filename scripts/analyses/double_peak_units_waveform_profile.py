"""Compact scatter grid: waveform profile of double-peak units.

Sessions: GRB006 20240821 + GRB058 20260312.

Layout
------
         GRB006 20240821     |  GRB058 20260312
Row 0:   FR vs spike_dur     |  FR vs spike_dur

All good units shown (not just excited). Double-peak units in orange,
all others in blue. FS/RS boundary line at 0.4 ms (visual reference only).

Classification uses canonical params from src/config/double_peak.py
(FDR selectivity + 5 sp/s height floor on both peaks).

GRB006 event loading uses local trial_ts.pkl (DB events not available).
GRB006 spike times use DB-backed good-unit rows.
GRB058 is fully DB-backed.

Output
------
    figures/double_peak/waveform_grid.pdf
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from labdata.schema import EphysRecording, SpikeSorting, UnitCount, UnitMetrics
from matplotlib.backends.backend_pdf import PdfPages

from ephys.src.config.double_peak import (
    PETH_KWARGS,
)
from ephys.src.utils.grb006_data import (
    fetch_grb006_db_spike_times,
    load_grb006_first_stim,
)
from ephys.src.utils.peak_classification import canonical_double_peak_rows
from ephys.src.utils.utils_IO import fetch_session_events
from ephys.src.utils.utils_analysis import compute_population_peth

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SESSIONS = [
    ("GRB006", "20240821_121447"),
    ("GRB058", "20260312_134952"),
]

OUT_PATH = Path("/Users/gabriel/lib/ephys/figures/double_peak/waveform_grid.pdf")
OUT_PATH_MONO = Path(
    "/Users/gabriel/lib/ephys/figures/double_peak/waveform_grid_nocolor.pdf"
)
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

UNIT_CRITERIA_ID = 1
NARROW_BROAD_MS = 0.4  # FS/RS boundary, visual reference only

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
            "firing_rate",
            as_dict=True,
        )
    )

    srate = float((EphysRecording.ProbeSetting() & sess_q).fetch("sampling_rate")[0])
    if subject == "GRB006":
        spk_map = _fetch_grb006_spike_times_map()
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


def _fetch_grb006_spike_times_map() -> dict:
    """Return {unit_id: spike_times_s} from the DB-backed good-unit rows."""
    unit_ids, spike_times = fetch_grb006_db_spike_times()
    return dict(zip(unit_ids, spike_times))


def _load_grb006_first_stim() -> np.ndarray:
    return load_grb006_first_stim()


def _get_first_stim(subject: str, session: str) -> np.ndarray:
    if subject == "GRB006":
        return _load_grb006_first_stim()
    align_ev = fetch_session_events(subject, session)
    return align_ev["first_stim_ev_15ms"]


def _classify_double_peak(df: pd.DataFrame, first_stim: np.ndarray) -> pd.DataFrame:
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
    double_peak_rows, _, _ = canonical_double_peak_rows(
        peth, bin_edges, bin_centers, unit_ids
    )
    double_ids = set(double_peak_rows["unit"].astype(int).tolist())

    df["is_double"] = df["unit_id"].isin(double_ids)
    if double_ids:
        print(f"    double-peak unit_ids: {sorted(double_ids)}")
    return df


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def make_grid(session_data, color_by_double: bool = True):
    """session_data: list of dicts with keys subject, session, df."""
    ncols = len(session_data)
    fig, axes = plt.subplots(
        1,
        ncols,
        figsize=(4.5 * ncols, 4.5),
        constrained_layout=True,
    )
    if ncols == 1:
        axes = [axes]

    for col, sd in enumerate(session_data):
        df = sd["df"]
        subj, sess = sd["subject"], sd["session"]

        col_title = f"{subj}  {sess[:8]}\nn={len(df)}  dp={int(df['is_double'].sum())}"

        ax = axes[col]
        if color_by_double:
            other = df[~df["is_double"]]
            double = df[df["is_double"]]
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
            ax.legend(frameon=False, fontsize=8, loc="upper right")
        else:
            ax.scatter(
                df["spike_duration_ms"],
                df["firing_rate"],
                s=16,
                alpha=0.45,
                color="0.25",
                rasterized=True,
            )
        ax.axvline(NARROW_BROAD_MS, color="k", lw=0.7, ls="--", alpha=0.6)
        ax.set_xlabel("Spike duration (ms)", fontsize=9)
        ax.set_ylabel("Firing rate (sp/s)", fontsize=9)
        ax.set_title(col_title, fontsize=9)
        ax.tick_params(labelsize=8)

    if color_by_double:
        fig.suptitle(
            "Double-peak unit waveform profile  ·  all good units shown  ·  "
            f"FS/RS boundary = {NARROW_BROAD_MS} ms",
            fontsize=10,
        )
    else:
        fig.suptitle(
            "Waveform profile of all good units  ·  no double-peak color split  ·  "
            f"FS/RS boundary = {NARROW_BROAD_MS} ms",
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
        pdf.savefig(fig, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"\nSaved → {OUT_PATH}")

    fig_mono = make_grid(session_data, color_by_double=False)
    with PdfPages(OUT_PATH_MONO) as pdf:
        pdf.savefig(fig_mono, bbox_inches="tight", dpi=300)
    plt.close(fig_mono)
    print(f"Saved → {OUT_PATH_MONO}")


if __name__ == "__main__":
    main()
