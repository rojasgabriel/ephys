"""Double-peak PSTH figure for the Dario email.

Produces figures/double_peak_dario.pdf — a single landscape page.

Layout: two rows, three columns.
  • Top row — single-peak reference examples (GRB058, 15 ms only):
    top-3 excited single-peak units by response amplitude above baseline.
  • Bottom row — double-peak units (GRB058 only, 15 ms + 30 ms overlaid):
    units 410 and 651 (3/12), 515 (3/19).

Double-peak criterion: excited by Wilcoxon test, exactly two peaks detected
by classify_peak_count (min_prominence_frac=0.25, min_distance=20 ms), AND
both peaks ≥ 20 sp/s above pre-stimulus baseline.  This floor removes low-SNR
cases (e.g. GRB060 unit 248, whose peaks were only 11–13 sp/s above baseline
with the second peak larger than the first — not a typical on-response shape).

GRB059: no double-peak units detected (0/4 excited units qualify).
GRB060: unit 248 passes the two-peak shape test but fails the 20 sp/s floor.
Both are absent from the figure; mention in the email.
GRB006 (~11/150 double-peak units, different recording hardware): referenced
in the email text; the nidq event pipeline is not yet ported.

Anne's guidance (2026-04-02): keep it simple — show examples with and without
long pulses; describe the stimulus; ask whether this response shape is known.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

matplotlib.use("Agg")

from ephys.src.utils.utils_IO import fetch_good_units, fetch_session_events
from ephys.src.utils.utils_analysis import (
    compute_population_peth,
    compute_unit_selectivity,
    classify_peak_count,
)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
PETH_KWARGS = dict(
    pre_seconds=0.1,
    post_seconds=0.15,
    binwidth_ms=10,
    t_rise=None,
    t_decay=None,  # no smoothing — required to resolve peaks ~30 ms apart
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

# Post-hoc quality filter for double-peak classification.
# Both peaks must exceed this height above pre-stimulus baseline (sp/s).
MIN_PEAK_HEIGHT_ABS = 20.0

GRB058_SESSIONS = ["20260312_134952", "20260319_131303"]
OUT_PATH = "/Users/gabriel/lib/ephys/figures/double_peak_dario.pdf"


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def _baseline_mean(peth_trials, bin_centers):
    """Mean firing rate over (-0.04, 0.0) s window, trial-averaged."""
    mask = (bin_centers >= -0.04) & (bin_centers < 0.0)
    return peth_trials.mean(axis=0)[mask].mean()


def _load_session(subject, session):
    """Return (unit_ids, spike_times, align_ev, peth_15, bin_edges, bin_centers, masks)."""
    st_per_unit = fetch_good_units(subject, session)
    align_ev = fetch_session_events(subject, session)
    unit_ids = list(st_per_unit.keys())
    spike_times = list(st_per_unit.values())

    peth_15, bin_edges, bin_centers = compute_population_peth(
        spike_times_per_unit=spike_times,
        alignment_times=align_ev["first_stim_ev_15ms"],
        **PETH_KWARGS,
    )
    _, masks = compute_unit_selectivity(
        peth_15, bin_edges, unit_ids=unit_ids, **SELECTIVITY_KWARGS
    )
    return unit_ids, spike_times, align_ev, peth_15, bin_edges, bin_centers, masks


def plot_trace(ax, bin_centers, peth_trials, color, label):
    mean = peth_trials.mean(axis=0)
    sem = peth_trials.std(axis=0) / np.sqrt(peth_trials.shape[0])
    ax.plot(bin_centers, mean, color=color, linewidth=1.5, label=label)
    ax.fill_between(bin_centers, mean - sem, mean + sem, alpha=0.25, color=color)
    return mean


def mark_peaks(ax, peak_row, color):
    for pt, ph in zip(peak_row["peak_times"], peak_row["peak_heights"]):
        ax.plot(pt, ph, "v", color=color, markersize=7, zorder=5)


# ---------------------------------------------------------------------------
# Collect double-peak units (GRB058, both sessions)
# ---------------------------------------------------------------------------
dp_rows = []  # list of dicts

for session in GRB058_SESSIONS:
    unit_ids, spike_times, align_ev, peth_15, bin_edges, bin_centers, masks = (
        _load_session("GRB058", session)
    )
    n_tr_15 = len(align_ev["first_stim_ev_15ms"])
    n_tr_30 = len(align_ev["first_stim_ev_30ms"])

    exc_idx = np.where(masks["excited"])[0]
    exc_peth = peth_15[exc_idx]
    exc_ids = [unit_ids[i] for i in exc_idx]
    exc_spike_times = [spike_times[i] for i in exc_idx]

    peaks_df = classify_peak_count(
        exc_peth, bin_centers, unit_ids=exc_ids, **PEAK_KWARGS
    )
    candidate_ids = peaks_df.loc[peaks_df["n_peaks"] == 2, "unit"].tolist()

    # Apply minimum absolute peak height filter
    double_ids = []
    for uid in candidate_ids:
        i = exc_ids.index(uid)
        peak_row = peaks_df[peaks_df["unit"] == uid].iloc[0]
        base = _baseline_mean(exc_peth[i], bin_centers)
        heights_above = [h - base for h in peak_row["peak_heights"]]
        if min(heights_above) >= MIN_PEAK_HEIGHT_ABS:
            double_ids.append(uid)
        else:
            print(
                f"  Excluded {uid}: min peak height above baseline = "
                f"{min(heights_above):.1f} sp/s (< {MIN_PEAK_HEIGHT_ABS})"
            )

    print(
        f"\nGRB058/{session[:8]}  15ms_trials={n_tr_15}  30ms_trials={n_tr_30}"
        f"  double-peak={double_ids}"
    )

    if not double_ids:
        continue

    dp_idx = [exc_ids.index(uid) for uid in double_ids]
    dp_peth_15 = exc_peth[dp_idx]
    dp_spike_times = [exc_spike_times[i] for i in dp_idx]

    peth_30_all, _, _ = compute_population_peth(
        spike_times_per_unit=dp_spike_times,
        alignment_times=align_ev["first_stim_ev_30ms"],
        **PETH_KWARGS,
    )

    for j, uid in enumerate(double_ids):
        dp_rows.append(
            dict(
                session=session,
                uid=uid,
                peth_15=dp_peth_15[j],
                peth_30=peth_30_all[j],
                n_tr_15=n_tr_15,
                n_tr_30=n_tr_30,
                peaks_df_row=peaks_df[peaks_df["unit"] == uid].iloc[0],
                bin_centers=bin_centers,
            )
        )

# ---------------------------------------------------------------------------
# Collect single-peak reference examples (GRB058, best by amplitude)
# Pick top N_PANELS single-peak excited units sorted by max firing rate.
# ---------------------------------------------------------------------------
N_PANELS = len(dp_rows)  # match column count

sp_candidates = []
for session in GRB058_SESSIONS:
    unit_ids, spike_times, align_ev, peth_15, bin_edges, bin_centers, masks = (
        _load_session("GRB058", session)
    )
    n_tr_15 = len(align_ev["first_stim_ev_15ms"])

    exc_idx = np.where(masks["excited"])[0]
    exc_peth = peth_15[exc_idx]
    exc_ids = [unit_ids[i] for i in exc_idx]

    peaks_df = classify_peak_count(
        exc_peth, bin_centers, unit_ids=exc_ids, **PEAK_KWARGS
    )
    single_ids = peaks_df.loc[peaks_df["n_peaks"] == 1, "unit"].tolist()

    for uid in single_ids:
        i = exc_ids.index(uid)
        mean_psth = exc_peth[i].mean(axis=0)
        base = _baseline_mean(exc_peth[i], bin_centers)
        excursion = mean_psth.max() - base  # response amplitude above baseline
        sp_candidates.append(
            dict(
                session=session,
                uid=uid,
                peth_15=exc_peth[i],
                n_tr_15=n_tr_15,
                peaks_df_row=peaks_df[peaks_df["unit"] == uid].iloc[0],
                bin_centers=bin_centers,
                excursion=excursion,
            )
        )

sp_candidates.sort(key=lambda r: -r["excursion"])
sp_rows = sp_candidates[:N_PANELS]
print(
    f"\nSingle-peak examples selected: {[(r['uid'], r['session'][:8]) for r in sp_rows]}"
)

# ---------------------------------------------------------------------------
# Figure — 2 rows × N_PANELS columns
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, N_PANELS, figsize=(3.5 * N_PANELS, 7), sharey=False)

# ---- Top row: single-peak reference ----------------------------------------
for col, row in enumerate(sp_rows):
    ax = axes[0, col]
    bc = row["bin_centers"]
    plot_trace(ax, bc, row["peth_15"], "tab:gray", f"15 ms (n={row['n_tr_15']})")
    mark_peaks(ax, row["peaks_df_row"], color="dimgray")
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title(
        f"GRB058  unit {row['uid']}\n{row['session'][:8]}  (single peak)", fontsize=8
    )
    ax.set_ylabel("sp/s", fontsize=8)
    ax.tick_params(labelsize=7)
    if col == 0:
        ax.set_xlabel("Time from stim onset (s)", fontsize=8)

# ---- Bottom row: double-peak units -----------------------------------------
for col, row in enumerate(dp_rows):
    ax = axes[1, col]
    bc = row["bin_centers"]

    plot_trace(ax, bc, row["peth_15"], "tab:blue", f"15 ms (n={row['n_tr_15']})")
    mark_peaks(ax, row["peaks_df_row"], color="tab:blue")

    plot_trace(ax, bc, row["peth_30"], "tab:orange", f"30 ms (n={row['n_tr_30']})")
    peak_df_30 = classify_peak_count(
        row["peth_30"][np.newaxis, :, :],
        bc,
        unit_ids=[row["uid"]],
        **PEAK_KWARGS,
    )
    if not peak_df_30.empty:
        mark_peaks(ax, peak_df_30.iloc[0], color="tab:orange")

    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    ax.set_title(
        f"GRB058  unit {row['uid']}\n{row['session'][:8]}  (double peak)", fontsize=8
    )
    ax.set_ylabel("sp/s", fontsize=8)
    ax.set_xlabel("Time from stim onset (s)", fontsize=8)
    ax.tick_params(labelsize=7)

# Row labels on the left margin
axes[0, 0].annotate(
    "Single-peak\nexamples",
    xy=(-0.22, 0.5),
    xycoords="axes fraction",
    fontsize=8,
    ha="right",
    va="center",
    rotation=90,
    fontweight="bold",
)
axes[1, 0].annotate(
    "Double-peak\nunits",
    xy=(-0.22, 0.5),
    xycoords="axes fraction",
    fontsize=8,
    ha="right",
    va="center",
    rotation=90,
    fontweight="bold",
)

fig.suptitle(
    "Double-peaked V1 responses to LED flashes  —  15 ms vs 30 ms pulse width\n"
    "Triangles mark detected peaks  ▼",
    fontsize=10,
    y=1.02,
)
fig.tight_layout()

with PdfPages(OUT_PATH) as pdf:
    pdf.savefig(fig, bbox_inches="tight")

print(f"\nFigure saved: {OUT_PATH}")
