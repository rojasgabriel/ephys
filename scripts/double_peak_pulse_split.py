"""Double-peak PSTH figure for the Dario email.

Produces figures/double_peak_dario.pdf — a single landscape page suitable
for attaching to an email without requiring the recipient to scroll.

Layout: one row, one panel per double-peak unit.
  • GRB058 units 410, 651 (session 20260312), 515 (session 20260319):
    15 ms (blue) and 30 ms (orange) PSTHs overlaid so the reader can
    immediately see whether the second peak shifts with pulse duration.
  • GRB060 unit 248 (session 20260319): 15 ms only — this animal's
    sessions contained no 30 ms stimulus blocks.

GRB059 (sessions 20260225 and 20260319) had no double-peak units among
its excited units and is therefore not shown.

GRB006 (~11/150 double-peak units in a prior recording with a different
hardware setup) is referenced in the accompanying email text rather than
in this figure because porting the nidq event pipeline is still pending.

Anne's guidance (2026-04-02): keep it simple — show example neurons with
and without long pulses to demonstrate the double peak appears either way.
The core question for Dario is whether this response shape is a known
phenomenon in V1.
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

# Sessions with 30 ms pulses (GRB058 only)
GRB058_SESSIONS = ["20260312_134952", "20260319_131303"]

# Sessions with only 15 ms pulses — show for prevalence
GRB060_SESSION = "20260319_151909"

OUT_PATH = "/Users/gabriel/lib/ephys/figures/double_peak_dario.pdf"


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def run_double_peak_pipeline(subject, session, alignment_key="first_stim_ev_15ms"):
    """Return (rows, bin_centers) for double-peak excited units in one session.

    Each element of rows is a dict with keys:
      subject, session, uid, peth_15, n_tr_15
    """
    st_per_unit = fetch_good_units(subject, session)
    align_ev = fetch_session_events(subject, session)
    unit_ids = list(st_per_unit.keys())
    spike_times = list(st_per_unit.values())

    n_tr_15 = len(align_ev["first_stim_ev_15ms"])
    print(f"\n{subject}/{session}  units={len(unit_ids)}  15ms_trials={n_tr_15}")

    peth_15, bin_edges, bin_centers = compute_population_peth(
        spike_times_per_unit=spike_times,
        alignment_times=align_ev[alignment_key],
        **PETH_KWARGS,
    )
    _, masks = compute_unit_selectivity(
        peth_15, bin_edges, unit_ids=unit_ids, **SELECTIVITY_KWARGS
    )

    exc_idx = np.where(masks["excited"])[0]
    exc_peth_15 = peth_15[exc_idx]
    exc_ids = [unit_ids[i] for i in exc_idx]
    exc_spike_times = [spike_times[i] for i in exc_idx]
    print(f"  excited={len(exc_ids)}")

    peaks_df = classify_peak_count(
        exc_peth_15, bin_centers, unit_ids=exc_ids, **PEAK_KWARGS
    )
    double_ids = peaks_df.loc[peaks_df["n_peaks"] == 2, "unit"].tolist()
    print(f"  double-peak={double_ids}")

    rows = []
    for uid in double_ids:
        i = exc_ids.index(uid)
        rows.append(
            dict(
                subject=subject,
                session=session,
                uid=uid,
                peth_15=exc_peth_15[i],  # (n_trials, n_bins)
                spike_times=exc_spike_times[i],
                n_tr_15=n_tr_15,
                peaks_df_row=peaks_df[peaks_df["unit"] == uid].iloc[0],
            )
        )
    return rows, align_ev, bin_centers


def add_peth_30(rows, align_ev, bin_centers):
    """Attach 30 ms PSTH data to each row dict (in-place)."""
    spike_times_list = [r["spike_times"] for r in rows]
    n_tr_30 = len(align_ev["first_stim_ev_30ms"])
    if n_tr_30 == 0 or not rows:
        for r in rows:
            r["peth_30"] = None
            r["n_tr_30"] = 0
        return

    peth_30_all, _, _ = compute_population_peth(
        spike_times_per_unit=spike_times_list,
        alignment_times=align_ev["first_stim_ev_30ms"],
        **PETH_KWARGS,
    )
    for i, r in enumerate(rows):
        r["peth_30"] = peth_30_all[i]
        r["n_tr_30"] = n_tr_30


def plot_psth_panel(ax, bin_centers, peth_trials, color, label):
    """Plot mean ± SEM trace. Returns mean array."""
    mean = peth_trials.mean(axis=0)
    sem = peth_trials.std(axis=0) / np.sqrt(peth_trials.shape[0])
    ax.plot(bin_centers, mean, color=color, linewidth=1.5, label=label)
    ax.fill_between(bin_centers, mean - sem, mean + sem, alpha=0.25, color=color)
    return mean


def add_peak_markers(ax, peak_row, color):
    for pt, ph in zip(peak_row["peak_times"], peak_row["peak_heights"]):
        ax.plot(pt, ph, "v", color=color, markersize=7, zorder=5)


# ---------------------------------------------------------------------------
# Collect data — GRB058 (has 30 ms pulses)
# ---------------------------------------------------------------------------
all_rows = []  # list of row dicts in display order

for session in GRB058_SESSIONS:
    rows, align_ev, bin_centers = run_double_peak_pipeline("GRB058", session)
    add_peth_30(rows, align_ev, bin_centers)
    for r in rows:
        r["bin_centers"] = bin_centers
    all_rows.extend(rows)

# ---------------------------------------------------------------------------
# Collect data — GRB060 (15 ms only)
# ---------------------------------------------------------------------------
rows_60, align_ev_60, bin_centers_60 = run_double_peak_pipeline(
    "GRB060", GRB060_SESSION
)
for r in rows_60:
    r["peth_30"] = None
    r["n_tr_30"] = 0
    r["bin_centers"] = bin_centers_60
all_rows.extend(rows_60)

if not all_rows:
    print("No double-peak units found in any session.")
    raise SystemExit(0)

print(f"\nTotal panels: {len(all_rows)}")

# ---------------------------------------------------------------------------
# Figure — wide, single row, one panel per unit
# ---------------------------------------------------------------------------
n_panels = len(all_rows)
fig, axes = plt.subplots(1, n_panels, figsize=(3.5 * n_panels, 4), sharey=False)
if n_panels == 1:
    axes = [axes]

for ax, row in zip(axes, all_rows):
    bc = row["bin_centers"]

    plot_psth_panel(ax, bc, row["peth_15"], "tab:blue", f"15 ms (n={row['n_tr_15']})")
    add_peak_markers(ax, row["peaks_df_row"], color="tab:blue")

    if row["peth_30"] is not None:
        plot_psth_panel(
            ax, bc, row["peth_30"], "tab:orange", f"30 ms (n={row['n_tr_30']})"
        )
        # Detect and mark peaks in 30 ms PSTH
        peak_df_30 = classify_peak_count(
            row["peth_30"][np.newaxis, :, :],
            bc,
            unit_ids=[row["uid"]],
            **PEAK_KWARGS,
        )
        if not peak_df_30.empty:
            add_peak_markers(ax, peak_df_30.iloc[0], color="tab:orange")

    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.legend(fontsize=7, frameon=False, loc="upper right")

    sess_short = row["session"][:8]  # e.g. 20260312
    has_30 = row["peth_30"] is not None
    title = f"{row['subject']}  unit {row['uid']}\n{sess_short}"
    if not has_30:
        title += "  (15 ms only)"
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("Time from stim onset (s)", fontsize=8)
    ax.set_ylabel("sp/s", fontsize=8)
    ax.tick_params(labelsize=7)

fig.suptitle(
    "Double-peaked V1 responses to LED flashes  —  15 ms vs 30 ms pulse width",
    fontsize=10,
    y=1.02,
)
fig.tight_layout()

with PdfPages(OUT_PATH) as pdf:
    pdf.savefig(fig, bbox_inches="tight")

print(f"\nFigure saved: {OUT_PATH}")
