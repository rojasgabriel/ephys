"""Double-peak PSTH figure for the Dario email.

Two pages saved to figures/double_peak_dario.pdf:

  Page 1 — Prevalence
    Shows that double-peak units exist across mice.  One row per
    double-peak unit:
      • GRB006 unit 77: embedded from a cached PSTH screenshot
        (nidq pipeline not yet supported in code).
      • GRB058 units from sessions 20260312_134952 and 20260319_131303:
        computed 15 ms PSTHs with detected-peak markers (▼).

  Page 2 — Offset hypothesis
    For GRB058 double-peak units only: 15 ms (blue) and 30 ms (orange)
    PSTHs overlaid on the same axes.  Peak markers (▼) are drawn for
    both conditions so timing differences are visually explicit.

Sessions used:
  • 20260312_134952  ~25 % of 4 Hz trials had 30 ms pulses
  • 20260319_131303  ~50 % of 4 Hz trials had 30 ms pulses
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
SUBJECT = "GRB058"
SESSIONS = ["20260312_134952", "20260319_131303"]

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

GRB006_IMAGE = "/Users/gabriel/lib/ephys/figures/GRB006/unit77_peth_first_stim.png"
OUT_PATH = "/Users/gabriel/lib/ephys/figures/double_peak_dario.pdf"

# ---------------------------------------------------------------------------
# Compute GRB058 double-peak units
# ---------------------------------------------------------------------------
# rows: list of (session, uid, peth_15_row, peth_30_row, peaks_df_row,
#                n_tr_15, n_tr_30, bin_centers)
rows = []

for session in SESSIONS:
    st_per_unit = fetch_good_units(SUBJECT, session)
    align_ev = fetch_session_events(SUBJECT, session)
    unit_ids = list(st_per_unit.keys())
    spike_times = list(st_per_unit.values())

    n_tr_15 = len(align_ev["first_stim_ev_15ms"])
    n_tr_30 = len(align_ev["first_stim_ev_30ms"])
    print(f"\n{SUBJECT} / {session}")
    print(
        f"  Units: {len(unit_ids)},  15 ms trials: {n_tr_15},  30 ms trials: {n_tr_30}"
    )

    peth_15, bin_edges, bin_centers = compute_population_peth(
        spike_times_per_unit=spike_times,
        alignment_times=align_ev["first_stim_ev_15ms"],
        **PETH_KWARGS,
    )

    _, masks = compute_unit_selectivity(
        peth_15, bin_edges, unit_ids=unit_ids, **SELECTIVITY_KWARGS
    )

    exc_idx = np.where(masks["excited"])[0]
    exc_peth_15 = peth_15[exc_idx]
    exc_ids = [unit_ids[i] for i in exc_idx]
    exc_spike_times = [spike_times[i] for i in exc_idx]
    print(f"  Excited: {len(exc_ids)}")

    peaks_df = classify_peak_count(
        exc_peth_15, bin_centers, unit_ids=exc_ids, **PEAK_KWARGS
    )
    double_ids = peaks_df.loc[peaks_df["n_peaks"] == 2, "unit"].tolist()
    print(f"  Double-peak: {double_ids}")

    if not double_ids:
        continue

    dp_idx = [exc_ids.index(uid) for uid in double_ids]
    dp_peth_15 = exc_peth_15[dp_idx]
    dp_spike_times = [exc_spike_times[i] for i in dp_idx]

    dp_peth_30, _, _ = compute_population_peth(
        spike_times_per_unit=dp_spike_times,
        alignment_times=align_ev["first_stim_ev_30ms"],
        **PETH_KWARGS,
    )

    for i, uid in enumerate(double_ids):
        peak_row = peaks_df[peaks_df["unit"] == uid].iloc[0]
        rows.append(
            (
                session,
                uid,
                dp_peth_15[i],
                dp_peth_30[i],
                peak_row,
                n_tr_15,
                n_tr_30,
                bin_centers,
            )
        )

if not rows:
    print("No double-peak units found in any session.")
    raise SystemExit(0)

print(f"\nTotal double-peak units found: {len(rows)}")

# ---------------------------------------------------------------------------
# Helper: draw a single PSTH trace + SEM shading on ax
# ---------------------------------------------------------------------------


def _plot_trace(ax, bin_centers, peth_trials, color, label):
    """peth_trials: (n_trials, n_bins) array."""
    mean = peth_trials.mean(axis=0)
    sem = peth_trials.std(axis=0) / np.sqrt(peth_trials.shape[0])
    ax.plot(bin_centers, mean, color=color, linewidth=1.5, label=label)
    ax.fill_between(bin_centers, mean - sem, mean + sem, alpha=0.25, color=color)
    return mean


def _add_peaks(ax, peak_row, bin_centers, color="crimson"):
    for pt, ph in zip(peak_row["peak_times"], peak_row["peak_heights"]):
        ax.plot(pt, ph, "v", color=color, markersize=8, zorder=5)


# ---------------------------------------------------------------------------
# Page 1 — Prevalence
# ---------------------------------------------------------------------------
# GRB006 unit 77 is one extra row at the top (image panel).
n_grb058_rows = len(rows)
n_total_rows = 1 + n_grb058_rows  # GRB006 + GRB058 units

fig1, axes1 = plt.subplots(
    n_total_rows,
    1,
    figsize=(5, 3 * n_total_rows),
    squeeze=False,
)

# -- GRB006 unit 77 (embedded screenshot) --
ax_img = axes1[0, 0]
img = plt.imread(GRB006_IMAGE)
ax_img.imshow(img)
ax_img.axis("off")
ax_img.set_title("GRB006  unit 77  —  15 ms  (archived figure)", fontsize=9)
ax_img.text(
    -0.05,
    0.5,
    "GRB006",
    transform=ax_img.transAxes,
    va="center",
    ha="right",
    fontsize=8,
    rotation=90,
)

# -- GRB058 units --
for i, (session, uid, p15, p30, peak_row, n_tr_15, n_tr_30, bin_centers) in enumerate(
    rows
):
    ax = axes1[1 + i, 0]
    mean15 = _plot_trace(ax, bin_centers, p15, "tab:blue", f"15 ms (n={n_tr_15})")
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    _add_peaks(ax, peak_row, bin_centers, color="crimson")
    ax.set_title(f"GRB058  unit {uid}  —  15 ms  (n={n_tr_15})", fontsize=9)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("sp/s", fontsize=8)
    ax.text(
        -0.12,
        0.5,
        session.replace("_", " "),
        transform=ax.transAxes,
        va="center",
        ha="right",
        fontsize=7,
        rotation=90,
    )

fig1.suptitle("Double-peak units across mice", fontsize=11, y=1.01)
fig1.tight_layout()

# ---------------------------------------------------------------------------
# Page 2 — Offset hypothesis
# ---------------------------------------------------------------------------
fig2, axes2 = plt.subplots(
    n_grb058_rows,
    1,
    figsize=(5, 3 * n_grb058_rows),
    squeeze=False,
)

for i, (session, uid, p15, p30, peak_row, n_tr_15, n_tr_30, bin_centers) in enumerate(
    rows
):
    ax = axes2[i, 0]

    _plot_trace(ax, bin_centers, p15, "tab:blue", f"15 ms (n={n_tr_15})")
    _plot_trace(ax, bin_centers, p30, "tab:orange", f"30 ms (n={n_tr_30})")
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)

    # Peak markers: 15 ms in blue, 30 ms in orange
    _add_peaks(ax, peak_row, bin_centers, color="tab:blue")

    # Detect peaks in 30ms PSTH for the timing comparison
    peak_df_30 = classify_peak_count(
        p30[np.newaxis, :, :],  # (1, n_trials, n_bins)
        bin_centers,
        unit_ids=[uid],
        **PEAK_KWARGS,
    )
    if not peak_df_30.empty:
        pr30 = peak_df_30.iloc[0]
        for pt, ph in zip(pr30["peak_times"], pr30["peak_heights"]):
            ax.plot(pt, ph, "v", color="tab:orange", markersize=8, zorder=5)

    ax.legend(fontsize=8, frameon=False)
    ax.set_title(f"GRB058  unit {uid}  —  15 ms vs 30 ms", fontsize=9)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("sp/s", fontsize=8)
    ax.text(
        -0.12,
        0.5,
        session.replace("_", " "),
        transform=ax.transAxes,
        va="center",
        ha="right",
        fontsize=7,
        rotation=90,
    )

fig2.suptitle(
    "Offset hypothesis: do 30 ms pulses shift the second peak?", fontsize=10, y=1.01
)
fig2.tight_layout()

# ---------------------------------------------------------------------------
# Save both pages to one PDF
# ---------------------------------------------------------------------------
with PdfPages(OUT_PATH) as pdf:
    pdf.savefig(fig1, bbox_inches="tight")
    pdf.savefig(fig2, bbox_inches="tight")

print(f"\nFigure saved: {OUT_PATH}")
