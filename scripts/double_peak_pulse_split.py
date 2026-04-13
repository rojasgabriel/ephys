"""Double-peak PSTH comparison: 15 ms vs 30 ms stimulus pulses.

Loads GRB058 / 20260312_134952 (the session with both 15 ms and 30 ms pulses),
identifies excited units with double-peaked responses to 15 ms pulses, then
plots their PSTHs side-by-side for 15 ms and 30 ms conditions.

Figure is saved to figures/double_peak_pulse_split_{subject}_{session}.pdf.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ephys.src.utils.utils_IO import fetch_good_units, fetch_session_events
from ephys.src.utils.utils_analysis import (
    compute_population_peth,
    compute_unit_selectivity,
    classify_peak_count,
)

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
subject = "GRB058"
session = "20260312_134952"

st_per_unit = fetch_good_units(subject, session)
align_ev = fetch_session_events(subject, session)
unit_ids = list(st_per_unit.keys())
spike_times = list(st_per_unit.values())

print(f"Session: {subject} / {session}")
print(f"Units loaded: {len(unit_ids)}")
print(f"15 ms trials: {len(align_ev['first_stim_ev_15ms'])}")
print(f"30 ms trials: {len(align_ev['first_stim_ev_30ms'])}")

# ---------------------------------------------------------------------------
# 2. PETH aligned to 15 ms pulses + selectivity + peak classification
# ---------------------------------------------------------------------------
peth_15, bin_edges, bin_centers = compute_population_peth(
    spike_times_per_unit=spike_times,
    alignment_times=align_ev["first_stim_ev_15ms"],
    pre_seconds=0.1,
    post_seconds=0.15,
    binwidth_ms=10,
    t_rise=None,
    t_decay=None,
)

results_df, masks = compute_unit_selectivity(
    peth_15,
    bin_edges,
    unit_ids=unit_ids,
    base_window=(-0.04, 0.0),
    resp_window=(0.06, 0.10),
    test="wilcoxon",
    correction="bonferroni",
    alpha=0.05,
)

exc_idx = np.where(masks["excited"])[0]
exc_peth_15 = peth_15[exc_idx]
exc_ids = [unit_ids[i] for i in exc_idx]
exc_spike_times = [spike_times[i] for i in exc_idx]

print(f"Excited: {len(exc_ids)}")

peaks_df = classify_peak_count(
    exc_peth_15,
    bin_centers,
    unit_ids=exc_ids,
    search_window=(0.0, 0.15),
    baseline_window=(-0.04, 0.0),
    min_prominence_frac=0.25,
    min_distance_ms=20.0,
    binwidth_ms=10.0,
)

# ---------------------------------------------------------------------------
# 3. Identify double-peak units
# ---------------------------------------------------------------------------
double_mask = peaks_df["n_peaks"] == 2
double_ids = peaks_df.loc[double_mask, "unit"].tolist()
print(f"Double-peak: {len(double_ids)}")

if not double_ids:
    print("No double-peak units found — nothing to plot.")
    raise SystemExit(0)

dp_peth_15 = exc_peth_15[[exc_ids.index(uid) for uid in double_ids]]
dp_spike_times = [exc_spike_times[exc_ids.index(uid)] for uid in double_ids]

# ---------------------------------------------------------------------------
# 4. PETH aligned to 30 ms pulses (double-peak units only)
# ---------------------------------------------------------------------------
dp_peth_30, _, _ = compute_population_peth(
    spike_times_per_unit=dp_spike_times,
    alignment_times=align_ev["first_stim_ev_30ms"],
    pre_seconds=0.1,
    post_seconds=0.15,
    binwidth_ms=10,
    t_rise=None,
    t_decay=None,
)

# ---------------------------------------------------------------------------
# 5. Plot: one row per unit, left = 15 ms, right = 30 ms
# ---------------------------------------------------------------------------
n_units = len(double_ids)
n_tr_15 = dp_peth_15.shape[1]
n_tr_30 = dp_peth_30.shape[1]

fig, axes = plt.subplots(
    n_units, 2, figsize=(7, 3 * n_units), sharey="row", squeeze=False
)

for i, uid in enumerate(double_ids):
    for col, (peth_arr, color, label, n_tr) in enumerate(
        [
            (dp_peth_15[i], "tab:blue", "15 ms", n_tr_15),
            (dp_peth_30[i], "tab:orange", "30 ms", n_tr_30),
        ]
    ):
        ax = axes[i, col]
        mean = peth_arr.mean(axis=0)
        sem = peth_arr.std(axis=0) / np.sqrt(peth_arr.shape[0])
        ax.plot(bin_centers, mean, color=color, linewidth=1.5)
        ax.fill_between(bin_centers, mean - sem, mean + sem, alpha=0.25, color=color)
        ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_title(f"Unit {uid} — {label}  (n={n_tr})", fontsize=9)
        ax.set_xlabel("Time (s)")

    # Mark detected peaks on the 15 ms column
    row = peaks_df[peaks_df["unit"] == uid].iloc[0]
    for pt, ph in zip(row["peak_times"], row["peak_heights"]):
        axes[i, 0].plot(pt, ph, "v", color="crimson", markersize=8, zorder=5)

    axes[i, 0].set_ylabel("sp/s")

fig.suptitle(
    f"{subject}  {session}  —  double-peak units: 15 ms vs 30 ms",
    fontsize=11,
)
plt.tight_layout()

out_path = (
    f"/Users/gabriel/lib/ephys/figures/double_peak_pulse_split_{subject}_{session}.pdf"
)
fig.savefig(out_path, bbox_inches="tight")
print(f"Figure saved: {out_path}")
