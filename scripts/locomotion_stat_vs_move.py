"""Stationary vs. movement stim response analysis.

Compares V1 responses to the last stationary stimulus in a trial (animal
still in center port) vs. the first movement stimulus (animal en route to
the response port).

Output: figures/locomotion_stat_vs_move_{subject}_{session}.pdf
  Row 0 — diagnostic panels: stim count distributions, center-port hold
           time, travel time, stims-per-trial scatter.
  Row 1 — PSTH + per-unit scatter for all units (left) and excited units
           (right).
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import sem, wilcoxon

from ephys.src.utils.utils_IO import (
    fetch_good_units,
    fetch_session_events,
    fetch_trial_metadata,
)
from ephys.src.utils.utils_analysis import (
    compute_population_peth,
    compute_unit_selectivity,
    build_trial_stim_classification,
)

subject = "GRB058"
session = "20260312_134952"
OUT_PATH = f"/Users/gabriel/lib/ephys/figures/locomotion_stat_vs_move_{subject}_{session[:8]}.pdf"

print("Loading data...")
st_per_unit = fetch_good_units(subject, session)
align_ev = fetch_session_events(subject, session)
trial_df = fetch_trial_metadata(subject, session, align_ev)
unit_ids = list(st_per_unit.keys())
spike_times = list(st_per_unit.values())

# Selectivity on full trial set
peth_all, bin_edges, bin_centers = compute_population_peth(
    spike_times_per_unit=spike_times,
    alignment_times=align_ev["first_stim_ev_15ms"],
    pre_seconds=0.1,
    post_seconds=0.15,
    binwidth_ms=10,
    t_rise=0.001,
    t_decay=0.025,
)
_, masks = compute_unit_selectivity(
    peth_all,
    bin_edges,
    unit_ids=unit_ids,
    base_window=(-0.04, 0.0),
    resp_window=(0.06, 0.10),
    test="wilcoxon",
    correction="bonferroni",
    alpha=0.05,
)
exc_idx = np.where(masks["excited"])[0]
exc_spks = [spike_times[i] for i in exc_idx]
print(f"  Excited units: {len(exc_idx)} / {len(unit_ids)}")

# Classify stims
trial_ts = build_trial_stim_classification(align_ev, trial_df)
n_trials = len(trial_ts)
print(f"  Chipmunk trials:            {len(trial_df)}")
print(f"  t_react missing (withdraw): {trial_df['t_react'].isna().sum()}")
print(f"  Trials with stat+move:      {n_trials}")

# ── Stim distribution stats ──────────────────────────────────────────────────
n_stat = trial_ts["stationary_stims"].apply(len)
n_move = trial_ts["movement_stims"].apply(len)
cp_hold = trial_ts["cp_exit_obx"] - trial_ts["cp_entry"]  # time in center port
travel = trial_ts["rp_entry"] - trial_ts["cp_exit_obx"]  # center exit → response

print("\n── Stationary stims per trial ──")
print(n_stat.describe().round(2).to_string())
print("\n── Movement stims per trial ──")
print(n_move.describe().round(2).to_string())
print("\n── Time in center port before exit (s) ──")
print(cp_hold.describe().round(3).to_string())
print("\n── Travel time: cp_exit → rp_entry (s) ──")
print(travel.describe().round(3).to_string())

# Flag suspiciously long center-port holds (> 3 s = longer than one stim train)
n_long = (cp_hold > 3.0).sum()
print(
    f"\nTrials with cp_hold > 3 s: {n_long} / {n_trials}  ({100 * n_long / n_trials:.0f}%)"
)

last_stat = np.array([s[-1] for s in trial_ts["stationary_stims"]])
first_move = np.array([s[0] for s in trial_ts["movement_stims"]])

PETH_KWARGS = dict(
    pre_seconds=0.1, post_seconds=0.15, binwidth_ms=10, t_rise=0.001, t_decay=0.025
)

print("\nComputing PETHs...")
peth_stat_all, _, bc = compute_population_peth(spike_times, last_stat, **PETH_KWARGS)
peth_move_all, _, _ = compute_population_peth(spike_times, first_move, **PETH_KWARGS)
peth_stat_exc, _, _ = compute_population_peth(exc_spks, last_stat, **PETH_KWARGS)
peth_move_exc, _, _ = compute_population_peth(exc_spks, first_move, **PETH_KWARGS)


def peak_per_unit(peth):
    return peth.max(axis=2).mean(axis=1)


def wilcox_str(a, b):
    s, p = wilcoxon(a, b)
    pct = 100 * (b > a).mean()
    return f"{pct:.0f}% above diag  p={p:.3f}"


pk_stat_all = peak_per_unit(peth_stat_all)
pk_move_all = peak_per_unit(peth_move_all)
pk_stat_exc = peak_per_unit(peth_stat_exc)
pk_move_exc = peak_per_unit(peth_move_exc)

print(f"\nAll units  ({len(unit_ids)}):      {wilcox_str(pk_stat_all, pk_move_all)}")
print(f"Excited units ({len(exc_idx)}): {wilcox_str(pk_stat_exc, pk_move_exc)}")

# ── Figure ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(2, 4, hspace=0.45, wspace=0.35)

# ── Row 0: stim distribution diagnostics ────────────────────────────────────
ax = fig.add_subplot(gs[0, 0])
ax.hist(
    n_stat,
    bins=range(0, int(n_stat.max()) + 2),
    color="steelblue",
    alpha=0.7,
    edgecolor="white",
    lw=0.5,
    label="stationary",
)
ax.hist(
    n_move,
    bins=range(0, int(n_move.max()) + 2),
    color="darkorange",
    alpha=0.7,
    edgecolor="white",
    lw=0.5,
    label="movement",
)
ax.set_xlabel("Stims per trial")
ax.set_ylabel("Trials")
ax.set_title("Stim count distribution")
ax.legend(fontsize=7, frameon=False)

ax = fig.add_subplot(gs[0, 1])
ax.hist(cp_hold, bins=30, color="mediumpurple", edgecolor="white", lw=0.5)
ax.axvline(1.0, color="k", linestyle="--", lw=0.8, label="1 s (1 train)")
ax.axvline(3.0, color="crimson", linestyle="--", lw=0.8, label="3 s cutoff")
ax.set_xlabel("Time in center port (s)")
ax.set_ylabel("Trials")
ax.set_title(f"cp_entry → cp_exit\n{n_long}/{n_trials} trials > 3 s")
ax.legend(fontsize=7, frameon=False)

ax = fig.add_subplot(gs[0, 2])
ax.hist(travel, bins=25, color="seagreen", edgecolor="white", lw=0.5)
ax.set_xlabel("Travel time (s)")
ax.set_ylabel("Trials")
ax.set_title("cp_exit → rp_entry")

ax = fig.add_subplot(gs[0, 3])
ax.scatter(n_stat, n_move, s=12, alpha=0.4, color="k")
ax.set_xlabel("Stationary stims")
ax.set_ylabel("Movement stims")
ax.set_title("Stims per trial: stat vs move")
lim = max(n_stat.max(), n_move.max()) + 1
ax.set_xlim(0, lim)
ax.set_ylim(0, lim)
ax.plot([0, lim], [0, lim], "k--", alpha=0.3, lw=0.8)


# ── Row 1: PSTH + scatter for all units (left) and excited units (right) ────
def plot_psth_scatter(ax_psth, ax_scatter, peth_s, peth_m, pk_s, pk_m, label, n_units):
    ms = peth_s.mean(axis=(0, 1))
    ss = sem(peth_s.mean(axis=1), axis=0)
    mm = peth_m.mean(axis=(0, 1))
    sm = sem(peth_m.mean(axis=1), axis=0)
    ax_psth.plot(
        bc, ms, color="steelblue", lw=1.6, label=f"last stat (n={len(last_stat)})"
    )
    ax_psth.fill_between(bc, ms - ss, ms + ss, alpha=0.25, color="steelblue")
    ax_psth.plot(
        bc, mm, color="darkorange", lw=1.6, label=f"first move (n={len(first_move)})"
    )
    ax_psth.fill_between(bc, mm - sm, mm + sm, alpha=0.25, color="darkorange")
    ax_psth.axvline(0, color="gray", linestyle="--", lw=0.8)
    ax_psth.set_xlabel("Time from stim onset (s)")
    ax_psth.set_ylabel("sp/s")
    ax_psth.set_title(f"Population PSTH\n{label} ({n_units} units)")
    ax_psth.legend(fontsize=6, frameon=False)

    lim = max(pk_s.max(), pk_m.max()) * 1.08
    ax_scatter.scatter(pk_s, pk_m, color="k", s=20, alpha=0.45, zorder=3)
    ax_scatter.plot([0, lim], [0, lim], "k--", alpha=0.4, lw=0.8)
    _, p = wilcoxon(pk_s, pk_m)
    pct = 100 * (pk_m > pk_s).mean()
    ax_scatter.set_xlabel("Stationary peak (sp/s)")
    ax_scatter.set_ylabel("Movement peak (sp/s)")
    ax_scatter.set_title(f"Per-unit mean peak\n{pct:.0f}% above diag  p={p:.3f}")
    ax_scatter.set_xlim(0, lim)
    ax_scatter.set_ylim(0, lim)
    ax_scatter.set_aspect("equal")


plot_psth_scatter(
    fig.add_subplot(gs[1, 0]),
    fig.add_subplot(gs[1, 1]),
    peth_stat_all,
    peth_move_all,
    pk_stat_all,
    pk_move_all,
    "all units",
    len(unit_ids),
)
plot_psth_scatter(
    fig.add_subplot(gs[1, 2]),
    fig.add_subplot(gs[1, 3]),
    peth_stat_exc,
    peth_move_exc,
    pk_stat_exc,
    pk_move_exc,
    "excited units",
    len(exc_idx),
)

fig.suptitle(f"{subject}  {session}  —  stationary vs. movement", fontsize=11, y=1.01)

fig.savefig(OUT_PATH, bbox_inches="tight")
print(f"\nFigure saved: {OUT_PATH}")
