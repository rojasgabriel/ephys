"""Stationary vs. movement stim response analysis.

Compares V1 responses to the FIRST stationary stim (animal in center port,
last entry before exit) vs. the FIRST movement stim (animal en route to the
response port).  Trials must contain at least one of each; reentrances are
handled by using the last cp_entry before t_react.

Output: figures/locomotion_stat_vs_move_{subject}_{session}.pdf

  Row 0 — diagnostics:
    1. Stim count distributions (stationary, movement)
    2. cp_entry → cp_exit hold time
    3. Travel time (cp_exit → rp_entry)
    4. Reentrance counts per trial

  Row 1 — clock + behavioral validation:
    1. Inter-trial-interval residuals (bpod sync vs OBX trial_start)
    2. Leave-one-out interpolation residual at each sync point
    3. Expected vs observed stim count per trial
    4. stim_rate_vision distribution per trial

  Row 2 — PSTH + per-unit scatter (first stat vs first move):
    Left  pair: all units
    Right pair: excited units only
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

# Pre-stim baseline window kept short (40 ms) so the previous stim's response
# (which can be as little as ~55 ms away at 18 Hz) doesn't bleed into the
# baseline.  Post-stim window is 150 ms.
PETH_KWARGS = dict(
    pre_seconds=0.04,
    post_seconds=0.15,
    binwidth_ms=10,
    t_rise=0.001,
    t_decay=0.025,
)

print("Loading data...")
st_per_unit = fetch_good_units(subject, session)
align_ev = fetch_session_events(subject, session)
trial_df = fetch_trial_metadata(subject, session, align_ev)
unit_ids = list(st_per_unit.keys())
spike_times = list(st_per_unit.values())
print(f"  Units: {len(unit_ids)}  Chipmunk trials: {len(trial_df)}")

# ── Selectivity on full trial set ────────────────────────────────────────────
peth_all, bin_edges, _ = compute_population_peth(
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

# ── Stim classification ─────────────────────────────────────────────────────
trial_ts = build_trial_stim_classification(align_ev, trial_df)
n_trials = len(trial_ts)
print(f"  Trials with stat+move stims: {n_trials}")

n_stat = trial_ts["stationary_stims"].apply(len)
n_move = trial_ts["movement_stims"].apply(len)
cp_hold = trial_ts["cp_exit_obx"] - trial_ts["cp_entry"]
travel = trial_ts["rp_entry"] - trial_ts["cp_exit_obx"]
n_reentries = trial_ts["n_cp_entries"]

print(f"  Trials with > 1 cp entry (reentrance): {(n_reentries > 1).sum()}")
print(
    f"  After reentrance fix, cp_hold > 3 s trials: {(cp_hold > 3.0).sum()}"
    f"  (max cp_hold = {cp_hold.max():.2f} s)"
)

# ── Clock validation ────────────────────────────────────────────────────────
n_aligned = min(len(trial_df), len(align_ev["trial_start"]))
bpod_sync = trial_df["t_sync"].iloc[:n_aligned].to_numpy(dtype=float)
obx_sync = align_ev["trial_start"][:n_aligned].astype(float)
valid = np.isfinite(bpod_sync) & np.isfinite(obx_sync)
bpod_sync = bpod_sync[valid]
obx_sync = obx_sync[valid]

# Leave-one-out residual: for each sync point, predict it from the others
loo_resid = np.full(len(bpod_sync), np.nan)
for i in range(1, len(bpod_sync) - 1):
    bp_train = np.delete(bpod_sync, i)
    obx_train = np.delete(obx_sync, i)
    pred = np.interp(bpod_sync[i], bp_train, obx_train)
    loo_resid[i] = obx_sync[i] - pred
loo_resid_valid = loo_resid[np.isfinite(loo_resid)]
print(
    f"  LOO interp residual: mean={np.mean(loo_resid_valid) * 1e3:.2f} ms  "
    f"std={np.std(loo_resid_valid) * 1e3:.2f} ms  "
    f"max|·|={np.max(np.abs(loo_resid_valid)) * 1e3:.2f} ms"
)

# Local clock-rate consistency: ratio of consecutive ITI gaps
iti_obx = np.diff(obx_sync)
iti_bpod = np.diff(bpod_sync)
slope_local = iti_obx / iti_bpod  # ≈ 1.0 if clocks track perfectly

# ── Stim count validation ───────────────────────────────────────────────────
# Expected count from behavioral metadata vs OBX-observed within trial window.
stim_times = np.asarray(align_ev["stim_ev_15ms"])
obs_counts = []
exp_counts = []
for _, row in trial_ts.iterrows():
    i = int(row["trial_idx"])
    rate = trial_df["stim_rate_vision"].iloc[i]
    dur = trial_df["stim_duration"].iloc[i]
    if rate is None or dur is None or not np.isfinite(rate) or not np.isfinite(dur):
        continue
    expected = int(round(rate * dur))
    # Observed = OBX 15 ms stims falling between cp_entry and rp_entry
    obs = int(((stim_times >= row["cp_entry"]) & (stim_times <= row["rp_entry"])).sum())
    obs_counts.append(obs)
    exp_counts.append(expected)
obs_counts = np.array(obs_counts)
exp_counts = np.array(exp_counts)
mismatch = obs_counts - exp_counts
print(
    f"  Stim count match (obs - exp): "
    f"median={int(np.median(mismatch))}  "
    f"mean={mismatch.mean():.2f}  "
    f"|diff|<=1 in {(np.abs(mismatch) <= 1).mean() * 100:.0f}% of trials"
)

# ── Build PETHs aligned to first stat / first move ──────────────────────────
first_stat = np.array([s[0] for s in trial_ts["stationary_stims"]])
first_move = np.array([s[0] for s in trial_ts["movement_stims"]])

print("Computing PETHs...")
peth_stat_all, _, bc = compute_population_peth(spike_times, first_stat, **PETH_KWARGS)
peth_move_all, _, _ = compute_population_peth(spike_times, first_move, **PETH_KWARGS)
peth_stat_exc, _, _ = compute_population_peth(exc_spks, first_stat, **PETH_KWARGS)
peth_move_exc, _, _ = compute_population_peth(exc_spks, first_move, **PETH_KWARGS)


def peak_per_unit(peth):
    return peth.max(axis=2).mean(axis=1)


def wilcox_str(a, b):
    _, p = wilcoxon(a, b)
    pct = 100 * (b > a).mean()
    return f"{pct:.0f}% above diag  p={p:.3f}"


pk_stat_all = peak_per_unit(peth_stat_all)
pk_move_all = peak_per_unit(peth_move_all)
pk_stat_exc = peak_per_unit(peth_stat_exc)
pk_move_exc = peak_per_unit(peth_move_exc)

print(f"\nAll units  ({len(unit_ids)}):      {wilcox_str(pk_stat_all, pk_move_all)}")
print(f"Excited units ({len(exc_idx)}): {wilcox_str(pk_stat_exc, pk_move_exc)}")

# ── Figure ──────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 14))
gs = fig.add_gridspec(3, 4, hspace=0.5, wspace=0.35)

# Row 0: stim distribution diagnostics ───────────────────────────────────────
ax = fig.add_subplot(gs[0, 0])
mx = max(int(n_stat.max()), int(n_move.max())) + 2
ax.hist(
    n_stat,
    bins=range(0, mx),
    color="steelblue",
    alpha=0.7,
    edgecolor="white",
    lw=0.5,
    label="stationary",
)
ax.hist(
    n_move,
    bins=range(0, mx),
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
ax.set_xlabel("Time in center port (s)")
ax.set_ylabel("Trials")
ax.set_title(f"cp_entry → cp_exit\nmax = {cp_hold.max():.2f} s")
ax.legend(fontsize=7, frameon=False)

ax = fig.add_subplot(gs[0, 2])
ax.hist(travel, bins=25, color="seagreen", edgecolor="white", lw=0.5)
ax.set_xlabel("Travel time (s)")
ax.set_ylabel("Trials")
ax.set_title("cp_exit → rp_entry")

ax = fig.add_subplot(gs[0, 3])
mx = int(n_reentries.max()) + 2
ax.hist(
    n_reentries, bins=range(1, mx + 1), color="indianred", edgecolor="white", lw=0.5
)
ax.set_xlabel("cp entries per trial")
ax.set_ylabel("Trials")
ax.set_title(f"Reentrance counts\n{(n_reentries > 1).sum()}/{n_trials} > 1 entry")

# Row 1: clock + behavioral validation ──────────────────────────────────────
ax = fig.add_subplot(gs[1, 0])
ax.plot(slope_local, "o", color="k", markersize=2, alpha=0.6)
ax.axhline(1.0, color="crimson", linestyle="--", lw=0.8)
ax.set_xlabel("Trial index")
ax.set_ylabel("OBX_iti / bpod_iti")
ax.set_title(f"Local clock rate ratio\nmean={slope_local.mean():.6f}")

ax = fig.add_subplot(gs[1, 1])
ax.hist(loo_resid_valid * 1e3, bins=30, color="slategray", edgecolor="white", lw=0.5)
ax.axvline(0, color="k", linestyle="--", lw=0.8)
ax.set_xlabel("LOO residual (ms)")
ax.set_ylabel("Sync points")
ax.set_title(
    f"t_sync interp residual\nstd={np.std(loo_resid_valid) * 1e3:.2f} ms  "
    f"max|·|={np.max(np.abs(loo_resid_valid)) * 1e3:.2f} ms"
)

ax = fig.add_subplot(gs[1, 2])
mx = max(obs_counts.max(), exp_counts.max()) + 2
ax.scatter(exp_counts, obs_counts, s=14, alpha=0.4, color="k")
ax.plot([0, mx], [0, mx], "k--", alpha=0.4, lw=0.8)
ax.set_xlabel("Expected (rate × duration)")
ax.set_ylabel("Observed (OBX 15 ms)")
ax.set_title(
    f"Stim count: behavioral vs OBX\n"
    f"|diff|≤1 in {(np.abs(mismatch) <= 1).mean() * 100:.0f}% of trials"
)
ax.set_xlim(0, mx)
ax.set_ylim(0, mx)
ax.set_aspect("equal")

ax = fig.add_subplot(gs[1, 3])
rates = trial_df["stim_rate_vision"].dropna().to_numpy()
ax.hist(rates, bins=20, color="darkkhaki", edgecolor="white", lw=0.5)
ax.set_xlabel("stim_rate_vision (Hz)")
ax.set_ylabel("Trials")
ax.set_title(f"Stim rate distribution\n(min ISI ≈ {1000 / rates.max():.0f} ms)")


# Row 2: PSTH + per-unit scatter ────────────────────────────────────────────
def plot_psth_scatter(ax_psth, ax_sc, peth_s, peth_m, pk_s, pk_m, label, n_units):
    ms = peth_s.mean(axis=(0, 1))
    ss = sem(peth_s.mean(axis=1), axis=0)
    mm = peth_m.mean(axis=(0, 1))
    sm = sem(peth_m.mean(axis=1), axis=0)
    ax_psth.plot(
        bc, ms, color="steelblue", lw=1.6, label=f"first stat (n={len(first_stat)})"
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
    ax_sc.scatter(pk_s, pk_m, color="k", s=20, alpha=0.45, zorder=3)
    ax_sc.plot([0, lim], [0, lim], "k--", alpha=0.4, lw=0.8)
    _, p = wilcoxon(pk_s, pk_m)
    pct = 100 * (pk_m > pk_s).mean()
    ax_sc.set_xlabel("Stationary peak (sp/s)")
    ax_sc.set_ylabel("Movement peak (sp/s)")
    ax_sc.set_title(f"Per-unit mean peak\n{pct:.0f}% above diag  p={p:.3f}")
    ax_sc.set_xlim(0, lim)
    ax_sc.set_ylim(0, lim)
    ax_sc.set_aspect("equal")


plot_psth_scatter(
    fig.add_subplot(gs[2, 0]),
    fig.add_subplot(gs[2, 1]),
    peth_stat_all,
    peth_move_all,
    pk_stat_all,
    pk_move_all,
    "all units",
    len(unit_ids),
)
plot_psth_scatter(
    fig.add_subplot(gs[2, 2]),
    fig.add_subplot(gs[2, 3]),
    peth_stat_exc,
    peth_move_exc,
    pk_stat_exc,
    pk_move_exc,
    "excited units",
    len(exc_idx),
)

fig.suptitle(
    f"{subject}  {session}  —  first stationary vs. first movement",
    fontsize=11,
    y=1.005,
)

fig.savefig(OUT_PATH, bbox_inches="tight")
print(f"\nFigure saved: {OUT_PATH}")
