"""Depth-binned locomotion modulation summary for a single session.

Classifies units as move-excited / move-suppressed / no-effect using the same
paired stationary-vs-movement logic as locomotion_stat_vs_move.py, then
summarizes class composition as a function of recording depth.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

from ephys.src.utils.utils_IO import (
    fetch_good_units_with_depth,
    fetch_session_events,
    fetch_trial_metadata,
)
from ephys.src.utils.utils_analysis import (
    build_trial_stim_classification,
    compute_population_peth,
    extract_conditioned_stim_anchors,
)

subject = "GRB058"
session = "20260312_134952"
OUT_PATH = f"/Users/gabriel/lib/ephys/figures/locomotion_depth_binned_{subject}_{session[:8]}.pdf"

PETH_KWARGS = dict(
    pre_seconds=0.04,
    post_seconds=0.15,
    binwidth_ms=10,
)
PETH_SCALE_BACK = PETH_KWARGS["binwidth_ms"] / 1000.0
RESP_WINDOW = (0.04, 0.10)
EFFECT_WINDOW = (0.0, 0.12)
PEAK_HALF_WINDOW_S = 0.015
SNR_THRESHOLD = 3.0
QVAL_ALPHA = 0.05
DEPTH_BIN_WIDTH_UM = 100.0


def resp_per_unit(peth, bc, window=RESP_WINDOW):
    """Mean firing rate in the response window, averaged across trials."""
    mask = (bc >= window[0]) & (bc < window[1])
    return peth[:, :, mask].mean(axis=(1, 2))


def snr_per_unit(peth, bc, window=RESP_WINDOW):
    """Per-unit response SNR using trial-wise response-window means."""
    mask = (bc >= window[0]) & (bc < window[1])
    resp = peth[:, :, mask].mean(axis=2)  # (n_units, n_trials)
    unit_mean = resp.mean(axis=1)
    unit_sem = resp.std(axis=1) / np.sqrt(resp.shape[1])
    return unit_mean / (unit_sem + 1e-3)


def per_unit_move_vs_stat_stats(
    peth_stat,
    peth_move,
    bc,
    window=EFFECT_WINDOW,
    half_window_s=PEAK_HALF_WINDOW_S,
):
    """Per-unit paired stats across trials for movement vs stationary response."""
    effect_mask = (bc >= window[0]) & (bc < window[1])
    if not effect_mask.any():
        raise ValueError("EFFECT_WINDOW does not overlap available bins.")

    n_units, n_trials, _ = peth_stat.shape
    stat_trials = np.zeros((n_units, n_trials), dtype=float)
    move_trials = np.zeros((n_units, n_trials), dtype=float)
    peak_latencies = np.full(n_units, np.nan, dtype=float)

    mean_stat = peth_stat.mean(axis=1)
    mean_move = peth_move.mean(axis=1)
    mean_combined = 0.5 * (mean_stat + mean_move)
    bc_effect = bc[effect_mask]

    for ui in range(n_units):
        peak_idx = np.argmax(mean_combined[ui, effect_mask])
        peak_t = bc_effect[peak_idx]
        peak_latencies[ui] = peak_t
        local_mask = (
            (bc >= peak_t - half_window_s)
            & (bc <= peak_t + half_window_s)
            & effect_mask
        )
        if not local_mask.any():
            local_mask = np.zeros_like(bc, dtype=bool)
            local_mask[np.argmin(np.abs(bc - peak_t))] = True
        stat_trials[ui] = peth_stat[ui][:, local_mask].mean(axis=1)
        move_trials[ui] = peth_move[ui][:, local_mask].mean(axis=1)

    delta = move_trials.mean(axis=1) - stat_trials.mean(axis=1)
    pvals = np.ones(len(delta), dtype=float)
    for ui in range(len(delta)):
        diff = move_trials[ui] - stat_trials[ui]
        if np.allclose(diff, 0.0):
            pvals[ui] = 1.0
            continue
        try:
            _, p = wilcoxon(
                move_trials[ui],
                stat_trials[ui],
                alternative="two-sided",
                zero_method="wilcox",
            )
        except ValueError:
            p = 1.0
        pvals[ui] = p if np.isfinite(p) else 1.0
    qvals = multipletests(pvals, alpha=QVAL_ALPHA, method="fdr_bh")[1]
    return delta, pvals, qvals, peak_latencies


def build_depth_bins(depth_um, width_um=DEPTH_BIN_WIDTH_UM):
    """Return fixed-width depth bin edges and centers."""
    lo = np.floor(depth_um.min() / width_um) * width_um
    hi = np.ceil(depth_um.max() / width_um) * width_um
    edges = np.arange(lo, hi + width_um, width_um)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return edges, centers


print("Loading data...")
st_per_unit, depth_per_unit = fetch_good_units_with_depth(subject, session)
align_ev = fetch_session_events(subject, session)
trial_df = fetch_trial_metadata(subject, session, align_ev)
unit_ids = list(st_per_unit.keys())
spike_times = list(st_per_unit.values())
depth_um = np.array([depth_per_unit[uid] for uid in unit_ids], dtype=float)

print(f"  Units: {len(unit_ids)}")
print(f"  Depth range: {depth_um.min():.1f}–{depth_um.max():.1f} µm")

trial_ts = build_trial_stim_classification(align_ev, trial_df)
anchors = extract_conditioned_stim_anchors(trial_ts)
paired_last_stat = anchors["paired_last_stationary"]
paired_first_move = anchors["paired_first_movement"]
print(f"  Trials with stat+move anchors: {len(paired_last_stat)}")

print("Computing PETHs...")
peth_stat_all, _, bc = compute_population_peth(
    spike_times, paired_last_stat, **PETH_KWARGS
)
peth_move_all, _, _ = compute_population_peth(
    spike_times, paired_first_move, **PETH_KWARGS
)
peth_stat_all *= PETH_SCALE_BACK
peth_move_all *= PETH_SCALE_BACK

pk_stat_all = resp_per_unit(peth_stat_all, bc)
pk_move_all = resp_per_unit(peth_move_all, bc)
snr_s = snr_per_unit(peth_stat_all, bc)
snr_m = snr_per_unit(peth_move_all, bc)
delta_move, _, qvals_move, peak_latencies = per_unit_move_vs_stat_stats(
    peth_stat_all, peth_move_all, bc
)

good_snr_both = (snr_s >= SNR_THRESHOLD) & (snr_m >= SNR_THRESHOLD)
sig_exc = (qvals_move < QVAL_ALPHA) & (delta_move > 0) & good_snr_both
sig_supp = (qvals_move < QVAL_ALPHA) & (delta_move < 0) & good_snr_both
nonsig = (~sig_exc) & (~sig_supp) & good_snr_both

print(
    f"  Class counts (SNR-both, q<{QVAL_ALPHA:.2f}): "
    f"exc={sig_exc.sum()}  supp={sig_supp.sum()}  no-effect={nonsig.sum()}"
)

edges, centers = build_depth_bins(depth_um, width_um=DEPTH_BIN_WIDTH_UM)
bin_idx = np.digitize(depth_um, edges) - 1
bin_idx = np.clip(bin_idx, 0, len(centers) - 1)

valid_mask = good_snr_both
n_bins = len(centers)
count_total = np.zeros(n_bins, dtype=int)
count_exc = np.zeros(n_bins, dtype=int)
count_supp = np.zeros(n_bins, dtype=int)
count_no = np.zeros(n_bins, dtype=int)
delta_med = np.full(n_bins, np.nan, dtype=float)
delta_q25 = np.full(n_bins, np.nan, dtype=float)
delta_q75 = np.full(n_bins, np.nan, dtype=float)

for bi in range(n_bins):
    in_bin = (bin_idx == bi) & valid_mask
    count_total[bi] = in_bin.sum()
    if count_total[bi] == 0:
        continue
    count_exc[bi] = (in_bin & sig_exc).sum()
    count_supp[bi] = (in_bin & sig_supp).sum()
    count_no[bi] = (in_bin & nonsig).sum()
    dvals = delta_move[in_bin]
    delta_med[bi] = np.median(dvals)
    delta_q25[bi] = np.percentile(dvals, 25)
    delta_q75[bi] = np.percentile(dvals, 75)

frac_exc = np.divide(count_exc, count_total, where=count_total > 0)
frac_supp = np.divide(count_supp, count_total, where=count_total > 0)
frac_no = np.divide(count_no, count_total, where=count_total > 0)

fig, axs = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)

ax = axs[0, 0]
ax.axhline(0, color="gray", linestyle="--", lw=0.8)
ax.scatter(
    depth_um[nonsig],
    delta_move[nonsig],
    s=18,
    alpha=0.55,
    color="0.45",
    label="No effect",
)
ax.scatter(
    depth_um[sig_exc],
    delta_move[sig_exc],
    s=18,
    alpha=0.70,
    color="tab:blue",
    label="Move-excited",
)
ax.scatter(
    depth_um[sig_supp],
    delta_move[sig_supp],
    s=18,
    alpha=0.70,
    color="tab:red",
    label="Move-suppressed",
)
ax.set_xlabel("Depth (µm)")
ax.set_ylabel("Δ move - stat (sp/s)")
ax.set_title("Per-unit locomotion effect vs depth")
ax.legend(frameon=False, fontsize=8, loc="best")

ax = axs[0, 1]
bar_w = DEPTH_BIN_WIDTH_UM * 0.85
ax.bar(centers, frac_supp, width=bar_w, color="tab:red", label="Move-suppressed")
ax.bar(centers, frac_no, width=bar_w, bottom=frac_supp, color="0.55", label="No effect")
ax.bar(
    centers,
    frac_exc,
    width=bar_w,
    bottom=frac_supp + frac_no,
    color="tab:blue",
    label="Move-excited",
)
ax.set_ylim(0, 1.02)
ax.set_xlabel("Depth bin center (µm)")
ax.set_ylabel("Fraction of analyzable units")
ax.set_title(f"Response-type composition by depth ({int(DEPTH_BIN_WIDTH_UM)} µm bins)")
ax.legend(frameon=False, fontsize=8, loc="best")

ax = axs[1, 0]
ax.bar(centers, count_total, width=bar_w, color="0.35")
ax.set_xlabel("Depth bin center (µm)")
ax.set_ylabel("Unit count")
ax.set_title("Analyzable units per depth bin")

ax = axs[1, 1]
valid_bins = np.isfinite(delta_med)
ax.axhline(0, color="gray", linestyle="--", lw=0.8)
ax.errorbar(
    centers[valid_bins],
    delta_med[valid_bins],
    yerr=np.vstack(
        [
            delta_med[valid_bins] - delta_q25[valid_bins],
            delta_q75[valid_bins] - delta_med[valid_bins],
        ]
    ),
    fmt="o-",
    color="k",
    markersize=4,
    lw=1.2,
)
ax.set_xlabel("Depth bin center (µm)")
ax.set_ylabel("Δ move - stat (sp/s)")
ax.set_title("Median locomotion effect by depth (IQR)")

fig.suptitle(
    (
        f"{subject} {session} — depth-binned locomotion modulation\n"
        f"(q<{QVAL_ALPHA:.2f}, SNR-both≥{SNR_THRESHOLD:.1f}, "
        f"bin={int(DEPTH_BIN_WIDTH_UM)} µm)"
    ),
    fontsize=11,
)
fig.savefig(OUT_PATH, bbox_inches="tight")
print(f"\nFigure saved: {OUT_PATH}")
