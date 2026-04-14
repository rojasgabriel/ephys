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
    1. Per-trial sync residual at anchor points (t_sync ↔ trial_start)
    2. Distribution of per-trial sync residuals
    3. Expected vs observed stim count per trial
    4. stim_rate_vision distribution per trial

  Row 2 — example single units (last stationary vs first movement):
    Col 0: best locomotion-excited unit
    Col 1: best locomotion-suppressed unit
    Col 2: best no-effect unit
    Col 3: per-unit scatter (all units)

  Row 3 — rate split (low vs high stim_rate_vision):
    Col 0-1: low-rate trials scatter (stat vs move)
    Col 2-3: high-rate trials scatter (stat vs move)
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import sem, wilcoxon
from scipy.signal import find_peaks
from statsmodels.stats.multitest import multipletests

from ephys.src.utils.utils_IO import (
    fetch_good_units,
    fetch_session_events,
    fetch_trial_metadata,
)
from ephys.src.utils.utils_analysis import (
    build_trial_stim_classification,
    compute_unit_selectivity,
    compute_population_peth,
    extract_conditioned_stim_anchors,
)

subject = "GRB058"
session = "20260312_134952"
OUT_PATH = f"/Users/gabriel/lib/ephys/figures/locomotion_stat_vs_move_{subject}_{session[:8]}.pdf"

# No kernel smoothing — raw 10 ms bins.  Pre-stim baseline kept short (40 ms)
# so the previous stim's response (≥~55 ms away at 18 Hz) doesn't bleed in.
PETH_KWARGS = dict(
    pre_seconds=0.04,
    post_seconds=0.15,
    binwidth_ms=10,
)
# In this environment, spks.population_peth already returns rate-like values.
# compute_population_peth applies an extra 1/binwidth scaling, so convert back
# to readable sp/s for plotting in this script.
PETH_SCALE_BACK = PETH_KWARGS["binwidth_ms"] / 1000.0
# Response characterisation window: 40–100 ms post-stim
# (responses begin after ~30 ms; upper bound keeps ISI-contamination out).
RESP_WINDOW = (0.04, 0.10)
# Locomotion modulation classification window:
# use trial-wise PEAK in 0–120 ms to capture transient responses (e.g. early
# stationary peaks that can disappear during movement).
EFFECT_WINDOW = (0.0, 0.12)
# Compare conditions using mean activity around each unit's own peak latency.
PEAK_HALF_WINDOW_S = 0.015
# Prefer canonical visual latencies for the excited exemplar panel.
CANONICAL_LATENCY_WINDOW = (0.015, 0.070)

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
)
peth_all *= PETH_SCALE_BACK
_, masks = compute_unit_selectivity(
    peth_all,
    bin_edges,
    unit_ids=unit_ids,
    base_window=(-0.04, 0.0),
    resp_window=RESP_WINDOW,
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
valid_trial_idx = np.where(valid)[0]  # original trial indices that survived

n_invalid_sync = (~valid).sum()
n_dup_bpod = (np.diff(bpod_sync) == 0).sum()
n_dup_obx = (np.diff(obx_sync) == 0).sum()
print(
    f"  Sync QC: invalid={n_invalid_sync}  "
    f"duplicate bpod={n_dup_bpod}  duplicate obx={n_dup_obx}"
)

global_poly = np.polyfit(bpod_sync, obx_sync, deg=1)
global_pred = np.polyval(global_poly, bpod_sync)
sync_resid_ms = (obx_sync - global_pred) * 1e3
sync_r2 = np.corrcoef(obx_sync, global_pred)[0, 1] ** 2
sync_sigma_ms = np.std(sync_resid_ms)
sync_p95_ms = np.percentile(np.abs(sync_resid_ms), 95)
print(
    f"  Global sync map (OBX = a*Bpod + b): a={global_poly[0]:.10f}, "
    f"b={global_poly[1]:.6f}, R²={sync_r2:.6f}"
)
print(
    f"  Per-trial anchor residual: mean={np.mean(sync_resid_ms):+.2f} ms  "
    f"std={sync_sigma_ms:.2f} ms  95%|·|={sync_p95_ms:.2f} ms"
)
print(
    f"  Estimated mapping uncertainty for Bpod-only timestamps (e.g., t_react): "
    f"~{sync_sigma_ms:.2f} ms (1σ), ~{2 * sync_sigma_ms:.2f} ms (~95% Gaussian)"
)

# Local clock-rate consistency: ratio of consecutive ITI gaps.
# Filter out pairs where bpod_iti < 0.5 s — these arise when adjacent valid
# sync points span many skipped (NaN) trials, collapsing the bpod gap while
# the OBX gap remains large, or from duplicate timestamps.
iti_obx = np.diff(obx_sync)
iti_bpod = np.diff(bpod_sync)
iti_trial_idx = valid_trial_idx[:-1]  # trial index of the FIRST sync point in each pair
reasonable = iti_bpod > 0.5
slope_local = iti_obx[reasonable] / iti_bpod[reasonable]
slope_trial_idx = iti_trial_idx[reasonable]
n_outlier_pairs = (~reasonable).sum()
print(
    f"  ITI ratio: {reasonable.sum()} pairs kept, {n_outlier_pairs} skipped "
    f"(bpod_iti < 0.5 s)"
)

# Diagnose trial pacing geometry: compare bpod ITI distribution before/after trial 100
cutoff = 100
print(
    f"  bpod ITI  <trial {cutoff}: "
    f"median={np.median(iti_bpod[iti_trial_idx < cutoff]):.2f}s  "
    f"mean={np.mean(iti_bpod[iti_trial_idx < cutoff]):.2f}s"
)
print(
    f"  bpod ITI ≥trial {cutoff}: "
    f"median={np.median(iti_bpod[iti_trial_idx >= cutoff]):.2f}s  "
    f"mean={np.mean(iti_bpod[iti_trial_idx >= cutoff]):.2f}s"
)
ratio_p5, ratio_p95 = np.percentile(slope_local, [5, 95])
n_outside = ((slope_local < ratio_p5) | (slope_local > ratio_p95)).sum()
print(
    f"  ITI ratio range (5th–95th pct): [{ratio_p5:.4f}, {ratio_p95:.4f}]  "
    f"({n_outside} pts outside)"
)

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

# ── Build PETHs aligned to paired last stationary / first movement ──────────
anchors = extract_conditioned_stim_anchors(trial_ts)
paired_last_stat = anchors["paired_last_stationary"]
paired_first_move = anchors["paired_first_movement"]

print("Computing PETHs (all units)...")
peth_stat_all, _, bc = compute_population_peth(
    spike_times, paired_last_stat, **PETH_KWARGS
)
peth_move_all, _, _ = compute_population_peth(
    spike_times, paired_first_move, **PETH_KWARGS
)
peth_stat_all *= PETH_SCALE_BACK
peth_move_all *= PETH_SCALE_BACK


def resp_per_unit(peth, bc, window=RESP_WINDOW):
    """Mean firing rate in the response window, averaged across trials."""
    mask = (bc >= window[0]) & (bc < window[1])
    return peth[:, :, mask].mean(axis=(1, 2))


def wilcox_str(a, b):
    _, p = wilcoxon(a, b)
    pct = 100 * (b > a).mean()
    return f"{pct:.0f}% above diag  p={p:.3f}"


pk_stat_all = resp_per_unit(peth_stat_all, bc)
pk_move_all = resp_per_unit(peth_move_all, bc)
print(f"\nAll units  ({len(unit_ids)}):  {wilcox_str(pk_stat_all, pk_move_all)}")

# ── Example unit selection ───────────────────────────────────────────────────
# Normalised (move - stat) score and per-unit SNR.
# SNR = (mean response in window) / (SEM across trials in window).
# Filter to units with SNR ≥ 3 in at least one condition, then pick
# representatives near the top/bottom/middle of the score distribution
# (not the extremes, which tend to be the noisiest).
avg_resp = (pk_move_all + pk_stat_all) / 2 + 1e-3
norm_diff = (pk_move_all - pk_stat_all) / avg_resp


def per_unit_move_vs_stat_stats(
    peth_stat,
    peth_move,
    bc,
    window=EFFECT_WINDOW,
    half_window_s=PEAK_HALF_WINDOW_S,
):
    """Per-unit paired stats across trials for movement vs stationary response.

    Statistic = trial-wise mean firing rate around each unit's peak latency.
    """
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
    qvals = multipletests(pvals, alpha=0.05, method="fdr_bh")[1]
    return delta, pvals, qvals, peak_latencies


def snr_per_unit(peth, bc, window=RESP_WINDOW):
    mask = (bc >= window[0]) & (bc < window[1])
    resp = peth[:, :, mask].mean(axis=2)  # (n_units, n_trials)
    unit_mean = resp.mean(axis=1)
    unit_sem = resp.std(axis=1) / np.sqrt(resp.shape[1])
    return unit_mean / (unit_sem + 1e-3)


snr_s = snr_per_unit(peth_stat_all, bc)
snr_m = snr_per_unit(peth_move_all, bc)
delta_move, pvals_move, qvals_move, peak_latencies = per_unit_move_vs_stat_stats(
    peth_stat_all, peth_move_all, bc
)
# Require good SNR in BOTH conditions so both stat and move show a real
# response — this ensures both curves are clean, not just one of them.
good_snr_both = (snr_s >= 3.0) & (snr_m >= 3.0)
good_idx = np.where(good_snr_both)[0]
print(
    f"  Units with SNR ≥ 3 in both conditions: {good_snr_both.sum()} / {len(unit_ids)}"
)
sig_exc = (qvals_move < 0.05) & (delta_move > 0) & good_snr_both
sig_supp = (qvals_move < 0.05) & (delta_move < 0) & good_snr_both
nonsig = (~sig_exc) & (~sig_supp) & good_snr_both
print(
    f"  Move-vs-stat peak-centered classes ({EFFECT_WINDOW[0]:.2f}-{EFFECT_WINDOW[1]:.2f}s, "
    f"FDR<0.05, SNR-both): "
    f"exc={sig_exc.sum()}  supp={sig_supp.sum()}  no-effect={nonsig.sum()}"
)

nd_good = norm_diff[good_idx]


def peak_count_in_window(peth_unit, bc, window=(0.0, 0.12)):
    """Number of peaks in the mean PSTH within the response window."""
    mean_psth = peth_unit.mean(axis=0)
    base = mean_psth[(bc >= -0.04) & (bc < 0.0)].mean()
    mask = (bc >= window[0]) & (bc < window[1])
    resp = mean_psth[mask]
    threshold = base + (resp.max() - base) * 0.25
    peaks, _ = find_peaks(resp, height=threshold, distance=2)
    return len(peaks)


def is_single_peak(ui):
    return (
        peak_count_in_window(peth_stat_all[ui], bc) == 1
        and peak_count_in_window(peth_move_all[ui], bc) == 1
    )


# Excited: prefer canonical-latency exemplars with movement-dominant peaks.
base_mask = (bc >= -0.04) & (bc < 0.0)
effect_mask = (bc >= EFFECT_WINDOW[0]) & (bc < EFFECT_WINDOW[1])


def movement_peak_gain(ui):
    mean_stat = peth_stat_all[ui].mean(axis=0)
    mean_move = peth_move_all[ui].mean(axis=0)
    stat_peak = (mean_stat[effect_mask] - mean_stat[base_mask].mean()).max()
    move_peak = (mean_move[effect_mask] - mean_move[base_mask].mean()).max()
    return move_peak - stat_peak


def single_peak_signature(ui):
    """Return robust single-peak diagnostics on baseline-subtracted mean PSTHs."""
    mean_stat = peth_stat_all[ui].mean(axis=0)
    mean_move = peth_move_all[ui].mean(axis=0)
    resp_stat = mean_stat[effect_mask] - mean_stat[base_mask].mean()
    resp_move = mean_move[effect_mask] - mean_move[base_mask].mean()
    t_resp = bc[effect_mask]
    prom_s = max(2.0, 0.35 * max(1e-6, float(resp_stat.max())))
    prom_m = max(2.0, 0.35 * max(1e-6, float(resp_move.max())))
    peaks_s, _ = find_peaks(resp_stat, prominence=prom_s, distance=2)
    peaks_m, _ = find_peaks(resp_move, prominence=prom_m, distance=2)
    if len(peaks_s):
        p_s = peaks_s[np.argmax(resp_stat[peaks_s])]
        lat_s = t_resp[p_s]
        amp_s = resp_stat[p_s]
    else:
        lat_s = np.nan
        amp_s = -np.inf
    if len(peaks_m):
        p_m = peaks_m[np.argmax(resp_move[peaks_m])]
        lat_m = t_resp[p_m]
        amp_m = resp_move[p_m]
    else:
        lat_m = np.nan
        amp_m = -np.inf
    return len(peaks_s), len(peaks_m), lat_s, lat_m, amp_s, amp_m


def transient_signature(ui):
    """Transientness of movement response in EFFECT_WINDOW."""
    mean_move = peth_move_all[ui].mean(axis=0)
    resp = mean_move[effect_mask] - mean_move[base_mask].mean()
    t_resp = bc[effect_mask]
    if resp.max() <= 0:
        return np.nan, np.nan, np.nan, 0, np.nan, np.nan
    prom = max(2.0, 0.35 * float(resp.max()))
    peaks, _ = find_peaks(resp, prominence=prom, distance=2)
    p_idx = int(np.argmax(resp))
    peak_t = float(t_resp[p_idx])
    half_h = 0.5 * float(resp[p_idx])
    above = np.where(resp >= half_h)[0]
    width_ms = (
        float((t_resp[above[-1]] - t_resp[above[0]]) * 1000) if above.size else np.nan
    )
    tail_mask = t_resp >= (peak_t + 0.03)
    tail_mean = float(resp[tail_mask].mean()) if tail_mask.any() else np.nan
    tail_ratio = (
        tail_mean / float(resp[p_idx])
        if np.isfinite(tail_mean) and resp[p_idx] > 0
        else np.nan
    )
    smooth_tv = float(np.abs(np.diff(resp)).mean())
    smooth_curv = float(np.abs(np.diff(resp, n=2)).mean()) if resp.size >= 3 else np.nan
    return peak_t, width_ms, tail_ratio, int(len(peaks)), smooth_tv, smooth_curv


exc_cands = np.where(sig_exc)[0]
exc_sp = np.array([ui for ui in exc_cands if is_single_peak(ui)])
strict_exc = []
single_peak_gain_exc = []
canonical_exc = []
canonical_scores = []
transient_exc = []
transient_scores = []
transient_pref_exc = []
transient_pref_scores = []
for ui in exc_cands:
    nps, npm, lat_s, lat_m, amp_s, amp_m = single_peak_signature(ui)
    peak_t_m, width_ms, tail_ratio, n_peaks_m, smooth_tv, smooth_curv = (
        transient_signature(ui)
    )
    if nps == 1 and npm == 1 and amp_m > amp_s:
        single_peak_gain_exc.append(ui)
    if (
        np.isfinite(lat_s)
        and np.isfinite(lat_m)
        and CANONICAL_LATENCY_WINDOW[0] <= lat_s <= CANONICAL_LATENCY_WINDOW[1]
        and CANONICAL_LATENCY_WINDOW[0] <= lat_m <= CANONICAL_LATENCY_WINDOW[1]
        and amp_m > amp_s
    ):
        complexity_penalty = abs(nps - 1) + abs(npm - 1)
        latency_penalty = abs(lat_m - lat_s)
        score = delta_move[ui] - 4.0 * complexity_penalty - 20.0 * latency_penalty
        canonical_exc.append(ui)
        canonical_scores.append(score)
    if (
        nps == 1
        and npm == 1
        and np.isfinite(lat_s)
        and np.isfinite(lat_m)
        and CANONICAL_LATENCY_WINDOW[0] <= lat_s <= CANONICAL_LATENCY_WINDOW[1]
        and CANONICAL_LATENCY_WINDOW[0] <= lat_m <= CANONICAL_LATENCY_WINDOW[1]
        and abs(lat_m - lat_s) <= 0.03
        and amp_m > amp_s
    ):
        strict_exc.append(ui)
    if (
        nps == 1
        and npm == 1
        and amp_m > amp_s
        and np.isfinite(peak_t_m)
        and CANONICAL_LATENCY_WINDOW[0] <= peak_t_m <= 0.080
        and np.isfinite(width_ms)
        and width_ms <= 45.0
        and np.isfinite(tail_ratio)
        and tail_ratio <= 0.25
        and n_peaks_m == 1
    ):
        transient_exc.append(ui)
        transient_scores.append(
            delta_move[ui]
            - 0.08 * width_ms
            - 4.0 * max(tail_ratio, 0.0)
            - 0.8 * smooth_tv
            - 0.2 * (smooth_curv if np.isfinite(smooth_curv) else 0.0)
        )
    if (
        npm <= 2
        and nps <= 2
        and amp_m > amp_s
        and np.isfinite(peak_t_m)
        and CANONICAL_LATENCY_WINDOW[0] <= peak_t_m <= 0.080
        and np.isfinite(width_ms)
        and width_ms <= 45.0
        and np.isfinite(tail_ratio)
        and tail_ratio <= 0.25
        and n_peaks_m <= 2
    ):
        transient_pref_exc.append(ui)
        transient_pref_scores.append(
            delta_move[ui]
            - 0.10 * width_ms
            - 5.0 * max(tail_ratio, 0.0)
            - 1.2 * smooth_tv
            - 0.4 * (smooth_curv if np.isfinite(smooth_curv) else 0.0)
            - 1.5 * max(0, npm - 1)
        )
strict_exc = np.array(strict_exc, dtype=int)
single_peak_gain_exc = np.array(single_peak_gain_exc, dtype=int)
canonical_exc = np.array(canonical_exc, dtype=int)
canonical_scores = np.array(canonical_scores, dtype=float)
transient_exc = np.array(transient_exc, dtype=int)
transient_scores = np.array(transient_scores, dtype=float)
transient_pref_exc = np.array(transient_pref_exc, dtype=int)
transient_pref_scores = np.array(transient_pref_scores, dtype=float)
if transient_exc.size:
    ex_excited_idx = transient_exc[np.argmax(transient_scores)]
elif transient_pref_exc.size:
    ex_excited_idx = transient_pref_exc[np.argmax(transient_pref_scores)]
elif strict_exc.size:
    ex_excited_idx = strict_exc[np.argmax(delta_move[strict_exc])]
elif canonical_exc.size:
    ex_excited_idx = canonical_exc[np.argmax(canonical_scores)]
elif single_peak_gain_exc.size:
    ex_excited_idx = single_peak_gain_exc[np.argmax(delta_move[single_peak_gain_exc])]
elif exc_sp.size:
    ex_excited_idx = exc_sp[np.argmax(delta_move[exc_sp])]
elif exc_cands.size:
    ex_excited_idx = exc_cands[np.argmax(delta_move[exc_cands])]
else:
    ex_excited_idx = good_idx[np.argmax(delta_move[good_idx])]  # fallback

# Suppressed: strongest negative modulation among significantly suppressed units.
supp_cands = np.where(sig_supp)[0]
supp_sp = np.array([ui for ui in supp_cands if is_single_peak(ui)])
if supp_sp.size:
    ex_suppressed_idx = supp_sp[np.argmin(delta_move[supp_sp])]
elif supp_cands.size:
    ex_suppressed_idx = supp_cands[np.argmin(delta_move[supp_cands])]
else:
    ex_suppressed_idx = good_idx[np.argmin(delta_move[good_idx])]  # fallback

# No effect: nearest-zero modulation among non-significant units.
noeff_cands = np.where(nonsig)[0]
noeff_sp = np.array([ui for ui in noeff_cands if is_single_peak(ui)])
if noeff_sp.size:
    ex_noeffect_idx = noeff_sp[np.argmin(np.abs(delta_move[noeff_sp]))]
elif noeff_cands.size:
    ex_noeffect_idx = noeff_cands[np.argmin(np.abs(delta_move[noeff_cands]))]
else:
    ex_noeffect_idx = good_idx[np.argmin(np.abs(delta_move[good_idx]))]  # fallback

ex_indices = [ex_excited_idx, ex_suppressed_idx, ex_noeffect_idx]
ex_labels = ["Loco-excited", "Loco-suppressed", "No effect"]
ex_colors = ["tab:blue", "tab:red", "tab:gray"]

if 664 in unit_ids:
    u664 = unit_ids.index(664)
    cls_664 = (
        "excited" if sig_exc[u664] else "suppressed" if sig_supp[u664] else "no-effect"
    )
    print(
        f"  Unit 664: class={cls_664}  Δ(move-stat)={delta_move[u664]:+.2f} sp/s  "
        f"q={qvals_move[u664]:.3f}  peak_t={peak_latencies[u664]:.3f}s  "
        f"normΔ={norm_diff[u664]:+.2f}"
    )
print(
    "  Example units: "
    f"exc={unit_ids[ex_excited_idx]}(t*={peak_latencies[ex_excited_idx]:.3f}s)  "
    f"supp={unit_ids[ex_suppressed_idx]}  "
    f"no-effect={unit_ids[ex_noeffect_idx]}"
)

# ── Rate split ───────────────────────────────────────────────────────────────
# Get stim rate for each trial in trial_ts
trial_rates = (
    trial_df["stim_rate_vision"].iloc[trial_ts["trial_idx"].values].values.astype(float)
)
valid_rate_mask = np.isfinite(trial_rates)
rate_median = np.nanmedian(trial_rates)
low_mask = valid_rate_mask & (trial_rates < rate_median)
high_mask = valid_rate_mask & (trial_rates >= rate_median)

print(
    f"\nRate split: median = {rate_median:.1f} Hz  "
    f"low = {low_mask.sum()} trials  high = {high_mask.sum()} trials"
)

print("Computing PETHs (rate split)...")
peth_stat_low, _, _ = compute_population_peth(
    spike_times, paired_last_stat[low_mask], **PETH_KWARGS
)
peth_move_low, _, _ = compute_population_peth(
    spike_times, paired_first_move[low_mask], **PETH_KWARGS
)
peth_stat_high, _, _ = compute_population_peth(
    spike_times, paired_last_stat[high_mask], **PETH_KWARGS
)
peth_move_high, _, _ = compute_population_peth(
    spike_times, paired_first_move[high_mask], **PETH_KWARGS
)
peth_stat_low *= PETH_SCALE_BACK
peth_move_low *= PETH_SCALE_BACK
peth_stat_high *= PETH_SCALE_BACK
peth_move_high *= PETH_SCALE_BACK

pk_stat_low = resp_per_unit(peth_stat_low, bc)
pk_move_low = resp_per_unit(peth_move_low, bc)
pk_stat_high = resp_per_unit(peth_stat_high, bc)
pk_move_high = resp_per_unit(peth_move_high, bc)

print(f"  Low rate:  {wilcox_str(pk_stat_low, pk_move_low)}")
print(f"  High rate: {wilcox_str(pk_stat_high, pk_move_high)}")

# ── Figure ──────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 18))
gs = fig.add_gridspec(4, 4, hspace=0.55, wspace=0.35)
TITLE_KW = dict(fontsize=9, pad=3)

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
ax.set_title("Stim count", **TITLE_KW)
ax.legend(fontsize=7, frameon=False)

ax = fig.add_subplot(gs[0, 1])
ax.hist(cp_hold, bins=30, color="mediumpurple", edgecolor="white", lw=0.5)
ax.axvline(1.0, color="k", linestyle="--", lw=0.8, label="1 s (1 train)")
ax.set_xlabel("Time in center port (s)")
ax.set_ylabel("Trials")
ax.set_title(f"Center hold\nmax {cp_hold.max():.2f} s", **TITLE_KW)
ax.legend(fontsize=7, frameon=False)

ax = fig.add_subplot(gs[0, 2])
ax.hist(travel, bins=25, color="seagreen", edgecolor="white", lw=0.5)
ax.set_xlabel("Travel time (s)")
ax.set_ylabel("Trials")
ax.set_title("Travel time", **TITLE_KW)

ax = fig.add_subplot(gs[0, 3])
mx = int(n_reentries.max()) + 2
ax.hist(
    n_reentries, bins=range(1, mx + 1), color="indianred", edgecolor="white", lw=0.5
)
ax.set_xlabel("cp entries per trial")
ax.set_ylabel("Trials")
ax.set_title(f"Re-entries\n{(n_reentries > 1).sum()}/{n_trials} >1", **TITLE_KW)

# Row 1: clock + behavioral validation ──────────────────────────────────────
ax = fig.add_subplot(gs[1, 0])
# Direct per-trial anchor fit residual: this is the relevant sync validation
# for mapping any Bpod-only timestamp (e.g., t_react) into OBX time.
ax.axhline(0, color="gray", linestyle="--", lw=0.8)
ax.axhline(+2 * sync_sigma_ms, color="crimson", linestyle="--", lw=0.9, alpha=0.7)
ax.axhline(-2 * sync_sigma_ms, color="crimson", linestyle="--", lw=0.9, alpha=0.7)
ax.plot(
    valid_trial_idx,
    sync_resid_ms,
    "o",
    color="k",
    markersize=2,
    alpha=0.55,
    label=f"residual (±2σ = {2 * sync_sigma_ms:.2f} ms)",
)
ax.set_xlabel("Trial index")
ax.set_ylabel("Anchor residual (ms)")
ax.set_title(
    f"Sync residuals\nσ={sync_sigma_ms:.2f} ms, p95={sync_p95_ms:.2f} ms",
    **TITLE_KW,
)
ax.legend(fontsize=6, frameon=False, loc="upper right")

ax = fig.add_subplot(gs[1, 1])
ax.hist(sync_resid_ms, bins=30, color="slategray", edgecolor="white", lw=0.5)
ax.axvline(0, color="k", linestyle="--", lw=0.8)
ax.axvline(+2 * sync_sigma_ms, color="crimson", linestyle="--", lw=0.9, alpha=0.7)
ax.axvline(-2 * sync_sigma_ms, color="crimson", linestyle="--", lw=0.9, alpha=0.7)
ax.set_xlabel("Anchor residual (ms)")
ax.set_ylabel("Sync points")
ax.set_title(
    f"Sync residual histogram\nσ={sync_sigma_ms:.2f} ms, max={np.max(np.abs(sync_resid_ms)):.2f} ms",
    **TITLE_KW,
)

ax = fig.add_subplot(gs[1, 2])
mx = max(obs_counts.max(), exp_counts.max()) + 2
ax.scatter(exp_counts, obs_counts, s=14, alpha=0.4, color="k")
ax.plot([0, mx], [0, mx], "k--", alpha=0.4, lw=0.8)
ax.set_xlabel("Expected (rate × duration)")
ax.set_ylabel("Observed (OBX 15 ms)")
ax.set_title(
    f"Behavioral vs OBX counts\n|diff|≤1: {(np.abs(mismatch) <= 1).mean() * 100:.0f}%",
    **TITLE_KW,
)
ax.set_xlim(0, mx)
ax.set_ylim(0, mx)
ax.set_aspect("equal")

ax = fig.add_subplot(gs[1, 3])
rates = trial_df["stim_rate_vision"].dropna().to_numpy()
ax.hist(rates, bins=20, color="darkkhaki", edgecolor="white", lw=0.5)
ax.axvline(
    rate_median,
    color="crimson",
    linestyle="--",
    lw=0.8,
    label=f"median={rate_median:.1f} Hz",
)
ax.set_xlabel("stim_rate_vision (Hz)")
ax.set_ylabel("Trials")
ax.set_title(f"Stim rate\nmin ISI ≈ {1000 / rates.max():.0f} ms", **TITLE_KW)
ax.legend(fontsize=7, frameon=False)

# Row 2: example single units (cols 0–2; col 3 unused) ──────────────────────
for col, (ui, label) in enumerate(zip(ex_indices, ex_labels)):
    ax = fig.add_subplot(gs[2, col])
    ps = peth_stat_all[ui]  # (n_trials, n_timebins)
    pm = peth_move_all[ui]
    ms = ps.mean(axis=0)
    ss = sem(ps, axis=0)
    mm = pm.mean(axis=0)
    sm = sem(pm, axis=0)
    ax.plot(
        bc,
        ms,
        color="steelblue",
        lw=1.6,
        label=f"last stationary (n={len(paired_last_stat)})",
    )
    ax.fill_between(bc, ms - ss, ms + ss, alpha=0.25, color="steelblue")
    ax.plot(
        bc,
        mm,
        color="darkorange",
        lw=1.6,
        label=f"first movement (n={len(paired_first_move)})",
    )
    ax.fill_between(bc, mm - sm, mm + sm, alpha=0.25, color="darkorange")
    ax.axvline(0, color="gray", linestyle="--", lw=0.8)
    ax.set_xlabel("Time from stim onset (s)")
    ax.set_ylabel("sp/s")
    uid = unit_ids[ui]
    nd = norm_diff[ui]
    qv = qvals_move[ui]
    dm = delta_move[ui]
    pt = peak_latencies[ui]
    ax.set_title(
        f"{label}  unit {uid}\nΔsp/s={dm:+.1f}, q={qv:.3f}, t*={pt * 1e3:.0f} ms",
        **TITLE_KW,
    )
    ax.legend(fontsize=6, frameon=False)


def scatter_log(ax, pk_s, pk_m, title):
    """Scatter plot on log-log axes with identity line."""
    all_pos = np.concatenate([pk_s, pk_m])
    all_pos = all_pos[all_pos > 0]
    lim_lo = max(1.0, np.percentile(all_pos, 2)) if all_pos.size else 1.0
    lim_hi = np.percentile(all_pos, 99) * 1.5 if all_pos.size else 100.0
    ax.scatter(pk_s, pk_m, color="k", s=20, alpha=0.45, zorder=3)
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", alpha=0.4, lw=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.set_aspect("equal")
    ax.set_xlabel("Stationary (sp/s)")
    ax.set_ylabel("Movement (sp/s)")
    _, p = wilcoxon(pk_s, pk_m)
    pct = 100 * (pk_m > pk_s).mean()
    ax.set_title(f"{title}\n{pct:.0f}% above, p={p:.3f}", **TITLE_KW)


# Row 3: all 3 scatters on one row (cols 0–2; col 3 unused) ─────────────────
scatter_log(
    fig.add_subplot(gs[3, 0]),
    pk_stat_all,
    pk_move_all,
    f"All units (n={len(unit_ids)})",
)
scatter_log(
    fig.add_subplot(gs[3, 1]),
    pk_stat_low,
    pk_move_low,
    f"Low rate (<{rate_median:.1f} Hz, n={low_mask.sum()} trials)",
)
scatter_log(
    fig.add_subplot(gs[3, 2]),
    pk_stat_high,
    pk_move_high,
    f"High rate (≥{rate_median:.1f} Hz, n={high_mask.sum()} trials)",
)

fig.suptitle(
    f"{subject}  {session}  —  first stationary vs. first movement",
    fontsize=10,
    y=1.005,
)

fig.savefig(OUT_PATH, bbox_inches="tight")
print(f"\nFigure saved: {OUT_PATH}")
