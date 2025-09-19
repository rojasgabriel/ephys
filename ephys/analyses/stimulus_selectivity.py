# %% Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as stats
from os.path import join as pjoin
import matplotlib.pyplot as plt
from spks.event_aligned import population_peth  # type: ignore
from spks.utils import alpha_function  # type: ignore

plt.rcParams["text.usetex"] = False
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.sans-serif"] = "Arial"
plt.rcParams["font.size"] = 12
plt.rcParams["figure.dpi"] = 100

# %% Load data
animal = "GRB006"  # example animal
session = "20240723_142451"  # example session

data_dir = "/Users/gabriel/data"
trial_ts = pd.read_pickle(
    pjoin(data_dir, animal, session, "pre_processed", "trial_ts.pkl")
)
spike_times_per_unit = np.load(
    pjoin(data_dir, animal, session, "pre_processed", "spike_times_per_unit.npy"),
    allow_pickle=True,
)


# %% Compute PETHs
trial_ts = trial_ts[
    trial_ts["stationary_stims"].apply(lambda x: len(x) > 0)
    & trial_ts["movement_stims"].apply(lambda x: len(x) > 0)
    & trial_ts["center_port_entries"].apply(lambda x: len(x) > 0)
].copy()

first_stim_ts = trial_ts["first_stim_ts"].to_numpy(dtype=float).copy()

binwidth_ms = 10
t_decay = 0.025
t_rise = 0.001
decay = t_decay / (binwidth_ms / 1000)
kern = alpha_function(
    int(decay * 15), t_rise=t_rise, t_decay=decay, srate=1.0 / (binwidth_ms / 1000)
)

first_stim_peth, bin_edges, event_index = population_peth(
    all_spike_times=spike_times_per_unit,
    alignment_times=first_stim_ts,
    pre_seconds=0.1,
    post_seconds=0.15,
    binwidth_ms=binwidth_ms,
    kernel=kern,
)


n_units, n_trials, n_timepoints = first_stim_peth.shape

# %% Selectivity: baseline vs response (simple + robust)
# Assumptions:
# - Baseline: -0.1 to 0.0 s relative to event
# - Response: +0.04 to +0.1 s relative to event
# - Test: paired within-trial baseline vs response per unit


def _benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    """Compute Benjamini-Hochberg FDR-adjusted q-values.
    Returns array of q-values in the original order.
    """
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = np.empty_like(order)
    ranked[order] = np.arange(1, n + 1)
    # Raw BH values
    q_raw = p * n / ranked
    # Enforce monotonicity
    q_sorted = q_raw[order]
    q_monotone = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q = np.empty_like(q_monotone)
    q[order] = np.clip(q_monotone, 0, 1)
    return q


def _paired_t_pvalue(x: np.ndarray, y: np.ndarray) -> float:
    res_t = stats.ttest_rel(x, y, alternative="two-sided", nan_policy="omit")
    return float(getattr(res_t, "pvalue", res_t[1]))


def _wilcoxon_pvalue(x: np.ndarray, y: np.ndarray) -> float:
    res_w = stats.wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
    return float(getattr(res_w, "pvalue", res_w[1]))


def compute_unit_selectivity(
    peth: np.ndarray,
    bin_edges: np.ndarray,
    base_window: tuple[float, float] = (-0.1, 0.0),
    resp_window: tuple[float, float] = (0.04, 0.10),
    test: str = "wilcoxon",
    alpha: float = 0.05,
):
    """Compute baseline vs response selectivity per unit.

    Args:
        peth: array (n_units, n_trials, n_timepoints) of firing rates (spk/s)
        bin_edges: array (n_timepoints + 1,) of bin edges in seconds relative to event
        base_window: (start, end) seconds for baseline
        resp_window: (start, end) seconds for response
        test: 'wilcoxon' (default) or 'ttest'
        alpha: FDR threshold

    Returns:
        results_df: DataFrame with per-unit stats
        masks: dict with boolean masks (excited, suppressed, selective_any)
    """
    n_units, n_trials, n_time = peth.shape
    t_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    base_mask = (t_centers >= base_window[0]) & (t_centers < base_window[1])
    resp_mask = (t_centers >= resp_window[0]) & (t_centers < resp_window[1])

    if not base_mask.any() or not resp_mask.any():
        raise ValueError("Baseline/response windows do not overlap available bins.")

    # Mean rate within window per trial
    base_rates = peth[:, :, base_mask].mean(axis=2)
    resp_rates = peth[:, :, resp_mask].mean(axis=2)

    pvals = np.ones(n_units, dtype=float)
    deltas = np.zeros(n_units, dtype=float)
    d_cohen = np.zeros(n_units, dtype=float)
    mean_base = base_rates.mean(axis=1)
    mean_resp = resp_rates.mean(axis=1)
    si = (mean_resp - mean_base) / (mean_resp + mean_base + 1e-9)  # selectivity index

    for u in range(n_units):
        x = resp_rates[u]
        y = base_rates[u]
        diff = x - y

        deltas[u] = diff.mean()
        sd = diff.std(ddof=1)
        d_cohen[u] = deltas[u] / sd if sd > 0 else 0.0

        # Guard against all-equal pairs (no variance / ties)
        if np.allclose(diff, 0):
            pvals[u] = 1.0
            continue

        if test == "ttest":
            pvals[u] = _paired_t_pvalue(x, y)
        elif test == "wilcoxon":
            try:
                pvals[u] = _wilcoxon_pvalue(x, y)
            except ValueError:
                # Fallback when all differences are zero after zero_method filtering
                pvals[u] = 1.0
        else:
            raise ValueError("test must be 'wilcoxon' or 'ttest'")

    qvals = _benjamini_hochberg(pvals)

    excited = (qvals < alpha) & (deltas > 0)
    suppressed = (qvals < alpha) & (deltas < 0)
    selective_any = excited | suppressed

    results_df = pd.DataFrame(
        {
            "unit": np.arange(n_units),
            "mean_base": mean_base,
            "mean_resp": mean_resp,
            "delta": deltas,
            "cohen_d": d_cohen,
            "si": si,
            "p": pvals,
            "q": qvals,
            "excited": excited,
            "suppressed": suppressed,
            "selective": selective_any,
        }
    )

    return results_df, {
        "excited": excited,
        "suppressed": suppressed,
        "selective": selective_any,
    }


# Compute selectivity with default windows and Wilcoxon test
results_df, masks = compute_unit_selectivity(
    first_stim_peth,
    bin_edges,
    base_window=(-0.1, 0.0),
    resp_window=(0.04, 0.10),
    test="wilcoxon",
    alpha=0.05,
)

print(
    f"Units selective for first stimulus: {masks['selective'].sum()} / {n_units} "
    f"({masks['excited'].sum()} excited, {masks['suppressed'].sum()} suppressed)"
)

# %% Optional quick looks
fig, ax = plt.subplots(1, 2, figsize=(8, 3))
sns.histplot(results_df["delta"], bins=31, ax=ax[0], color="steelblue")
ax[0].set(
    title="Mean(resp - base) across units", xlabel="Delta FR (spk/s)", ylabel="Count"
)

sns.scatterplot(
    data=results_df,
    x="cohen_d",
    y=-np.log10(np.clip(results_df["q"], 1e-12, 1.0)),
    hue="selective",
    palette={True: "crimson", False: "gray"},
    ax=ax[1],
    s=20,
)
ax[1].set(title="Volcano (effect vs -log10 q)", xlabel="Cohen's d", ylabel="-log10(q)")
ax[1].legend(title="Selective", loc="best")
plt.tight_layout()


# %% Example raster/PSTH for a top selective unit (if available)
if masks["selective"].any():
    # Choose the most selective by q then |delta|
    top_idx = results_df.sort_values(["q", "delta"], ascending=[True, False]).iloc[0][
        "unit"
    ]

    t_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    mean_psth = first_stim_peth[top_idx].mean(axis=0)

    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    ax.plot(t_centers, mean_psth, color="black")
    ax.axvspan(-0.1, 0.0, color="tab:blue", alpha=0.15, label="Baseline")
    ax.axvspan(0.04, 0.10, color="tab:orange", alpha=0.15, label="Response")
    ax.axvline(0, color="red", linestyle="--", linewidth=1, label="Stimulus")
    ax.set(
        xlabel="Time (s)",
        ylabel="Firing rate (spk/s)",
        title=f"Unit {int(top_idx)}: First-stim PSTH",
    )
    ax.legend(loc="best")
    plt.tight_layout()

# %%
