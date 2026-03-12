import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.signal import find_peaks
from statsmodels.stats.multitest import multipletests
from typing import Optional, Sequence
from spks.event_aligned import population_peth  # type: ignore


def compute_population_peth(
    spike_times_per_unit: Sequence[np.ndarray],
    alignment_times: np.ndarray,
    pre_seconds: float = 0.1,
    post_seconds: float = 0.15,
    binwidth_ms: int = 10,
    t_rise: Optional[float] = None,
    t_decay: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute population PETH aligned to event times.

    Args:
        spike_times_per_unit: list/sequence of arrays, one per unit (seconds).
        alignment_times: array of event times to align to (seconds).
        pre_seconds: time before event.
        post_seconds: time after event.
        binwidth_ms: bin width in milliseconds.
        t_rise: alpha-function rise time (seconds). If None, no smoothing.
        t_decay: alpha-function decay time (seconds). If None, no smoothing.

    Returns:
        peth: array (n_units, n_trials, n_timebins)
        bin_edges: array (n_timebins + 1,) in seconds relative to event
        bin_centers: array (n_timebins,) in seconds relative to event
    """
    kernel = None
    if t_rise is not None and t_decay is not None:
        from spks.utils import alpha_function  # type: ignore

        decay_bins = t_decay / (binwidth_ms / 1000)
        kernel = alpha_function(
            int(decay_bins * 15),
            t_rise=t_rise,
            t_decay=decay_bins,
            srate=1.0 / (binwidth_ms / 1000),
        )

    peth, bin_edges, event_index = population_peth(
        all_spike_times=spike_times_per_unit,
        alignment_times=alignment_times,
        pre_seconds=pre_seconds,
        post_seconds=post_seconds,
        binwidth_ms=binwidth_ms,
        kernel=kernel,
    )
    # Convert spike counts per bin to firing rate (sp/s)
    peth = peth / (binwidth_ms / 1000.0)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    return peth, bin_edges, bin_centers


def compute_unit_selectivity(
    peth: np.ndarray,
    bin_edges: np.ndarray,
    unit_ids: Sequence,
    base_window: tuple[float, float] = (-0.1, 0.0),
    resp_window: tuple[float, float] = (0.04, 0.10),
    test: str = "wilcoxon",
    correction: str = "fdr_bh",
    alpha: float = 0.05,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Compute baseline vs response selectivity per unit.

    Args:
        peth: array (n_units, n_trials, n_timepoints) of firing rates (spk/s)
        bin_edges: array (n_timepoints + 1,) of bin edges in seconds relative to event
        unit_ids: sequence of unit IDs matching peth's first axis
        base_window: (start, end) seconds for baseline
        resp_window: (start, end) seconds for response
        test: 'wilcoxon' (default) or 'ttest'
        correction: multiple-comparisons method passed to
            statsmodels.stats.multitest.multipletests, e.g. 'fdr_bh'
            (Benjamini-Hochberg, default) or 'bonferroni'.
        alpha: significance threshold applied to corrected p-values

    Returns:
        results_df: DataFrame with per-unit stats
        masks: dict with boolean masks (excited, suppressed, selective)
    """
    n_units, n_trials, n_time = peth.shape
    if len(unit_ids) != n_units:
        raise ValueError(f"len(unit_ids)={len(unit_ids)} != peth n_units={n_units}")
    t_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    base_mask = (t_centers >= base_window[0]) & (t_centers < base_window[1])
    resp_mask = (t_centers >= resp_window[0]) & (t_centers < resp_window[1])

    if not base_mask.any() or not resp_mask.any():
        raise ValueError("Baseline/response windows do not overlap available bins.")

    base_rates = peth[:, :, base_mask].mean(axis=2)
    resp_rates = peth[:, :, resp_mask].mean(axis=2)

    pvals = np.ones(n_units, dtype=float)
    deltas = np.zeros(n_units, dtype=float)
    d_cohen = np.zeros(n_units, dtype=float)
    mean_base = base_rates.mean(axis=1)
    mean_resp = resp_rates.mean(axis=1)
    si = (mean_resp - mean_base) / (mean_resp + mean_base + 1e-9)

    for u in range(n_units):
        x = resp_rates[u]
        y = base_rates[u]
        diff = x - y

        deltas[u] = diff.mean()
        sd = diff.std(ddof=1)
        d_cohen[u] = deltas[u] / sd if sd > 0 else 0.0

        if np.allclose(diff, 0):
            pvals[u] = 1.0
            continue

        if test == "ttest":
            _, pvals[u] = stats.ttest_rel(
                x, y, alternative="two-sided", nan_policy="omit"
            )
        elif test == "wilcoxon":
            try:
                _, pvals[u] = stats.wilcoxon(
                    x, y, zero_method="wilcox", alternative="two-sided"
                )
            except ValueError:
                pvals[u] = 1.0
        else:
            raise ValueError("test must be 'wilcoxon' or 'ttest'")

    _, qvals, _, _ = multipletests(pvals, alpha=alpha, method=correction)

    excited = (qvals < alpha) & (deltas > 0)
    suppressed = (qvals < alpha) & (deltas < 0)
    selective_any = excited | suppressed

    results_df = pd.DataFrame(
        {
            "unit": list(unit_ids),
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


def classify_peak_count(
    peth: np.ndarray,
    bin_centers: np.ndarray,
    unit_ids: Sequence,
    search_window: tuple[float, float] = (0.0, 0.15),
    baseline_window: tuple[float, float] = (-0.1, 0.0),
    min_prominence_frac: float = 0.25,
    min_prominence_abs: float = 1.0,
    min_distance_ms: float = 20.0,
    binwidth_ms: float = 10.0,
    mode: str = "peaks",
) -> pd.DataFrame:
    """Classify units by the number of peaks (or dips) in their trial-averaged PSTH.

    Uses scipy.signal.find_peaks with a prominence threshold that is the
    *larger* of an adaptive per-unit fraction and an absolute floor.

    In "peaks" mode, only excursions above baseline are counted.
    In "dips" mode, the signal is inverted so that suppression troughs
    are detected instead.

    Args:
        peth: array (n_units, n_trials, n_timebins) firing rates (sp/s).
        bin_centers: array (n_timebins,) in seconds relative to event.
        unit_ids: sequence of unit IDs matching peth's first axis.
        search_window: (start, end) seconds to search for peaks/dips.
        baseline_window: (start, end) seconds for baseline subtraction.
        min_prominence_frac: minimum prominence as a fraction of
            the unit's max excursion (0-1).
        min_prominence_abs: absolute minimum prominence in sp/s. Acts as
            a floor so that noise bumps in low-rate units are ignored.
        min_distance_ms: minimum distance between peaks/dips in ms.
        binwidth_ms: bin width in ms (used to convert distance to bins).
        mode: "peaks" to detect excitatory peaks, "dips" to detect
            suppression troughs.

    Returns:
        DataFrame with columns: unit, n_peaks, peak_times, peak_heights.
        In "dips" mode, peak_heights are the actual (sub-baseline) firing
        rates at the trough locations.
    """
    if mode not in ("peaks", "dips"):
        raise ValueError("mode must be 'peaks' or 'dips'")

    n_units = peth.shape[0]
    if len(unit_ids) != n_units:
        raise ValueError(f"len(unit_ids)={len(unit_ids)} != peth n_units={n_units}")

    base_mask = (bin_centers >= baseline_window[0]) & (bin_centers < baseline_window[1])
    search_mask = (bin_centers >= search_window[0]) & (bin_centers < search_window[1])
    search_centers = bin_centers[search_mask]
    dist_bins = max(1, int(round(min_distance_ms / binwidth_ms)))

    records = []
    for u in range(n_units):
        mean_psth = peth[u].mean(axis=0)
        baseline_mean = mean_psth[base_mask].mean() if base_mask.any() else 0.0
        excess = mean_psth[search_mask] - baseline_mean

        # In dips mode, invert so troughs become peaks for find_peaks
        signal = -excess if mode == "dips" else excess

        max_signal = signal.max()
        if max_signal <= 0:
            records.append(
                dict(unit=unit_ids[u], n_peaks=0, peak_times=[], peak_heights=[])
            )
            continue

        prom_thresh = max(min_prominence_frac * max_signal, min_prominence_abs)
        detected, props = find_peaks(signal, prominence=prom_thresh, distance=dist_bins)

        # Only keep detections on the correct side of baseline
        if mode == "dips":
            valid = excess[detected] < 0
        else:
            valid = excess[detected] > 0
        detected = detected[valid]

        peak_times = search_centers[detected].tolist()
        peak_heights = (mean_psth[search_mask][detected]).tolist()

        # Fallback: if no peaks survived but there IS a clear excursion,
        # assign 1 peak at argmax (catches broad plateaus).
        if len(detected) == 0 and max_signal > 0:
            best = int(np.argmax(signal))
            peak_times = [float(search_centers[best])]
            peak_heights = [float(mean_psth[search_mask][best])]
            records.append(
                dict(
                    unit=unit_ids[u],
                    n_peaks=1,
                    peak_times=peak_times,
                    peak_heights=peak_heights,
                )
            )
        else:
            records.append(
                dict(
                    unit=unit_ids[u],
                    n_peaks=len(detected),
                    peak_times=peak_times,
                    peak_heights=peak_heights,
                )
            )

    return pd.DataFrame(records)


def classify_peak_count_sweep(
    peth: np.ndarray,
    bin_centers: np.ndarray,
    unit_ids: Sequence,
    prominence_fracs: Optional[Sequence[float]] = None,
    **kwargs,
) -> pd.DataFrame:
    """Run classify_peak_count across a range of prominence fractions.

    Returns a tidy DataFrame with columns: prominence_frac, n_peaks, count.
    Useful for plotting sensitivity of peak classification to the
    prominence threshold.
    """
    if prominence_fracs is None:
        prominence_fracs = np.arange(0.05, 0.55, 0.05)

    rows = []
    for frac in prominence_fracs:
        df = classify_peak_count(
            peth, bin_centers, unit_ids, min_prominence_frac=frac, **kwargs
        )
        counts = df["n_peaks"].value_counts()
        for np_val, cnt in counts.items():
            rows.append(
                dict(
                    prominence_frac=round(float(frac), 3),
                    n_peaks=int(np_val),
                    count=int(cnt),
                )
            )
    return pd.DataFrame(rows)


def calculate_stim_offsets(trial_ts, trial_start_col="center_port_entries"):
    """
    Calculates the offset of each stationary and movement stimulus relative
    to the specified trial start time column for each trial.

    Returns a DataFrame with trial_idx, stim_time, offset, and movement_status.
    """
    records = []
    for idx, row in trial_ts.iterrows():
        trial_start_time = (row[trial_start_col])[0]

        # Handle potential None values in stim lists
        stationary_stims = (
            row["stationary_stims"][1:] if row["stationary_stims"] is not None else []
        )  # exclude the first stim
        movement_stims = (
            row["movement_stims"] if row["movement_stims"] is not None else []
        )

        for st in stationary_stims:
            records.append(
                {
                    "trial_idx": idx,
                    "stim_time": st,
                    "offset": st - trial_start_time,
                    "movement_status": 0,  # 0 for stationary
                }
            )

        for mt in movement_stims:
            records.append(
                {
                    "trial_idx": idx,
                    "stim_time": mt,
                    "offset": mt - trial_start_time,
                    "movement_status": 1,  # 1 for movement
                }
            )

    stims_offset_df = pd.DataFrame(records)
    return stims_offset_df


def find_unique_cross_trial_offset_pairs(stims_offset_df, wiggle_room=0.1):
    """
    Finds pairs of stationary (movement_status=0) and movement (movement_status=1)
    stimuli from *different* trials whose offsets are within wiggle_room of each other.
    Ensures that each stimulus (stationary or movement) is used in at most one pair.

    Args:
        stims_offset_df (pd.DataFrame): DataFrame from calculate_stim_offsets.
        wiggle_room (float): Maximum allowed absolute difference between offsets.

    Returns:
        pd.DataFrame: DataFrame containing uniquely matched pairs.
    """

    # Use original index for tracking usage
    stims_offset_df = stims_offset_df.reset_index().rename(
        columns={"index": "original_index"}
    )

    stationary_stims = stims_offset_df[stims_offset_df["movement_status"] == 0].copy()
    movement_stims = stims_offset_df[stims_offset_df["movement_status"] == 1].copy()

    matched_pairs = []
    used_stationary_indices = set()
    used_movement_indices = set()

    # Sort stationary stimuli by offset to potentially make matching more consistent (optional)
    # stationary_stims = stationary_stims.sort_values('offset')

    # Iterate through each stationary stimulus
    for _, stat_row in stationary_stims.iterrows():
        stat_idx = stat_row["original_index"]

        # Skip if this stationary stimulus has already been used
        if stat_idx in used_stationary_indices:
            continue

        # Find potential movement stimuli in DIFFERENT trials with close offsets
        potential_matches = movement_stims[
            (movement_stims["trial_idx"] != stat_row["trial_idx"])
            & (np.abs(movement_stims["offset"] - stat_row["offset"]) <= wiggle_room)
            & (
                ~movement_stims["original_index"].isin(used_movement_indices)
            )  # Only consider unused movement stims
        ].copy()

        # Optional: Sort potential matches by offset difference to pick the closest one first
        potential_matches["offset_diff"] = np.abs(
            potential_matches["offset"] - stat_row["offset"]
        )
        potential_matches = potential_matches.sort_values("offset_diff")

        # If there are any available matches, take the first one (closest offset diff if sorted)
        if not potential_matches.empty:
            move_row = potential_matches.iloc[0]
            move_idx = move_row["original_index"]

            # Record the pair
            matched_pairs.append(
                {
                    "stat_trial_idx": stat_row["trial_idx"],
                    "stat_stim_time": stat_row["stim_time"],
                    "stat_offset": stat_row["offset"],
                    "move_trial_idx": move_row["trial_idx"],
                    "move_stim_time": move_row["stim_time"],
                    "move_offset": move_row["offset"],
                    "offset_diff": move_row["offset_diff"],  # Use calculated diff
                }
            )

            # Mark both stimuli as used
            used_stationary_indices.add(stat_idx)
            used_movement_indices.add(move_idx)

    matched_df = pd.DataFrame(matched_pairs)
    print(
        f"Found {len(matched_df)} unique cross-trial pairs with offset difference <= {wiggle_room}s."
    )
    return matched_df


def compute_stim_response_for_trial_subset(
    spike_times_per_unit,
    trial_subset,
    pre_seconds,
    post_seconds,
    binwidth_ms,
    stim_window_start,
    stim_window_end,
    wiggle_room=0.010,
):
    """
    Given a trial subset (slow or fast RT), compute the mean population response
    for stationary and movement conditions using matched unique pairs.
    Returns a dict with keys 'Stationary' and 'Movement'.
    """
    # TODO: make this function general for comparing any two subsets of trials
    from spks.event_aligned import population_peth  # type: ignore

    data = trial_subset.copy()
    stims_offset_df = calculate_stim_offsets(
        data, trial_start_col="center_port_entries"
    )
    matched_pairs_df = find_unique_cross_trial_offset_pairs(
        stims_offset_df, wiggle_room=wiggle_room
    )
    n_matched_stims = len(matched_pairs_df)
    if matched_pairs_df.empty:
        print("No matched pairs found!")
        return None

    # Alignment times for stationary and movement stims
    stat_alignment_times = matched_pairs_df["stat_stim_time"].values
    move_alignment_times = matched_pairs_df["move_stim_time"].values

    pop_peth_stat, timebin_edges_stat, _ = population_peth(
        all_spike_times=spike_times_per_unit,
        alignment_times=stat_alignment_times,
        pre_seconds=pre_seconds,
        post_seconds=post_seconds,
        binwidth_ms=binwidth_ms,
        pad=0,
        kernel=None,
    )
    pop_peth_move, _, _ = population_peth(
        all_spike_times=spike_times_per_unit,
        alignment_times=move_alignment_times,
        pre_seconds=pre_seconds,
        post_seconds=post_seconds,
        binwidth_ms=binwidth_ms,
        pad=0,
        kernel=None,
    )

    # Define response window based on timebins (use stationary timebins as reference)
    stimulus_window_bool = (timebin_edges_stat[:-1] >= stim_window_start) & (
        timebin_edges_stat[:-1] <= stim_window_end
    )

    stationary_response = np.max(
        np.mean(pop_peth_stat[:, :, stimulus_window_bool], axis=1), axis=1
    )  # mean across trials and then max of the stimulus window
    running_response = np.max(
        np.mean(pop_peth_move[:, :, stimulus_window_bool], axis=1), axis=1
    )

    n_neurons = pop_peth_stat.shape[1]
    stationary_sem = np.std(
        np.max(pop_peth_stat[:, :, stimulus_window_bool], axis=2), axis=1
    ) / np.sqrt(n_neurons)
    running_sem = np.std(
        np.max(pop_peth_move[:, :, stimulus_window_bool], axis=2), axis=1
    ) / np.sqrt(n_neurons)
    return {
        "stationary": stationary_response,
        "stationary_sem": stationary_sem,
        "running": running_response,
        "running_sem": running_sem,
    }, n_matched_stims


# %% General functions
def get_nth_element(x, i):
    """
    Get the n-th event in an array.

    Parameters:
    -----------
    x : array-like
        Input array.
    i : int
        Index of the element to retrieve.

    Returns:
    --------
    float or np.nan
        The i-th element of x if it exists, otherwise np.nan.

    Example usage:
    --------------
    for i, ax in enumerate(axs):
        for outcome, c in zip(np.unique(stim_ts_per_trial.trial_outcome), ['b', 'k', 'r', 'y']):
            ts = np.hstack(stim_ts_per_trial[stim_ts_per_trial.trial_outcome == outcome].stim_ts.apply(lambda x: get_nth_element(x, i)))
    """
    if len(x) > i and not np.isnan(x[0]):
        return x[i]
    return np.nan


def moving_average(data, window_size):
    """
    Calculate the moving average of a 1D array.

    Parameters:
    -----------
    data : array-like
        Input data.
    window_size : int
        Size of the moving window.

    Returns:
    --------
    array-like
        Moving average of the input data.
    """
    return np.convolve(data, np.ones(window_size), "valid") / window_size


def compute_mean_sem(psth: np.ndarray):
    """
    Compute mean and standard error of the mean for a given PSTH.

    Parameters:
    -----------
    psth : np.ndarray
        Population spike time histogram.

    Returns:
    --------
    tuple
        Mean and standard error of the mean of the PSTH.
    """
    return np.mean(psth, axis=0), np.std(psth, axis=0) / np.sqrt(psth.shape[0])
