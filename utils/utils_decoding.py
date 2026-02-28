import pandas as pd


def get_balanced_trials(trial_ts, require_both_stim_types=True):
    """This balances the number of rewarded vs. unrewarded trials and makes sure that there are both stationary and movement stimuli in each trial

    Args:
        trial_ts (pandas dataframe): contains the trial data as well as the nidaq events
        require_both_stim_types (bool, optional): require both stationary and movement stimuli in every trial. Defaults to True.

    Returns:
        pandas dataframe: balanced dataframe
        int: minimum number of rewarded and unrewarded trials
    """
    # Get valid trials (exclude early withdrawals)
    valid_trials = trial_ts[trial_ts.trial_outcome.isin([0, 1])]

    # Optionally require both stim types
    if require_both_stim_types:
        valid_trials = valid_trials[
            (valid_trials.movement_stims.apply(len) > 0)
            & (valid_trials.stationary_stims.apply(len) > 0)
        ]

    # Find minimum number of trials between conditions
    min_trials = min(
        len(valid_trials[valid_trials.trial_outcome == 1]),
        len(valid_trials[valid_trials.trial_outcome == 0]),
    )

    # Sample equal numbers from each condition
    balanced_trials = pd.concat(
        [
            valid_trials[valid_trials.trial_outcome == 1].sample(
                n=min_trials, random_state=42
            ),
            valid_trials[valid_trials.trial_outcome == 0].sample(
                n=min_trials, random_state=42
            ),
        ]
    )

    return balanced_trials, min_trials
