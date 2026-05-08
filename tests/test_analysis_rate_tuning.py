import unittest

import numpy as np
import pandas as pd

from ephys.src.utils.analysis_rate_tuning import (
    aggregate_tuning_curves,
    build_task_stimulus_windows,
    compute_trial_responses,
    first_event_in_window,
    response_events_for_choice,
    summarize_units,
)


class RateTuningWindowTests(unittest.TestCase):
    def test_first_event_in_window_uses_half_open_bounds(self):
        events = np.array([0.0, 0.1, 0.2, 0.3])

        self.assertEqual(first_event_in_window(events, 0.1, 0.3), 0.1)
        self.assertEqual(
            first_event_in_window(events, 0.1, 0.3, include_start=False),
            0.2,
        )
        self.assertTrue(np.isnan(first_event_in_window(events, 0.31, 0.5)))

    def test_response_events_for_choice_selects_side_port(self):
        align_ev = {
            "left_port": np.array([1.0]),
            "right_port": np.array([2.0]),
        }

        np.testing.assert_allclose(response_events_for_choice(align_ev, -1), [1.0])
        np.testing.assert_allclose(response_events_for_choice(align_ev, 1), [2.0])
        self.assertEqual(response_events_for_choice(align_ev, 0).size, 0)

    def test_build_task_windows_keeps_only_valid_trials(self):
        align_ev = {
            "first_stim_ev_15ms": np.array([0.2, 1.2, 2.2, 3.2, 4.2]),
            "left_port": np.array([0.8, 4.8]),
            "right_port": np.array([1.8, 2.1, 3.8]),
        }
        trial_df = pd.DataFrame(
            {
                "trial_start_ts": [0.0, 1.0, 2.0, 3.0, 4.0],
                "stim_rate_vision": [4, 8, 12, 22, 20],
                "response": [-1, 1, 1, 1, -1],
                "with_choice": [1, 1, 1, 1, 0],
            },
            index=[10, 11, 12, 13, 14],
        )

        windows = build_task_stimulus_windows(align_ev, trial_df)

        self.assertEqual(windows["trial_idx"].tolist(), [10, 11])
        np.testing.assert_allclose(windows["window_start_s"], [0.2, 1.2])
        np.testing.assert_allclose(windows["window_end_s"], [0.8, 1.8])
        np.testing.assert_allclose(windows["stim_rate_vision"], [4, 8])


class RateTuningResponseTests(unittest.TestCase):
    def test_compute_trial_responses_counts_spikes_in_window(self):
        windows = pd.DataFrame(
            {
                "trial_idx": [0, 1],
                "stim_rate_vision": [4.0, 8.0],
                "response_side": [-1, 1],
                "with_choice": [1, 1],
                "window_start_s": [0.2, 1.0],
                "window_end_s": [0.7, 1.5],
                "window_duration_s": [0.5, 0.5],
            }
        )
        spikes = {
            101: np.array([0.2, 0.4, 0.7, 1.1, 1.4]),
            202: np.array([0.1, 1.6]),
        }

        responses = compute_trial_responses(windows, spikes)

        unit_101 = responses[responses["unit_id"] == 101].sort_values("trial_idx")
        self.assertEqual(unit_101["spike_count"].tolist(), [2, 2])
        np.testing.assert_allclose(unit_101["response_sp_s"], [4.0, 4.0])
        unit_202 = responses[responses["unit_id"] == 202].sort_values("trial_idx")
        self.assertEqual(unit_202["spike_count"].tolist(), [0, 0])
        np.testing.assert_allclose(unit_202["response_sp_s"], [0.0, 0.0])

    def test_aggregate_and_summarize_tuning_curves(self):
        responses = pd.DataFrame(
            {
                "unit_id": [1, 1, 1, 1, 2, 2],
                "stim_rate_vision": [4.0, 4.0, 8.0, 8.0, 4.0, 8.0],
                "response_sp_s": [1.0, 3.0, 5.0, 7.0, 10.0, 4.0],
                "window_duration_s": [0.5, 0.7, 0.5, 0.7, 0.5, 0.5],
            }
        )

        tuning = aggregate_tuning_curves(responses)
        unit_1 = tuning[tuning["unit_id"] == 1].sort_values("stim_rate_vision")
        np.testing.assert_allclose(unit_1["mean_sp_s"], [2.0, 6.0])
        np.testing.assert_allclose(unit_1["median_sp_s"], [2.0, 6.0])
        np.testing.assert_allclose(unit_1["n_trials"], [2, 2])

        summary = summarize_units(tuning)
        unit_1_summary = summary[summary["unit_id"] == 1].iloc[0]
        self.assertEqual(unit_1_summary["preferred_stim_rate"], 8.0)
        self.assertEqual(unit_1_summary["tuning_range_sp_s"], 4.0)
        self.assertEqual(unit_1_summary["frequency_selectivity_index"], 0.5)

        unit_2_summary = summary[summary["unit_id"] == 2].iloc[0]
        self.assertAlmostEqual(
            unit_2_summary["frequency_selectivity_index"],
            6.0 / 14.0,
        )


if __name__ == "__main__":
    unittest.main()
