import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from ephys.src.utils import utils_analysis


class ComputePopulationPethTests(unittest.TestCase):
    def test_population_peth_converts_counts_to_spikes_per_second(self):
        peth_counts = np.array([[[0.0, 1.0, 2.0]]])
        bin_edges = np.array([-0.01, 0.0, 0.01, 0.02])

        with patch.object(
            utils_analysis,
            "population_peth",
            return_value=(peth_counts, bin_edges, np.array([0])),
        ):
            peth, returned_edges, bin_centers = utils_analysis.compute_population_peth(
                [np.array([0.0])],
                np.array([0.0]),
                pre_seconds=0.01,
                post_seconds=0.02,
                binwidth_ms=10,
            )

        np.testing.assert_allclose(peth, peth_counts / 0.01)
        np.testing.assert_allclose(returned_edges, bin_edges)
        np.testing.assert_allclose(bin_centers, np.array([-0.005, 0.005, 0.015]))

    def test_population_peth_rejects_implausible_rates(self):
        peth_counts = np.array([[[11.0]]])
        bin_edges = np.array([0.0, 0.01])

        with patch.object(
            utils_analysis,
            "population_peth",
            return_value=(peth_counts, bin_edges, np.array([0])),
        ):
            with self.assertRaises(AssertionError):
                utils_analysis.compute_population_peth(
                    [np.array([0.0])],
                    np.array([0.0]),
                    pre_seconds=0.0,
                    post_seconds=0.01,
                    binwidth_ms=10,
                )


class ConditionedStimAnchorTests(unittest.TestCase):
    def test_extract_conditioned_stim_anchors_uses_paired_trials_only_for_pairs(self):
        trial_ts = pd.DataFrame(
            {
                "trial_idx": [0, 1, 2],
                "stationary_stims": [[0.1, 0.2], [], [2.1, 2.2, 2.3]],
                "movement_stims": [[0.5], [1.5], []],
            }
        )

        anchors = utils_analysis.extract_conditioned_stim_anchors(trial_ts)

        np.testing.assert_allclose(anchors["first_stationary_all"], [0.1, 2.1])
        np.testing.assert_allclose(anchors["paired_last_stationary"], [0.2])
        np.testing.assert_allclose(anchors["paired_first_movement"], [0.5])
        np.testing.assert_array_equal(anchors["paired_trial_idx"], [0])


if __name__ == "__main__":
    unittest.main()
