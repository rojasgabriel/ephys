import unittest

import numpy as np
import pandas as pd

from ephys.src.utils.analysis_rastermap import (
    build_continuous_spike_count_matrix,
    build_trial_window_spike_count_matrix,
    choose_rastermap_cluster_count,
    fit_rastermap,
    heatmap_for_result,
    RastermapResult,
    trial_windows_from_metadata,
)


class ContinuousSpikeMatrixTests(unittest.TestCase):
    def test_build_continuous_spike_count_matrix_preserves_unit_order_and_counts(self):
        spike_times_by_unit = {
            20: np.array([0.05, 0.11, 0.19, 0.21, np.nan]),
            10: np.array([0.0, 0.1, 0.299]),
        }

        matrix = build_continuous_spike_count_matrix(
            spike_times_by_unit,
            bin_ms=100,
            t_stop_s=0.3,
        )

        np.testing.assert_array_equal(matrix.unit_ids, np.array([20, 10]))
        np.testing.assert_allclose(matrix.bin_edges_s, np.array([0.0, 0.1, 0.2, 0.3]))
        np.testing.assert_array_equal(
            matrix.spike_counts,
            np.array(
                [
                    [1, 2, 1],
                    [1, 1, 1],
                ],
                dtype=np.float32,
            ),
        )

    def test_build_continuous_spike_count_matrix_rejects_empty_units(self):
        with self.assertRaises(ValueError):
            build_continuous_spike_count_matrix({})

    def test_trial_windows_from_metadata_maps_initiation_to_response_to_obx_clock(self):
        trial_df = pd.DataFrame(
            {
                "t_sync": [0.0, 10.0, 20.0],
                "trial_start_ts": [100.0, 110.0, 120.0],
                "t_initiate": [1.0, 11.0, np.nan],
                "t_response": [4.0, 14.0, 24.0],
            }
        )

        windows = trial_windows_from_metadata(trial_df)

        np.testing.assert_allclose(windows, np.array([[101.0, 104.0], [111.0, 114.0]]))

    def test_build_trial_window_spike_count_matrix_concatenates_trial_windows(self):
        spike_times_by_unit = {
            20: np.array([0.05, 0.11, 1.05, 1.25, 1.45, 2.0]),
            10: np.array([0.15, 1.15, 1.35]),
        }
        trial_windows_s = np.array([[0.0, 0.25], [1.0, 1.4]])

        matrix = build_trial_window_spike_count_matrix(
            spike_times_by_unit,
            trial_windows_s,
            bin_ms=100,
        )

        np.testing.assert_array_equal(matrix.unit_ids, np.array([20, 10]))
        np.testing.assert_allclose(
            matrix.bin_edges_s, np.array([0.0, 0.1, 0.2, 0.25, 0.35, 0.45, 0.55, 0.65])
        )
        np.testing.assert_array_equal(
            matrix.trial_idx_by_bin,
            np.array([0, 0, 0, 1, 1, 1, 1]),
        )
        np.testing.assert_array_equal(
            matrix.spike_counts,
            np.array(
                [
                    [1, 1, 0, 1, 0, 1, 0],
                    [0, 1, 0, 0, 1, 0, 1],
                ],
                dtype=np.float32,
            ),
        )


class RastermapFitTests(unittest.TestCase):
    def test_choose_rastermap_cluster_count_clamps_below_unit_count(self):
        self.assertEqual(choose_rastermap_cluster_count(12), 11)
        self.assertEqual(choose_rastermap_cluster_count(100), 100)
        self.assertEqual(choose_rastermap_cluster_count(189), 100)

    def test_choose_rastermap_cluster_count_rejects_too_few_units(self):
        with self.assertRaises(ValueError):
            choose_rastermap_cluster_count(9)

    def test_fit_rastermap_uses_expected_parameters_without_real_fit(self):
        class FakeRastermap:
            last_instance = None

            def __init__(self, **kwargs):
                self.kwargs = kwargs
                FakeRastermap.last_instance = self

            def fit(self, data):
                self.data = data
                self.isort = np.arange(data.shape[0] - 1, -1, -1)
                self.embedding = np.arange(data.shape[0])[:, np.newaxis]
                self.X_embedding = data[self.isort]
                return self

        spike_counts = np.arange(60, dtype=float).reshape(12, 5)

        model, n_clusters = fit_rastermap(
            spike_counts,
            rastermap_cls=FakeRastermap,
        )

        self.assertIs(model, FakeRastermap.last_instance)
        self.assertEqual(n_clusters, 11)
        self.assertEqual(
            FakeRastermap.last_instance.kwargs,
            {
                "n_PCs": 200,
                "n_clusters": 11,
                "locality": 0.75,
                "time_lag_window": 5,
            },
        )
        self.assertEqual(FakeRastermap.last_instance.data.dtype, np.float32)
        np.testing.assert_array_equal(model.isort, np.arange(11, -1, -1))

    def test_heatmap_falls_back_to_zscored_sorted_counts(self):
        result = RastermapResult(
            subject="SUBJ",
            session="SESSION",
            unit_ids=np.array([1, 2]),
            depth=np.array([100.0, 200.0]),
            bin_edges_s=np.array([0.0, 0.1, 0.2, 0.3]),
            spike_counts=np.array([[1.0, 2.0, 3.0], [5.0, 5.0, 5.0]]),
            trial_idx_by_bin=None,
            absolute_bin_start_s=None,
            absolute_bin_stop_s=None,
            isort=np.array([1, 0]),
            embedding=np.array([[0.0], [1.0]]),
            x_embedding=None,
            n_clusters=2,
        )

        heatmap = heatmap_for_result(result)

        np.testing.assert_allclose(heatmap[0], np.zeros(3))
        np.testing.assert_allclose(
            heatmap[1],
            np.array([-1.22474487, 0.0, 1.22474487]),
        )


if __name__ == "__main__":
    unittest.main()
