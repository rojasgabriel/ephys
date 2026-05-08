import unittest

import numpy as np

from scripts.diagnostics.seed_obx_audio_events import recover_io1_epochs


class RecoverIo1EpochsTests(unittest.TestCase):
    def test_recover_io1_epochs_uses_exact_sample_rate_for_bin_edges(self):
        sample_rate_hz = 10.5
        bin_s = 1.0
        n_bins = 60
        sample_edges = np.rint(np.arange(n_bins + 1) * sample_rate_hz * bin_s).astype(
            np.int64
        )
        dat = np.zeros((sample_edges[-1], 1), dtype=np.int16)

        for bin_idx in range(20, 23):
            start = sample_edges[bin_idx]
            stop = sample_edges[bin_idx + 1]
            dat[start] = -100
            dat[stop - 1] = 100

        starts, stops, _, _ = recover_io1_epochs(
            dat,
            {"sRateHz": sample_rate_hz},
            channel_index=0,
            bin_ms=1000.0,
            threshold_z=1.0,
            merge_gap_ms=0.0,
            min_duration_ms=1000.0,
        )

        np.testing.assert_allclose(starts, [sample_edges[20] / sample_rate_hz])
        np.testing.assert_allclose(stops, [sample_edges[23] / sample_rate_hz])


if __name__ == "__main__":
    unittest.main()
