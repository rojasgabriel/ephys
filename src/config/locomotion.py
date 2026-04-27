"""Shared parameters for locomotion analyses.

Last reviewed: 2026-04-23.

These windows are shared by the primary condition-peak locomotion analysis and
the older shared-peak control script.

Only `PETH_KWARGS`, `BASELINE_WINDOW`, and `RESP_WINDOW` should be assumed to
apply across locomotion analyses in general. The remaining constants are kept
here for the older control script and rate-split figures.

DO NOT define shared locomotion windows inline in scripts. Import them here.
"""

PETH_KWARGS = dict(
    pre_seconds=0.04,
    post_seconds=0.15,
    binwidth_ms=10,
)

BASELINE_WINDOW = (-0.04, 0.0)
RESP_WINDOW = (0.03, 0.12)

# Control-script-specific constants used by the same-latency locomotion analysis.
PEAK_HALF_WINDOW_S = 0.015

QVAL_ALPHA = 0.05

# 12 Hz separates "low rate" from "high rate" in the rate-split scatters
RATE_SPLIT_HZ = 12.0
