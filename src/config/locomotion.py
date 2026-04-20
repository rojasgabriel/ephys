"""Canonical parameters for locomotion stat-vs-move analysis.

Last reviewed: 2026-04-19.

Aligned with double_peak.WINDOW for project-wide consistency. The same
response window is used for SNR computation, statistical tests, and
peak-centered measurements (no separate RESP_WINDOW vs EFFECT_WINDOW).

Baseline window matches double_peak.BASELINE_WINDOW.

DO NOT define these inline in scripts. Import from here.
"""

PETH_KWARGS = dict(
    pre_seconds=0.04,
    post_seconds=0.15,
    binwidth_ms=10,
)

BASELINE_WINDOW = (-0.04, 0.0)
RESP_WINDOW = (0.03, 0.12)  # SNR + effect tests + peak-window — single window
PEAK_HALF_WINDOW_S = 0.015  # for peak-centered measurements

QVAL_ALPHA = 0.05

# 12 Hz separates "low rate" from "high rate" in the rate-split scatters
RATE_SPLIT_HZ = 12.0
