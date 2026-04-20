"""Canonical parameters for double-peak V1 unit classification.

Last reviewed: 2026-04-19.

Pipeline:
  1. PETH (sp/s) with kernel disabled — peaks 30–45 ms apart need fine
     temporal resolution; the default spks kernel (t_decay=0.025s) merges them.
  2. Selectivity (excited): FDR-corrected Wilcoxon comparing per-trial
     baseline vs response means.
  3. Peak detection: prominence-based; both peaks must fall within
     PEAK_KWARGS["search_window"].
  4. Height filter: both peaks must clear MIN_PEAK_HEIGHT_ABS sp/s above
     baseline.

A unit is "double-peak" iff selectivity passes (excited=True) AND step 3
returns n_peaks==2 AND step 4 passes.

DO NOT define these dicts inline in scripts. Import from here so all
analyses move together.
"""

PETH_KWARGS = dict(
    pre_seconds=0.1,
    post_seconds=0.15,
    binwidth_ms=10,
    t_rise=None,  # disable kernel
    t_decay=None,  # default merges 30-45 ms peaks
)

BASELINE_WINDOW = (-0.04, 0.0)
WINDOW = (0.03, 0.12)  # used as both selectivity resp_window AND peak search_window

SELECTIVITY_KWARGS = dict(
    base_window=BASELINE_WINDOW,
    resp_window=WINDOW,
    test="wilcoxon",
    correction="fdr_bh",
    alpha=0.05,
)

PEAK_KWARGS = dict(
    search_window=WINDOW,
    baseline_window=BASELINE_WINDOW,
    min_prominence_frac=0.25,
    min_distance_ms=20.0,
    binwidth_ms=10.0,
)

MIN_PEAK_HEIGHT_ABS = 5.0  # sp/s — both peaks must clear this above baseline
