# TODO

## Next up

**Double-peak PSTH figure for Dario email — DONE (draft).**
- Output: `figures/double_peak_dario.pdf` (2 pages).
  - Page 1 (Prevalence): GRB006 unit 77 archived screenshot + GRB058 units 410, 651 (3/12), 515 (3/19), 15 ms PSTHs with peak markers.
  - Page 2 (Offset hypothesis): GRB058 units 410/651/515, 15 ms (blue) vs 30 ms (orange) overlaid on same axes, peak markers on both conditions.
- Script: `scripts/double_peak_pulse_split.py`
- GRB006 archived figures stored in `figures/GRB006/` (unit 77 PETHs, rasters, spike duration histogram, copied from Marsa's Dec 2025 email).
- Remaining: draft and send the Dario email (user task).
- Confounds to note in email: (1) 30 ms = ~2× photons at same LED intensity (luminance vs width confound); (2) small 30 ms sample (30–34 first-of-train events); (3) first-pulse-only alignment; (4) example units, not population stats.

## Analysis TODOs

- **GRB006 events pipeline**: currently fails because events are handled through the obx logic. Needs a path for sessions recorded on nidq.
- **Population summary**: compute % of units that are (a) stimulus-selective, (b) rate-selective, (c) double-peak. `selectivity.ipynb` already does the first; (b) and (c) still need tying together in a single summary.
- **Data export for Marsa (GLM fitting)**: scratch code lives in `notebooks/export_data_for_marsa.ipynb`. Needs to produce:
  - `neural_data` DataFrame (one row per unit): `unit_id`, `spike_times`.
  - `behavior_data` DataFrame (one row per trial): `trial_start`, `center_poke`, `stim_onsets`, `left_poke`, `right_poke`.

## Code cleanup

- `src/utils/utils_analysis.py:447` — generalize `compute_stim_response_for_trial_subset` so it can compare any two trial subsets, not just stationary vs movement.
- `scripts/testing_timing_stim_events.py:331` — collapse the redundant `df` and `df_q`; build one combined DataFrame upfront, then plot from it.

## Post-SfN (June–November 2026)

Anne's explicit scope says to hold off on these until after the 2026-06-10 abstract:
- TIM (task-independent movement) analysis — requires DLC pipeline.
- Shared-gain GLM (Aim 2).
- Stringer (2019) replication.
- DLC pipeline setup — blocked on verifying the visible-light filter doesn't tank tracking contrast.
