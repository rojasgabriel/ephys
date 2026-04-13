# TODO

## Next up

**Double-peak PSTH figure for Dario email — DONE (draft).**
- Output: `figures/double_peak_dario.pdf` — single landscape page, 2 rows × 3 columns.
  - Top row: best single-peak excited unit per animal (GRB058, GRB059, GRB060), 15 ms only, gray.
  - Bottom row: double-peak units from GRB058 — units 410 and 651 (3/12), unit 515 (3/19) — 15 ms (blue) + 30 ms (orange) overlaid.
  - GRB059 and GRB060 have no double-peak units (small samples, barely trained); absent from bottom row.
  - GRB006 (~11/150 double-peak units): referenced in email text; nidq pipeline not yet ported.
- Script: `scripts/double_peak_pulse_split.py`
- GRB006 archived figures stored in `figures/GRB006/` (unit 77 PETHs, rasters, spike duration histogram, copied from Marsa's Dec 2025 email).
- Remaining: attach figure and send the Dario email (Gmail draft ready, CC Anne + Marsa). Manually delete the superseded older draft.
- Confounds to note in email: (1) 30 ms = ~2× photons at same LED intensity; (2) small 30 ms sample (30–34 first-of-train events); (3) first-pulse-only alignment.

## Analysis TODOs

- **Locomotion effect replication (GRB058–060)**: replicate the GRB006 locomotion × V1 gain effect in the newer cohort. GRB058 is the primary SfN target; GRB059 and GRB060 are additional candidates once they have enough trials. Requires movement data (wheel/DLC) aligned to electrophysiology sessions.

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
