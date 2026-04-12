# TODO

## Next up

**Double-peak PSTH split for Dario email (side thread, not SfN).**
- Status: infrastructure merged in PR #7 (strict 15/30 ms classification in `fetch_session_events()`); the analysis itself is not written yet.
- Plan: add a section to `notebooks/selectivity.ipynb` that (1) recomputes the PSTH twice, aligned to `align_ev["first_stim_ev_15ms"]` and `align_ev["first_stim_ev_30ms"]`; (2) runs `classify_peak_count` on the 15 ms PSTH to auto-pick 2-peak units; (3) plots those units' 15 ms vs 30 ms PSTHs side-by-side. Freeze into `scripts/double_peak_pulse_split.py` once the figure is right.
- Open question: which session to use — `GRB058 / 20260224_152424` (what `selectivity.ipynb` currently runs on) or `GRB058 / 20260312_134952` (what the pulse-width prototype was validated on). Check which sessions actually contain 30 ms pulses before deciding.
- Scientific confounds to acknowledge in the figure caption / email: (1) 30 ms delivers ~2× photons at same LED intensity — luminance vs pulse-width confound; (2) sample imbalance (prototype counts were 1757 vs 88 on GRB058); (3) interleaved vs blocked presentation; (4) first-pulse vs all-pulse alignment choice; (5) example units only, not population stats (per Anne's simplified brief).

## Analysis TODOs

- **GRB006 events pipeline**: currently fails because events are handled through the obx logic. Needs a path for sessions recorded on nidq. (Ported from `notebooks/to-do.ipynb`.)
- **Population summary**: compute % of units that are (a) stimulus-selective, (b) rate-selective, (c) double-peak. `selectivity.ipynb` already does the first; (b) and (c) still need tying together in a single summary.
- **Data export for Marsa (GLM fitting)**: scratch code lives in `notebooks/to-do.ipynb`. Needs to produce:
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
