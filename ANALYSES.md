# Ephys Analysis Guide

Current analysis surface for this repo.

Run from the repo root:

```bash
uv run python scripts/<group>/<script_name>.py
```

Analysis figure scripts write PDF outputs under `figures/`.

## Regenerate Main Figures

```bash
uv run python scripts/analyses/locomotion_condition_specific_vs_same_latency_responses.py
uv run python scripts/analyses/double_peak_responses_across_sessions.py
```

## Canonical Entrypoints

### Locomotion

Primary locomotion analysis:

`scripts/analyses/locomotion_condition_specific_vs_same_latency_responses.py`

Writes:

- `figures/locomotion/readout_comparison_behavior_matching.pdf`
- `figures/locomotion/readout_comparison_latency_jitter.pdf`

Current policy:

- This is the canonical locomotion entrypoint.
- The main figure for now is `readout_comparison_behavior_matching.pdf`.
- It is a 2x2 comparison surface:
  - top row: baseline-gated shared-baseline views
  - bottom row: all-units raw-response views
  - left column: condition-peak readout
  - right column: shared-peak control readout
- The behavior-matching scatter uses shared log-scale axes across all four panels.
- This is the current figure to regenerate when the user asks for the main locomotion figure.

Secondary single-overlay figure:

`scripts/analyses/locomotion_condition_specific_peak_responses.py`

- writes `figures/locomotion/condition_peak_paired_last_stat_first_move_shared_stat_baseline_no_waveform_split.pdf`
- writes `figures/locomotion/condition_peak_paired_last_stat_first_move_shared_stat_baseline.pdf` when run with `--split-by-waveform`
- this is now a follow-up condition-peak-only figure rather than the primary output

Stricter control surface:

`scripts/analyses/locomotion_same_latency_response_control.py`

- This is no longer the main locomotion analysis.
- It remains the legacy shared-peak control surface with baseline gating and
  rate-split / timing-control figures.

### Double-Peak

Canonical parameters live in `src/config/double_peak.py`.

- PETH: `10 ms` bins, kernel disabled
- Baseline: `(-0.04, 0.0)` s
- Response / peak search: `(0.03, 0.12)` s
- Selectivity: Wilcoxon + FDR
- Height floor: both peaks must be at least `5 sp/s` above baseline

`compute_population_peth` converts the default `spks.population_peth` output to
`sp/s`. Do not rescale it again downstream.

Analysis scripts:

- `scripts/analyses/grb006_double_peak_example_units.py`
  - output: `figures/double_peak/grb006_examples.pdf`
  - scope: GRB006-only example figure
- `scripts/analyses/double_peak_responses_across_sessions.py`
  - output: `figures/double_peak/dario_story.pdf`
  - scope: collaborator-facing summary figure
- `scripts/analyses/double_peak_units_waveform_profile.py`
  - output: `figures/double_peak/waveform_grid.pdf`
  - scope: firing rate vs spike duration only
- `scripts/analyses/double_peak_responses_by_pulse_width.py`
  - output: `figures/double_peak/pulse_split.pdf`
  - scope: `15 ms` vs `30 ms` pulse-width control

Current counts:

- GRB006 `20240821_121447`: `5 / 189` double-peak units
  - `[217, 570, 579, 694, 720]`
- GRB058 `20260312_134952`: `2 / 142`
  - `[410, 651]`
- GRB058 `20260319_131303`: `1`
  - `[515]`

Interpretation boundary:

- The pulse-width figure rules out a simple pulse-offset explanation for the
  second peak.
- It does not, by itself, prove a specific mechanism.

## Data Path Caveats

### GRB006

GRB006 `20240821_121447` is hybrid-loaded.

- Spike times: DB-backed good units via `fetch_good_units`
- Trial anchors: local `trial_ts.pkl`
  - `~/data/GRB006/20240821_121447/pre_processed/trial_ts.pkl`
  - fallback: `~/Downloads/Organized/Code/trial_ts.pkl`
- Behavior summaries: DB-backed

### GRB058

GRB058 is DB-backed through `fetch_good_units`, `fetch_session_events`, and
`fetch_trial_metadata`.

`fetch_trial_metadata` mismatch policy:

- `1` trial mismatch: warn and truncate
- larger mismatch: raise

Current expected warnings:

- `20260312_134952`: `OBX=236`, `Chipmunk=235`
- `20260319_131303`: `OBX=219`, `Chipmunk=218`

## Other Scripts

Still usable, but not part of the main figure surface:

- `scripts/tools/manual_conditioned_psth_browser.py`
  - interactive locomotion PSTH browser
- `scripts/analyses/grb006_first_stimulus_selectivity.py`
  - local first-stim selectivity utility
- `scripts/analyses/grb006_visual_response_adaptation.py`
  - archived local GRB006 adaptation analysis
- `scripts/diagnostics/locomotion_response_snr_distribution.py`
  - reference-only diagnostic for the removed SNR gate

## Retired Surface

Not part of the current analysis interface:

- duplicate older locomotion scripts
- broken pre-canonical timing scripts
- depth-based locomotion figures
- depth-based double-peak figures
- removed duplicate stimulus-scatter entrypoints

## Script Layout

`scripts/` is grouped by scientific use:

- `scripts/analyses/`
  - scripts that answer a scientific question or generate analysis figures
- `scripts/diagnostics/`
  - one-off checks of data quality, sync, or pipeline assumptions
- `scripts/tools/`
  - interactive utilities and browsers
