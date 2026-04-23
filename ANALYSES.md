# Ephys Analysis Guide

Current maintained analysis surface for this repo.

Run from the repo root:

```bash
uv run python scripts/<group>/<script_name>.py
```

All maintained figure scripts write PDF outputs under `figures/`.

## Canonical Entrypoints

### Locomotion

`scripts/maintained/locomotion_stat_vs_move.py`

Writes:

- `figures/locomotion/scatter.pdf`
- `figures/locomotion/overlay.pdf`
- `figures/locomotion/timing_ctrl.pdf`

Current maintained policy:

- This is the only maintained locomotion entrypoint.
- Baseline-only soft gate: a unit is analyzable if it rises above baseline in
  at least one condition.
- Main comparison:
  - GRB006: `3rd stationary` vs `first movement`
  - GRB058: `last stationary` vs `first movement`
- Rate split: low `< 12 Hz`, high `> 12 Hz`
- Offset-matched low/high panels keep only same-stratum pairs. Cross-stratum
  and threshold-boundary pairs are excluded.

Current verified snapshot from the 2026-04-20 live run:

- GRB006 `20240821_121447`: `79%` above diagonal, `Δmedian=+2.36 sp/s`,
  analyzable `n=168`
- GRB058 `20260312_134952`: `72%` above diagonal, `Δmedian=+0.78 sp/s`,
  analyzable `n=98`

### Double-Peak

Canonical parameters live in `src/config/double_peak.py`.

- PETH: `10 ms` bins, kernel disabled
- Baseline: `(-0.04, 0.0)` s
- Response / peak search: `(0.03, 0.12)` s
- Selectivity: Wilcoxon + FDR
- Height floor: both peaks must be at least `5 sp/s` above baseline

`compute_population_peth` already returns `sp/s`. Do not rescale it.

Maintained scripts:

- `scripts/maintained/double_peak_grb006_examples.py`
  - output: `figures/double_peak/grb006_examples.pdf`
  - scope: GRB006-only example figure
- `scripts/maintained/double_peak_story.py`
  - output: `figures/double_peak/dario_story.pdf`
  - scope: collaborator-facing summary figure
- `scripts/maintained/double_peak_waveform_grid.py`
  - output: `figures/double_peak/waveform_grid.pdf`
  - scope: firing rate vs spike duration only
- `scripts/maintained/double_peak_pulse_width_control.py`
  - output: `figures/double_peak/pulse_split.pdf`
  - scope: `15 ms` vs `30 ms` pulse-width control

Current maintained counts:

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

- Spike times: local KS4 export
  - `~/data/GRB006/20240821_121447/pre_processed/20240821_121447_ks4_spike_times.pkl`
  - fallback: `~/Downloads/Organized/Code/20240821_121447_ks4_spike_times.pkl`
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

## Supporting Tools

Still usable, but not part of the main maintained figure surface:

- `scripts/supporting/manual_conditioned_psth_browser.py`
  - interactive locomotion PSTH browser
- `scripts/supporting/stimulus_selectivity.py`
  - local first-stim selectivity utility
- `scripts/supporting/adaptation.py`
  - archived local GRB006 adaptation analysis
- `scripts/supporting/locomotion_snr_reference.py`
  - reference-only diagnostic for the removed SNR gate

## Retired Surface

Not part of the maintained interface:

- duplicate older locomotion scripts
- broken pre-canonical timing scripts
- depth-based locomotion figures
- depth-based double-peak figures
- removed duplicate stimulus-scatter entrypoints

## Script Layout

`scripts/` is now grouped by role instead of keeping every entrypoint flat:

- `scripts/maintained/`
  - canonical figure generators
- `scripts/supporting/`
  - still-usable helpers and reference analyses
- `scripts/diagnostics/`
  - one-off sync/debug investigations
- `scripts/tools/`
  - general-purpose interactive utilities
