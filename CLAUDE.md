# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync                   # install / sync dependencies
uvx ruff check .          # lint
uvx ruff format .         # format
uvx ruff check --fix .    # lint + auto-fix
```

There is no test suite; validation is done by running notebooks and scripts interactively. CI runs ruff check + format only.

## Architecture

Analysis of how body movement modulates V1 neural activity in freely-moving mice (Neuropixels + behavioral task data from the Churchland Lab).

### Source library (`src/utils/`)

Three modules form the core library; everything else (notebooks, scripts) consumes them.

- **`utils_IO.py`** — Database queries via `labdata`. Key functions:
  - `fetch_good_units(subject, session, unit_criteria_id)` → spike times + waveforms
  - `fetch_session_events(subject, session)` → parsed stimulus pulse events
  - `fetch_trial_metadata(subject, session)` → behavioral trial data (via `chipmunk`, optional)

- **`utils_analysis.py`** — PETH computation, selectivity testing, peak classification.
  - `compute_population_peth()` wraps `spks.population_peth`; output shape is `(n_units, n_trials, n_timebins)` in sp/s
  - Standard kernel: `t_rise=0.001s`, `t_decay=0.025s`
  - Standard window: `pre=0.1s`, `post=0.15s`, `binwidth=10ms`
  - Selectivity test: Wilcoxon signed-rank on baseline `(-0.04, 0.0)s` vs response `(0.06, 0.10)s`, Bonferroni-corrected

- **`viz.py`** — `PSTHViewer` and `PSTHWidget` for interactive Jupyter visualization. The kernel is cached at init; all `population_peth` calls within a viewer must use `self._kernel`.

### Application layer

- **`notebooks/`** — Primary workspace for exploratory analysis. `selectivity.ipynb` is the main working notebook.
- **`scripts/`** — Frozen, self-contained snapshots of successful analyses. They use hardcoded session IDs and paths (`/Users/gabriel/data`, `/Users/gabriel/lib/ephys/figures/`); they are not parameterized utilities.

### Other directories

- **`labdata_plugin/`** — Custom schema stubs for the `labdata` database interface.
- **`hardware/`** — CAD files for the Neuropixels probe holder (not code).
- **`metadata/`** — NIDAQ connectome file for DAQ channel configuration.

## Key domain conventions

**Stimulus pulse parsing**: Raw OBX digital edges are noisy (multiple toggles per pulse). `fetch_session_events` merges edges into bursts (gap threshold: 20ms) and classifies each burst as `"15ms"` or `"30ms"` (±2ms tolerance) or `"unknown"` (excluded). Two event streams result: `stim_ev_15ms` (all classified pulses) and `first_stim_ev_15ms` (first pulse per 1s train).

**Unit quality**: Filtering is database-driven via `UnitCount.Unit` with `unit_criteria_id` and `passes=1`. Different criteria IDs represent different quality thresholds.

**Trial sync**: Chipmunk behavioral trial counts are aligned to OBX trial-start pulses. Count mismatches are warned but silently truncated to the shorter sequence.

## Import paths

The package is installed in editable mode as `ephys`. Imports use the `src` layout:

```python
from ephys.src.utils.utils_IO import fetch_good_units
from ephys.src.utils.utils_analysis import compute_population_peth
from ephys.src.utils.viz import PSTHViewer
```

## GitHub agents

Two agent definitions live in `.github/agents/`:
- `code.agent.md` — for writing/editing Python (analysis, viz, I/O)
- `research.agent.md` — for research interpretation, experimental design, literature context
