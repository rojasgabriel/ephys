# Ephys Analysis Guide

A practical reference for running the analyses in this project, understanding what each script computes, and interpreting the outputs.

---

## How to run any script

From the repo root:

```bash
uv run python scripts/<script_name>.py
```

All scripts write their output to `figures/` (gitignored). Re-running always overwrites.

---

## Data sources

### GRB006 — hybrid loading

GRB006's spike times in the database are stale (sorting was redone locally after the DB import). Two files must exist on disk:

| File | What it contains |
|---|---|
| `~/data/GRB006/20240821_121447/pre_processed/trial_ts.pkl` | Trial-by-trial timestamps (first stim, stationary stims, movement stims, trial rate, outcome) |
| `~/data/GRB006/20240821_121447/pre_processed/20240821_121447_ks4_spike_times.pkl` OR `~/Downloads/Organized/Code/20240821_121447_ks4_spike_times.pkl` | KS4 spike times for every unit (unit_id, spike_times in samples at 30 kHz) |

Scripts check both paths for the spike times pkl and use whichever exists.

Behavior metadata (trial outcomes, stim rates) comes from Chipmunk via the DB.

### GRB058 — fully DB-backed

Spike times, events, and behavior all come from the database via `fetch_good_units`, `fetch_session_events`, and `fetch_trial_metadata`.

**Canonical sessions:**
- `20260312_134952` — primary session used for all non-longstim analyses; richer signal
- `20260319_131303` — secondary; longstim-focused scripts only (has more 30 ms trials)

**Important:** ~25% of 4 Hz trials in the 20260312 session used a 30 ms pulse. When comparing GRB006 (which only has 15 ms pulses) to GRB058, alignment to `first_stim_ev_15ms` handles this automatically — it only includes trials whose first stimulus was 15 ms.

---

## Double-peak analyses

### What is a "double-peak" unit?

Some V1 neurons respond to a brief LED flash with two distinct firing rate peaks — one ~35 ms and another ~90 ms after stimulus onset. The leading hypothesis is that the first peak reflects direct retinal input and the second reflects a delayed recurrent or feedback signal. A simple test is whether the second peak shifts with pulse width (if it's offset-driven, it should shift with the end of the pulse). It does not — supporting the recurrent interpretation.

### Canonical classification pipeline

Parameters live in `src/config/double_peak.py`. Steps:

1. **PETH** — 10 ms bins, no smoothing kernel (kernel disabled because the default 25 ms decay merges peaks that are 30–45 ms apart), −100 ms to +150 ms around first stim
2. **Selectivity gate** — FDR-corrected Wilcoxon test on per-trial means in the response window (30–120 ms). Only "excited" units proceed (trial-mean response > trial-mean baseline, q < 0.05)
3. **Peak detection** — `scipy.find_peaks` with prominence ≥ 25% of the max signal in window, minimum distance 20 ms between peaks. Both peaks must fall in 30–120 ms.
4. **Height filter** — both peaks must rise ≥ 5 sp/s above the mean baseline (−40 to 0 ms). This prevents statistically-significant but biologically-tiny responses from passing.

A unit is **double-peak** if and only if all four steps pass.

**Key parameters:**

| Parameter | Value | Why |
|---|---|---|
| `binwidth_ms` | 10 ms | Fine enough to resolve 30–45 ms separation |
| `t_rise, t_decay` | None (disabled) | Default kernel at 25 ms decay merges the two peaks |
| `BASELINE_WINDOW` | −40 to 0 ms | Pre-stimulus baseline (short window to match PETH pre-time) |
| `WINDOW` | 30–120 ms | Where both peaks must appear |
| `correction` | `fdr_bh` | Benjamin-Hochberg FDR across all units in the session |
| `min_prominence_frac` | 0.25 | Fraction of max signal; prevents tiny local bumps |
| `min_distance_ms` | 20 ms | Minimum separation between peaks (prevents split on one peak) |
| `MIN_PEAK_HEIGHT_ABS` | 5 sp/s | Both peaks must clear this above baseline |

---

### `scripts/double_peak_grb006.py`

**What it does:** Classifies all GRB006 units, then plots the top 6 double-peak units (ranked by the smaller of their two peak heights above baseline) as a 2×3 grid of PETHs.

**Session:** GRB006 `20240821_121447`

**Data loading:** Local KS4 pkl + local trial_ts.pkl (DB is stale)

**Output:** `figures/double_peak/grb006_examples.pdf`

**Stdout:** Prints unit count, excited count, and a ranked list with peak latencies and heights.

---

### `scripts/double_peak_waveform_grid.py`

**What it does:** For each session (GRB006 and GRB058 20260312), plots firing rate vs spike duration colored by double-peak (orange) vs other (blue). All good units shown (not just excited ones). The FS/RS boundary line at 0.4 ms is a visual reference only — the double-peak classification doesn't use this threshold.

**Sessions:** GRB006 `20240821_121447` + GRB058 `20260312_134952`

**Output:** `figures/double_peak/waveform_grid.pdf`

---

### `scripts/double_peak_pulse_split.py`

**What it does:** Shows that the second peak does NOT shift with pulse width. Layout is 2 rows:
- Top row: a single-peak GRB058 reference unit (15 ms trials only, gray)
- Bottom row: each GRB058 double-peak unit with 15 ms (blue) and 30 ms (orange) overlaid; triangles mark detected peaks for each condition

This uses both GRB058 longstim sessions (20260312 + 20260319), which are the only sessions with enough 30 ms trials to make this comparison.

**Sessions:** GRB058 `20260312_134952` and `20260319_131303`

**Output:** `figures/double_peak/pulse_split.pdf`

**Note:** GRB006 is excluded here — it was only recorded with 15 ms pulses.

---

### `scripts/dario_double_peak_story.py`

**What it does:** The figure that was sent to Dario Ringach at UCLA. Tells the full discovery story:
- Column 1: Three hand-picked GRB006 exemplar units (unit IDs 579, 694, 217) with the double-peak shape
- Columns 2–3: Hand-picked GRB058 exemplars from the two longstim sessions (units 410, 651 from 20260312; unit 515 from 20260319)
- Each column also shows the algorithmic count for that session

The hand-picked unit IDs are hardcoded (`GRB006_SHOW_UNITS`, `SESSION_SHOW_UNITS`) and should not change without careful consideration — those specific units were what Dario saw. The algorithmic classification in the background uses the same canonical pipeline as all other scripts.

**Sessions:** GRB006 `20240821_121447` + GRB058 `20260312_134952` + `20260319_131303`

**Output:** `figures/double_peak/dario_story.pdf`

---

## Locomotion analyses

### What is the locomotion effect?

V1 neurons typically fire more strongly during locomotion than during stationary fixation, even when the visual stimulus is identical. We quantify this by aligning to two anchor events within the same trial: a stationary stim pulse and a movement stim pulse. Comparing the per-trial firing rates at the response peak gives a paired within-trial measure of locomotion modulation.

### Trial structure and anchor selection

Each trial in the task has:
- One or more **stationary stim pulses** — delivered while the animal is still
- One or more **movement stim pulses** — delivered while the animal is running

The locomotion comparison picks one stationary anchor and one movement anchor per trial, then computes a PETH for each. The **main analysis** uses:
- **GRB006:** 3rd stationary stim (index 2) — chosen because GRB006 is an expert mouse that has already licked and engaged before the 3rd stim, making its stationary period behaviorally more comparable to movement
- **GRB058:** last stationary stim — GRB058 is still in training, so the last stationary before movement begins is the most natural comparison point

The `timing_ctrl` figure repeats the scatter under 4 anchor configurations (last stat, 2nd stat, 3rd stat, offset-matched 0.5–0.7 s) to show the finding is not an artifact of which stationary stim you pick.

### SNR gate and soft baseline gate

Before computing the locomotion effect, each unit is screened:

1. **SNR gate:** trial-mean response / SEM ≥ 3.0, in **both** conditions (stat and move). Units that are pure noise in either condition are excluded.
2. **Soft baseline gate:** at least one condition must show a response **above baseline**. A unit suppressed below baseline in both conditions is not useful — it might look like "less suppressed in move" but that's not a locomotion enhancement. These units are dropped and counted in the output.

Units passing both gates are called **analyzable**.

### Statistical test (Wilcoxon)

For each analyzable unit, the per-trial firing rate at the response peak (±15 ms around the peak of the trial-averaged PETH) is extracted for stat and move trials. A paired Wilcoxon signed-rank test compares these two distributions. P-values are FDR-corrected across all units. Units with q < 0.05 and Δ > 0 are **move-excited**; q < 0.05 and Δ < 0 are **move-suppressed**.

### Canonical parameters

From `src/config/locomotion.py`:

| Parameter | Value | Meaning |
|---|---|---|
| `BASELINE_WINDOW` | −40 to 0 ms | Pre-stim baseline for all effect size calculations |
| `RESP_WINDOW` | 30–120 ms | Window for SNR, Wilcoxon, and peak search |
| `PEAK_HALF_WINDOW_S` | ±15 ms | Half-window around trial-averaged peak for per-trial rate extraction |
| `SNR_THRESHOLD` | 3.0 | Minimum signal-to-noise ratio to include a unit |
| `QVAL_ALPHA` | 0.05 | FDR threshold for the Wilcoxon test |
| `RATE_SPLIT_HZ` | 12 Hz | Boundary between "low" and "high" stimulus frequency subsets |
| `DEPTH_BIN_WIDTH_UM` | 100 µm | Bin width for depth-stratified summaries |

---

### `scripts/locomotion_stat_vs_move.py`

**What it does:** Main locomotion figure. Generates 4 output files.

**Sessions:**
- GRB006 `20240821_121447` — hybrid: spikes from local KS4 pkl, behavior from Chipmunk DB
- GRB058 `20260312_134952` — fully DB-backed

**Outputs:**

| File | Contents |
|---|---|
| `figures/locomotion/scatter.pdf/.png` | Main 6-row × 3-column figure. Per animal: behavior summary, 2 example unit PETHs, scatter of stat vs move response rates (all / low-rate / high-rate trials). Points above the diagonal = move-enhanced. |
| `figures/locomotion/overlay.pdf/.png` | Population-mean PETH overlaid for stat (gray) and move (orange) conditions, for all / low / high trial subsets. |
| `figures/locomotion/timing_ctrl.pdf/.png` | The same scatter under 4 anchor configurations (last stat, 2nd stat, 3rd stat, offset-matched). Shows the effect is robust to anchor choice. |
| `figures/locomotion/depth.pdf/.png` | Scatter points colored by recording depth. |

**Scatter axes:** x = stat-condition response (sp/s, mean in RESP_WINDOW), y = move-condition response. Each point is one unit. The annotation reports: fraction above diagonal among analyzable units, number of units, median Δ (move − stat).

**Rate split:** "low" = trials with stim frequency ≤ 12 Hz, "high" = > 12 Hz. The rate split lets you see whether the locomotion effect depends on how visually demanding the trial was.

---

### `scripts/locomotion_depth_binned.py`

**What it does:** Depth-stratified summary for GRB058. Bins units into 100 µm depth bins, then plots for each bin: fraction that are move-excited, move-suppressed, or no significant effect. Gives a sense of whether superficial vs deep layers show different locomotion modulation.

**Session:** GRB058 `20260312_134952`

**Output:** `figures/locomotion/depth_binned.pdf`

**Note:** GRB006 is excluded here because depth estimates for GRB006 are less reliable (probe positioning was not as well characterized).

---

## Canonical config — changing parameters

All analysis parameters live in two files:

- `src/config/double_peak.py` — double-peak classification
- `src/config/locomotion.py` — locomotion analysis

**Never define these inline in scripts.** If a parameter needs to change, change it in `src/config/` so all scripts move together. A one-line change in one script that doesn't flow through the config will create drift — the same unit will be classified differently by different scripts.

Before changing any parameter that affects which units pass a classifier, consider:
1. Does this change the unit count in `dario_story.pdf`? That figure was sent to a collaborator.
2. Will the same scientific claim still hold across both GRB006 and GRB058?

---

## File organization rules

```
figures/
  double_peak/      ← one file per double-peak script
  locomotion/       ← all locomotion outputs in one flat folder
```

Rules:
- **No session IDs in filenames** — sessions are canonical and documented in scripts; encoding them in filenames forces renames when sessions change
- **No animal IDs** unless the figure is single-animal by design
- **No subfolders within a topic** unless you're comparing multiple distinct experimental versions
- **No redundant prefix** — `locomotion/scatter.pdf` not `locomotion/loco_scatter.pdf`
