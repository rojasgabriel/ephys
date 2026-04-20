"""Depth and waveform-width breakdown of double-peak V1 units.

Sessions in current scope: GRB058 20260312 only.
(GRB059 and GRB060 are not in the current analysis scope.)

History of analyses sent to Marsa Taheri (Slack, Churchland Lab):
  Dec 12 2025 — firing_rate vs spike_duration scatter, colored by is_double.
    Finding: double-peak units cluster at short spike_duration (~0.2–0.45 ms)
    and high firing rate → they are primarily putative Fast-Spiking (FS) neurons.
    Marsa: correct terminology is "putative Fast-Spiking (FS) and Regular-Spiking
    (RS) neuronal populations" (per IBL paper). Also noted inter-peak distances
    are 20–50 ms (sorted: [20 20 20 30 30 30 30 30 40 40 40 40 50]).

  Jan 13 2026 — spike_duration vs depth scatter + depth KDE by is_double.
    Finding: double-peak units tend to be at higher depth values (~3200–3500 µm),
    corresponding to shallower cortex closer to L4 (first recipient of LGN input).
    Gabriel confirmed: higher numeric depth = closer to pia / shallower.

Classification uses canonical params from src/config/double_peak.py
(FDR selectivity + 5 sp/s height floor on both peaks).

Figure layout per session (5 panels):
  A. Scatter: firing_rate (y) vs spike_duration (x), colored by is_double
     → reproduces the Dec 12 FS/RS figure sent to Marsa
  B. Scatter: depth (y) vs spike_duration (x), colored by is_double
     → reproduces the Jan 13 figure sent to Marsa
  C. KDE: depth distribution, hue=is_double
     → reproduces the Jan 13 second figure sent to Marsa
  D. KDE: spike_duration distribution, hue=is_double (FS vs RS proxy)
  E. Double-peak fraction per 100 µm depth bin

Pooled figure: panels A–C across all sessions combined.

Usage
-----
    python scripts/double_peak_depth_waveform.py

Outputs
-------
    figures/double_peak/depth_waveform_{SUBJECT}_{YYYYMMDD}.pdf  (per session)
    figures/double_peak/depth_waveform_pooled.pdf                 (pooled)
"""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import mannwhitneyu

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from labdata.schema import (
    EphysRecording,
    SpikeSorting,
    UnitCount,
    UnitMetrics,
)

from ephys.src.config.double_peak import (
    BASELINE_WINDOW,
    MIN_PEAK_HEIGHT_ABS,
    PEAK_KWARGS,
    PETH_KWARGS,
    SELECTIVITY_KWARGS,
)
from ephys.src.utils.utils_IO import fetch_session_events
from ephys.src.utils.utils_analysis import (
    classify_peak_count,
    compute_population_peth,
    compute_unit_selectivity,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SESSIONS = [
    ("GRB058", "20260312_134952"),
]

FIGURE_DIR = Path("/Users/gabriel/lib/ephys/figures/double_peak")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

UNIT_CRITERIA_ID = 1
NARROW_BROAD_MS = 0.4  # FS/RS boundary, visual reference only
DEPTH_BIN_UM = 100.0

# Colours — consistent with Slack plots
COL_OTHER = "#4C72B0"  # blue  (is_double=False)
COL_DOUBLE = "#DD8452"  # orange (is_double=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _fetch_unit_table(subject: str, session: str) -> pd.DataFrame:
    """Return good units with depth and waveform metrics from UnitMetrics.

    Columns: unit_id, spike_times_s, depth, spike_duration_ms, fw3m,
             spike_amplitude.
    """
    sess_q = (
        SpikeSorting() & f'subject_name = "{subject}"' & f'session_name = "{session}"'
    ).proj()

    good_ids = (
        sess_q
        * (UnitCount.Unit & f"unit_criteria_id = {UNIT_CRITERIA_ID}" & "passes = 1")
    ).fetch("subject_name", "session_name", "unit_id", as_dict=True)

    df = pd.DataFrame(
        ((SpikeSorting.Unit & good_ids) * UnitMetrics).fetch(
            "unit_id",
            "spike_times",
            "depth",
            "spike_duration",
            "fw3m",
            "spike_amplitude",
            "firing_rate",
            as_dict=True,
        )
    )

    srate = float((EphysRecording.ProbeSetting() & sess_q).fetch("sampling_rate")[0])
    df["spike_times_s"] = df["spike_times"].apply(
        lambda st: np.asarray(st, dtype=float) / srate
    )

    # spike_duration: spks.waveforms.compute_waveform_metrics returns ms already,
    # but guard against samples if the value looks large.
    if df["spike_duration"].notna().any():
        med = df["spike_duration"].median()
        df["spike_duration_ms"] = (
            df["spike_duration"] / srate * 1000.0 if med > 100 else df["spike_duration"]
        )
    else:
        df["spike_duration_ms"] = np.nan

    return df.sort_values("depth", ascending=True).reset_index(drop=True)


def _add_double_peak_labels(df: pd.DataFrame, align_ev: dict) -> pd.DataFrame:
    """Classify each unit as is_double=True/False and is_excited=True/False."""
    unit_ids = df["unit_id"].tolist()
    spike_times = df["spike_times_s"].tolist()

    first_stim = align_ev["first_stim_ev_15ms"]
    if len(first_stim) == 0:
        df["is_excited"] = False
        df["is_double"] = False
        return df

    peth, bin_edges, bin_centers = compute_population_peth(
        spike_times_per_unit=spike_times,
        alignment_times=first_stim,
        **PETH_KWARGS,
    )
    _, masks = compute_unit_selectivity(
        peth, bin_edges, unit_ids=unit_ids, **SELECTIVITY_KWARGS
    )

    exc_idx = np.where(masks["excited"])[0]
    exc_ids = [unit_ids[i] for i in exc_idx]
    exc_peth = peth[exc_idx]

    peaks_df = classify_peak_count(
        exc_peth, bin_centers, unit_ids=exc_ids, **PEAK_KWARGS
    )

    bl_mask = (bin_centers >= BASELINE_WINDOW[0]) & (bin_centers < BASELINE_WINDOW[1])
    double_ids: set[int] = set()
    for _, row in peaks_df.loc[peaks_df["n_peaks"] == 2].iterrows():
        uid = int(row["unit"])
        i = exc_ids.index(uid)
        bl = exc_peth[i].mean(axis=0)[bl_mask].mean()
        if min(float(h) - bl for h in row["peak_heights"]) >= MIN_PEAK_HEIGHT_ABS:
            double_ids.add(uid)

    df["is_excited"] = df["unit_id"].isin(set(exc_ids))
    df["is_double"] = df["unit_id"].isin(double_ids)
    return df


# ---------------------------------------------------------------------------
# Individual panels
# ---------------------------------------------------------------------------


def _panel_fr_vs_waveform(ax, df_exc, subject, session):
    """Scatter: firing_rate (y) vs spike_duration (x), colored by is_double.
    Reproduces the Dec 12 2025 figure sent to Marsa.
    Finding: double-peak units are primarily putative FS (narrow waveform).
    """
    other = df_exc[~df_exc["is_double"]]
    double = df_exc[df_exc["is_double"]]

    ax.scatter(
        other["spike_duration_ms"],
        other["firing_rate"],
        s=15,
        alpha=0.45,
        color=COL_OTHER,
        label=f"RS / other (n={len(other)})",
        rasterized=True,
    )
    ax.scatter(
        double["spike_duration_ms"],
        double["firing_rate"],
        s=25,
        alpha=0.85,
        color=COL_DOUBLE,
        label=f"Double-peak (n={len(double)})",
        edgecolors="k",
        linewidths=0.35,
        zorder=3,
    )
    ax.axvline(
        NARROW_BROAD_MS,
        color="k",
        lw=0.7,
        ls="--",
        label=f"FS/RS boundary ({NARROW_BROAD_MS} ms)",
    )
    ax.set_xlabel("Spike duration trough-to-peak (ms)", fontsize=9)
    ax.set_ylabel("Firing rate (sp/s)", fontsize=9)
    ax.set_title(f"{subject} {session[:8]}\nfiring rate vs waveform width", fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.tick_params(labelsize=8)


def _panel_depth_vs_waveform(ax, df_exc, subject="", session=""):
    """Scatter: depth (y) vs spike_duration (x), colored by is_double.
    Reproduces the Jan 13 2026 figure sent to Marsa.
    """
    other = df_exc[~df_exc["is_double"]]
    double = df_exc[df_exc["is_double"]]

    ax.scatter(
        other["spike_duration_ms"],
        other["depth"],
        s=15,
        alpha=0.45,
        color=COL_OTHER,
        label=f"RS / other (n={len(other)})",
        rasterized=True,
    )
    ax.scatter(
        double["spike_duration_ms"],
        double["depth"],
        s=25,
        alpha=0.85,
        color=COL_DOUBLE,
        label=f"Double-peak (n={len(double)})",
        edgecolors="k",
        linewidths=0.35,
        zorder=3,
    )
    ax.axvline(
        NARROW_BROAD_MS,
        color="k",
        lw=0.7,
        ls="--",
        label=f"FS/RS boundary ({NARROW_BROAD_MS} ms)",
    )
    ax.set_xlabel("Spike duration trough-to-peak (ms)", fontsize=9)
    ax.set_ylabel("Depth (µm)", fontsize=9)
    ax.set_title("Spike duration vs depth", fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.tick_params(labelsize=8)


def _panel_depth_kde(ax, df_exc):
    """KDE of depth distribution, hue=is_double.
    Matches plot 2 sent to Marsa on Jan 13 2026.
    """
    for is_dbl, color, label in [
        (False, COL_OTHER, "Other"),
        (True, COL_DOUBLE, "Double-peak"),
    ]:
        vals = df_exc.loc[df_exc["is_double"] == is_dbl, "depth"].dropna()
        if len(vals) >= 3:
            sns.kdeplot(
                vals,
                ax=ax,
                color=color,
                label=f"{label} (n={len(vals)})",
                fill=True,
                alpha=0.25,
                linewidth=1.5,
            )
            ax.axvline(vals.median(), color=color, lw=1.2, ls="--", alpha=0.8)

    ax.set_xlabel("Depth (µm)", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.set_title("Depth distribution by response type", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    ax.tick_params(labelsize=8)

    # Annotate MWU
    d_dp = df_exc.loc[df_exc["is_double"], "depth"].dropna()
    d_sp = df_exc.loc[~df_exc["is_double"], "depth"].dropna()
    if len(d_dp) >= 3 and len(d_sp) >= 3:
        _, p = mannwhitneyu(d_dp, d_sp, alternative="two-sided")
        ax.text(
            0.97,
            0.96,
            f"MWU p={p:.3g}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
        )


def _panel_waveform_kde(ax, df_exc):
    """KDE of spike_duration distribution, hue=is_double."""
    for is_dbl, color, label in [
        (False, COL_OTHER, "Other"),
        (True, COL_DOUBLE, "Double-peak"),
    ]:
        vals = df_exc.loc[df_exc["is_double"] == is_dbl, "spike_duration_ms"].dropna()
        if len(vals) >= 3:
            sns.kdeplot(
                vals,
                ax=ax,
                color=color,
                label=f"{label} (n={len(vals)})",
                fill=True,
                alpha=0.25,
                linewidth=1.5,
            )
            ax.axvline(vals.median(), color=color, lw=1.2, ls="--", alpha=0.8)

    ax.axvline(
        NARROW_BROAD_MS, color="k", lw=0.7, ls=":", label=f"{NARROW_BROAD_MS} ms cutoff"
    )
    ax.set_xlabel("Spike duration (ms)", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.set_title("Waveform width distribution", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    ax.tick_params(labelsize=8)

    d_dp = df_exc.loc[df_exc["is_double"], "spike_duration_ms"].dropna()
    d_sp = df_exc.loc[~df_exc["is_double"], "spike_duration_ms"].dropna()
    if len(d_dp) >= 3 and len(d_sp) >= 3:
        _, p = mannwhitneyu(d_dp, d_sp, alternative="two-sided")
        ax.text(
            0.97,
            0.96,
            f"MWU p={p:.3g}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
        )


def _panel_fraction_by_depth(ax, df_exc, bin_um=DEPTH_BIN_UM):
    """Double-peak fraction per depth bin (horizontal bar chart)."""
    depth_all = df_exc["depth"].dropna()
    depth_dp = df_exc.loc[df_exc["is_double"], "depth"].dropna()

    if len(depth_all) == 0:
        ax.text(
            0.5,
            0.5,
            "No excited units",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.axis("off")
        return

    lo = np.floor(depth_all.min() / bin_um) * bin_um
    hi = np.ceil(depth_all.max() / bin_um) * bin_um
    edges = np.arange(lo, hi + bin_um, bin_um)
    centers = 0.5 * (edges[:-1] + edges[1:])

    n_all, _ = np.histogram(depth_all, bins=edges)
    n_dp, _ = np.histogram(depth_dp, bins=edges)
    frac = np.where(n_all > 0, n_dp / n_all, np.nan)

    bar_w = bin_um * 0.80
    valid = np.isfinite(frac)
    ax.barh(centers[valid], frac[valid], height=bar_w, color=COL_DOUBLE, alpha=0.75)

    for c, n_e, n_d, f in zip(centers, n_all, n_dp, frac):
        if n_e > 0:
            ax.text(
                (f + 0.02) if np.isfinite(f) else 0.02,
                c,
                f"{n_d}/{n_e}",
                va="center",
                fontsize=7,
            )

    ax.set_xlim(0, 1)
    ax.set_xlabel("Double-peak fraction", fontsize=9)
    ax.set_ylabel("Depth (µm)", fontsize=9)
    ax.set_title(f"Prevalence by depth ({int(bin_um)} µm bins)", fontsize=9)
    ax.tick_params(labelsize=8)


# ---------------------------------------------------------------------------
# Per-session figure
# ---------------------------------------------------------------------------


def _session_figure(subject, session, df):
    df_exc = df[df["is_excited"]].copy()
    n_good = len(df)
    n_exc = len(df_exc)
    n_dp = int(df["is_double"].sum())

    fig, axes = plt.subplots(1, 5, figsize=(22, 5), constrained_layout=True)

    _panel_fr_vs_waveform(axes[0], df_exc, subject, session)  # Dec 12 plot
    _panel_depth_vs_waveform(axes[1], df_exc, subject, session)  # Jan 13 plot 1
    _panel_depth_kde(axes[2], df_exc)  # Jan 13 plot 2
    _panel_waveform_kde(axes[3], df_exc)  # spike_duration KDE
    _panel_fraction_by_depth(axes[4], df_exc)  # depth bin fractions

    fig.suptitle(
        f"{subject}  {session}  —  good units: {n_good}  |  "
        f"excited: {n_exc}  |  double-peak: {n_dp} "
        f"({100 * n_dp / max(n_exc, 1):.1f}% of excited)  |  "
        f"putative FS = spike_dur < {NARROW_BROAD_MS} ms",
        fontsize=10,
    )
    return fig


# ---------------------------------------------------------------------------
# Pooled figure
# ---------------------------------------------------------------------------


def _pooled_figure(records):
    pooled = pd.concat([r["df"] for r in records], ignore_index=True)
    df_exc = pooled[pooled["is_excited"]].copy()

    n_dp = int(pooled["is_double"].sum())
    n_exc = len(df_exc)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5), constrained_layout=True)

    _panel_fr_vs_waveform(axes[0], df_exc, "Pooled", "all sessions")  # Dec 12 plot
    _panel_depth_vs_waveform(axes[1], df_exc, "Pooled", "all sessions")  # Jan 13 plot 1
    _panel_depth_kde(axes[2], df_exc)  # Jan 13 plot 2
    _panel_waveform_kde(axes[3], df_exc)  # waveform KDE

    subjects = " / ".join(sorted({r["subject"] for r in records}))
    n_sessions = len(records)
    fig.suptitle(
        f"Pooled — {subjects}  ({n_sessions} sessions)  |  "
        f"excited: {n_exc}  |  double-peak: {n_dp} "
        f"({100 * n_dp / max(n_exc, 1):.1f}% of excited)",
        fontsize=10,
    )
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    records = []

    for subject, session in SESSIONS:
        print(f"\n{'=' * 60}")
        print(f"  {subject} / {session}")
        print(f"{'=' * 60}")

        try:
            df = _fetch_unit_table(subject, session)
        except Exception as e:
            print(f"  ✗ units: {e}")
            continue
        print(
            f"  units: {len(df)}  depth {df['depth'].min():.0f}–{df['depth'].max():.0f} µm"
        )

        try:
            align_ev = fetch_session_events(subject, session)
        except Exception as e:
            print(f"  ✗ events: {e}")
            continue

        df = _add_double_peak_labels(df, align_ev)

        n_exc = int(df["is_excited"].sum())
        n_dp = int(df["is_double"].sum())
        print(f"  excited: {n_exc}  |  double-peak: {n_dp}")

        # Print waveform summary
        for label, mask in [
            ("double-peak", df["is_double"]),
            ("other exc", df["is_excited"] & ~df["is_double"]),
        ]:
            dur = df.loc[mask, "spike_duration_ms"].dropna()
            if len(dur):
                print(
                    f"  spike_duration_ms [{label}]:  "
                    f"median={dur.median():.3f}  "
                    f"IQR=[{dur.quantile(0.25):.3f}, {dur.quantile(0.75):.3f}]"
                    f"  n={len(dur)}"
                )

        fig = _session_figure(subject, session, df)
        out = FIGURE_DIR / f"depth_waveform_{subject}_{session[:8]}.pdf"
        with PdfPages(out) as pdf:
            pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        print(f"  → {out}")

        records.append(dict(subject=subject, session=session, df=df))

    if len(records) >= 2:
        fig_pool = _pooled_figure(records)
        out_pool = FIGURE_DIR / "depth_waveform_pooled.pdf"
        with PdfPages(out_pool) as pdf:
            pdf.savefig(fig_pool, bbox_inches="tight")
        plt.close(fig_pool)
        print(f"\nPooled → {out_pool}")
    else:
        print("\nOnly one session — skipping pooled figure.")


if __name__ == "__main__":
    main()
