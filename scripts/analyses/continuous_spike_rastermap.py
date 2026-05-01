"""Run Rastermap on trial-window continuous spike activity."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import types

import matplotlib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if "ephys" not in sys.modules:
    package = types.ModuleType("ephys")
    package.__path__ = [str(REPO_ROOT)]
    sys.modules["ephys"] = package
sys.path.insert(0, str(REPO_ROOT))

from ephys.src.utils.analysis_rastermap import (  # noqa: E402
    DEFAULT_BIN_MS,
    RastermapResult,
    fit_session_continuous_rastermap,
    heatmap_for_result,
)

DEFAULT_SUBJECT_SESSIONS = [
    ("GRB006", "20240821_121447"),
    ("GRB058", "20260312_134952"),
]
FIGURE_ROOT = Path(
    os.environ.get("EPHYS_FIGURE_ROOT", "/Users/gabriel/lib/ephys/figures")
)
DEFAULT_OUTPUT_DIR = FIGURE_ROOT / "rastermap"
DEFAULT_FIGURE_NAME = "trial_window_spike_rastermap.pdf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subject",
        help="Run one subject/session instead of the default SfN comparison sessions.",
    )
    parser.add_argument(
        "--session",
        help="Run one subject/session instead of the default SfN comparison sessions.",
    )
    parser.add_argument(
        "--bin-ms",
        type=float,
        default=DEFAULT_BIN_MS,
        help=f"Continuous spike bin size in milliseconds. Default: {DEFAULT_BIN_MS:g}.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for PDF and NPZ outputs. Default: {DEFAULT_OUTPUT_DIR}.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Run Rastermap and build the figure without writing PDF or NPZ files.",
    )
    parser.add_argument(
        "--full-recording",
        action="store_true",
        help="Use the full recording instead of concatenated trial windows.",
    )
    return parser.parse_args()


def subject_sessions_from_args(args: argparse.Namespace) -> list[tuple[str, str]]:
    if bool(args.subject) != bool(args.session):
        raise ValueError("--subject and --session must be provided together.")
    if args.subject and args.session:
        return [(args.subject, args.session)]
    return DEFAULT_SUBJECT_SESSIONS


def save_result_npz(result: RastermapResult, output_dir: Path) -> Path:
    output_path = output_dir / f"{result.subject}_{result.session}_rastermap.npz"
    payload = {
        "unit_ids": result.unit_ids,
        "depth": result.depth,
        "bin_edges_s": result.bin_edges_s,
        "spike_counts": result.spike_counts,
        "isort": result.isort,
        "embedding": result.embedding,
    }
    if result.trial_idx_by_bin is not None:
        payload["trial_idx_by_bin"] = result.trial_idx_by_bin
    if result.absolute_bin_start_s is not None:
        payload["absolute_bin_start_s"] = result.absolute_bin_start_s
    if result.absolute_bin_stop_s is not None:
        payload["absolute_bin_stop_s"] = result.absolute_bin_stop_s
    if result.x_embedding is not None:
        payload["X_embedding"] = result.x_embedding
    np.savez_compressed(output_path, **payload)
    return output_path


def plot_results(
    results: list[RastermapResult], *, bin_ms: float, full_recording: bool
):
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(
        len(results),
        1,
        figsize=(11.0, 4.0 * len(results)),
        squeeze=False,
        constrained_layout=True,
    )
    for ax, result in zip(axes[:, 0], results, strict=True):
        heatmap = heatmap_for_result(result)
        vmax = max(1.0, float(np.nanpercentile(heatmap, 99)))
        image = ax.imshow(
            heatmap,
            aspect="auto",
            cmap="gray_r",
            interpolation="nearest",
            vmin=0,
            vmax=vmax,
        )
        n_bins = result.spike_counts.shape[1]
        duration_min = (n_bins * bin_ms / 1000.0) / 60.0
        title_context = "full recording" if full_recording else "trial windows"
        ax.set_title(
            f"{result.subject} {result.session}: "
            f"{len(result.unit_ids)} units, {duration_min:.1f} min {title_context}"
        )
        ax.set_xlabel(f"Time bin ({bin_ms:g} ms)")
        ax.set_ylabel("Rastermap-sorted units")
        fig.colorbar(image, ax=ax, fraction=0.025, pad=0.01, label="Activity")
    return fig


def main() -> None:
    args = parse_args()
    matplotlib.use("Agg")
    subject_sessions = subject_sessions_from_args(args)

    results = []
    for subject, session in subject_sessions:
        print(f"\nRunning Rastermap: {subject} {session}")
        result = fit_session_continuous_rastermap(
            subject,
            session,
            bin_ms=args.bin_ms,
            trial_window_only=not args.full_recording,
        )
        print(
            f"  Matrix: {result.spike_counts.shape[0]} units x "
            f"{result.spike_counts.shape[1]} time bins"
        )
        print(f"  n_clusters: {result.n_clusters}")
        results.append(result)

    fig = plot_results(results, bin_ms=args.bin_ms, full_recording=args.full_recording)
    if not args.no_save:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        figure_path = args.output_dir / DEFAULT_FIGURE_NAME
        fig.savefig(figure_path, bbox_inches="tight", dpi=300)
        print(f"\nSaved -> {figure_path}")
        for result in results:
            npz_path = save_result_npz(result, args.output_dir)
            print(f"Saved -> {npz_path}")


if __name__ == "__main__":
    main()
