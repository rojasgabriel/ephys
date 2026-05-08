from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np
from spks.spikeglx_utils import load_spikeglx_binary

from scripts.diagnostics.seed_obx_audio_events import (
    DEFAULT_DATA_ROOTS,
    recover_io1_epochs,
    resolve_obx_bin_path,
)


ROLE_WINDOWS = {
    "audio_stim": (0.015, 0.050),
    "go_cue": (0.050, 0.250),
    "punish_wrong": (0.750, 1.250),
    "punish_early": (1.750, 2.250),
}
ROLE_COLORS = {
    "audio_stim": "tab:cyan",
    "go_cue": "tab:green",
    "punish_wrong": "tab:red",
    "punish_early": "tab:orange",
    "unknown": "0.5",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot corrected raw OBX audio-event duration distributions."
    )
    parser.add_argument("--subject", default="GRB058")
    parser.add_argument("--session", default="20260421_160125")
    parser.add_argument("--dataset", default="ephys_g0")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Mounted data root. Defaults to first existing path in /Volumes/data, ~/data.",
    )
    parser.add_argument("--channel-index", type=int, default=1)
    parser.add_argument("--bin-ms", type=float, default=10.0)
    parser.add_argument("--threshold-z", type=float, default=10.0)
    parser.add_argument("--merge-gap-ms", type=float, default=30.0)
    parser.add_argument("--min-duration-ms", type=float, default=50.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/diagnostics/audio_event_durations.pdf"),
    )
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def classify_durations(durations: np.ndarray) -> dict[str, np.ndarray]:
    classified = {}
    known = np.zeros(durations.shape, dtype=bool)
    for role, (lo, hi) in ROLE_WINDOWS.items():
        mask = (durations >= lo) & (durations <= hi)
        classified[role] = durations[mask]
        known |= mask
    classified["unknown"] = durations[~known]
    return classified


def recover_durations(args: argparse.Namespace) -> tuple[dict[str, np.ndarray], float]:
    data_root = args.data_root
    if data_root is None:
        data_root = next((root for root in DEFAULT_DATA_ROOTS if root.exists()), None)
    if data_root is None:
        raise FileNotFoundError(f"No default data root found: {DEFAULT_DATA_ROOTS}")

    obx_bin_path = resolve_obx_bin_path(
        args.subject,
        args.session,
        args.dataset,
        data_root,
    )
    dat, meta = load_spikeglx_binary(obx_bin_path)
    starts, stops, _, threshold = recover_io1_epochs(
        dat,
        meta,
        channel_index=args.channel_index,
        bin_ms=args.bin_ms,
        threshold_z=args.threshold_z,
        merge_gap_ms=args.merge_gap_ms,
        min_duration_ms=args.min_duration_ms,
    )
    return classify_durations(stops - starts), threshold


def print_duration_summary(classified: dict[str, np.ndarray], threshold: float) -> None:
    print(f"Detection threshold p2p: {threshold:.3f}")
    for role in (*ROLE_WINDOWS, "unknown"):
        durations = classified[role]
        if durations.size:
            print(
                f"{role}: n={durations.size}, "
                f"min={durations.min():.4f}s, "
                f"median={np.median(durations):.4f}s, "
                f"max={durations.max():.4f}s"
            )
        else:
            print(f"{role}: n=0")


def plot_duration_distributions(classified: dict[str, np.ndarray]) -> plt.Figure:
    roles = [role for role in ROLE_WINDOWS if classified[role].size]
    if classified["unknown"].size:
        roles.append("unknown")
    if not roles:
        raise ValueError("No recovered audio durations to plot.")

    fig, axes = plt.subplots(
        len(roles),
        1,
        figsize=(6.5, 1.8 * len(roles)),
        sharex=False,
    )
    if len(roles) == 1:
        axes = [axes]

    for ax, role in zip(axes, roles):
        durations = classified[role]
        if role in ROLE_WINDOWS:
            lo, hi = ROLE_WINDOWS[role]
            x_min = max(0.0, lo - 0.02)
            x_max = hi + 0.02
        else:
            x_min = max(0.0, float(durations.min()) - 0.02)
            x_max = float(durations.max()) + 0.02
        bins = np.arange(x_min, x_max + 0.005, 0.005)
        ax.hist(
            durations,
            bins=bins,
            color=ROLE_COLORS[role],
            alpha=0.75,
            edgecolor="none",
        )
        if role in ROLE_WINDOWS:
            ax.axvspan(lo, hi, color=ROLE_COLORS[role], alpha=0.12)
            ax.axvline(lo, color=ROLE_COLORS[role], linewidth=1)
            ax.axvline(hi, color=ROLE_COLORS[role], linewidth=1)
        ax.set_ylabel(f"{role}\nn={durations.size}")
        ax.set_xlim(x_min, x_max)
        ax.grid(axis="x", alpha=0.2)

    axes[-1].set_xlabel("audio epoch duration (s)")
    fig.tight_layout()
    return fig


def main() -> None:
    args = parse_args()
    classified, threshold = recover_durations(args)
    print_duration_summary(classified, threshold)
    fig = plot_duration_distributions(classified)
    if not args.no_save:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output)
        print(f"Saved {args.output}")
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
