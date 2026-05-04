from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np

from src.utils.io_digital_events import fetch_session_events


DEFAULT_TRIALS = (0, 25, 50, 75, 100)
DEFAULT_EVENTS = (
    "center_entry",
    "visual_stim",
    "go_cue",
    "center_exit",
    "punish_early",
    "left_port",
    "right_port",
    "punish_wrong",
)
EVENT_COLORS = {
    "center_entry": "tab:brown",
    "visual_stim": "tab:blue",
    "go_cue": "tab:green",
    "center_exit": "tab:brown",
    "punish_early": "tab:orange",
    "left_port": "tab:purple",
    "right_port": "tab:pink",
    "punish_wrong": "tab:red",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot trial-aligned hardware event timestamps for example trials."
    )
    parser.add_argument("--subject", default="GRB058")
    parser.add_argument("--session", default="20260421_160125")
    parser.add_argument(
        "--trials",
        type=int,
        nargs="+",
        default=DEFAULT_TRIALS,
        help="Zero-based trial indices to plot.",
    )
    parser.add_argument(
        "--window",
        type=float,
        nargs=2,
        default=(-0.1, 2.0),
        metavar=("START_S", "STOP_S"),
        help="Time window around fixation / center entry.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/diagnostics/trial_event_timestamps.pdf"),
        help="Output figure path.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the figure interactively after saving.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save the figure.",
    )
    return parser.parse_args()


def build_plot_events(events: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "center_entry": events["center_port"],
        "visual_stim": events["stim_ev"],
        "go_cue": events["go_cue"],
        "center_exit": events["center_port_exit"],
        "punish_early": events["punish_early"],
        "left_port": events["left_port"],
        "right_port": events["right_port"],
        "punish_wrong": events["punish_wrong"],
    }


def plot_trial_event_timestamps(
    subject: str,
    session: str,
    trial_indices: np.ndarray,
    window_s: tuple[float, float],
) -> plt.Figure:
    events = fetch_session_events(subject, session)
    center_entry = np.asarray(events["center_port"], dtype=float)
    trial_indices = trial_indices[trial_indices < len(center_entry)]
    if trial_indices.size == 0:
        raise ValueError(
            "No requested trial indices are present in center_port events."
        )

    plot_events = build_plot_events(events)
    fig, axes = plt.subplots(
        len(trial_indices),
        1,
        figsize=(6, 1.6 * len(trial_indices)),
        sharex=True,
    )
    if len(trial_indices) == 1:
        axes = [axes]

    for ax, trial_idx in zip(axes, trial_indices):
        t0 = center_entry[trial_idx]
        for y, (event_name, timestamps) in enumerate(plot_events.items(), start=1):
            rel = np.asarray(timestamps, dtype=float) - t0
            rel = rel[(window_s[0] <= rel) & (rel <= window_s[1])]
            if rel.size:
                ax.vlines(
                    rel,
                    y - 0.38,
                    y + 0.38,
                    color=EVENT_COLORS[event_name],
                    linewidth=1.2,
                )

        ax.set_yticks(np.arange(1, len(plot_events) + 1))
        ax.set_yticklabels(plot_events.keys())
        ax.set_ylim(0.4, len(plot_events) + 0.6)
        ax.invert_yaxis()
        ax.set_ylabel(f"trial {trial_idx}")
        ax.grid(axis="x", alpha=0.2)

    axes[-1].set_xlim(*window_s)
    axes[-1].set_xlabel("time from fixation (s)")
    fig.tight_layout()
    return fig


def main() -> None:
    args = parse_args()
    fig = plot_trial_event_timestamps(
        args.subject,
        args.session,
        np.asarray(args.trials, dtype=int),
        tuple(args.window),
    )
    if not args.no_save:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output)
        print(f"Saved {args.output}")
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
