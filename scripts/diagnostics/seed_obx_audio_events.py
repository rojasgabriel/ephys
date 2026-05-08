from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from labdata.schema import DatasetEvents
from spks.spikeglx_utils import load_spikeglx_binary

from labdata_plugin.analysisschema import EventMapping


DEFAULT_SESSION = {
    "subject_name": "GRB058",
    "session_name": "20260421_160125",
    "dataset_name": "ephys_g0",
    "stream_name": "obx",
}
DEFAULT_DATA_ROOTS = (
    Path("/Volumes/data"),
    Path.home() / "data",
)
AUDIO_ROLE_NAMES = ("audio_stim", "go_cue", "punish_wrong", "punish_early")


def resolve_obx_bin_path(
    subject: str, session: str, dataset: str, data_root: Path
) -> Path:
    path = data_root / subject / session / dataset / f"{dataset}_t0.obx0.obx.bin"
    if not path.exists():
        raise FileNotFoundError(f"Could not find OBX bin file: {path}")
    return path


def recover_io1_epochs(
    dat: np.ndarray,
    meta: dict,
    channel_index: int,
    bin_ms: float,
    threshold_z: float,
    merge_gap_ms: float,
    min_duration_ms: float,
) -> dict:
    if channel_index >= dat.shape[1]:
        raise ValueError(
            f"Channel index {channel_index} not present in data shape {dat.shape}"
        )

    sample_rate_hz = float(meta["sRateHz"])
    bin_s = bin_ms / 1000.0
    n_bins = int(np.floor(dat.shape[0] / (sample_rate_hz * bin_s)))
    sample_edges = np.rint(np.arange(n_bins + 1) * sample_rate_hz * bin_s).astype(
        np.int64
    )
    sample_edges = np.clip(sample_edges, 0, dat.shape[0])
    group_bins = 5000
    amplitudes: list[np.ndarray] = []
    for bin_start in range(0, n_bins, group_bins):
        bin_stop = min(bin_start + group_bins, n_bins)
        edges = sample_edges[bin_start : bin_stop + 1]
        samples = np.asarray(dat[edges[0] : edges[-1], channel_index])
        local_starts = edges[:-1] - edges[0]
        amplitudes.append(
            np.maximum.reduceat(samples, local_starts).astype(np.float32)
            - np.minimum.reduceat(samples, local_starts).astype(np.float32)
        )

    peak_to_peak = np.concatenate(amplitudes)
    median = float(np.median(peak_to_peak))
    mad = float(np.median(np.abs(peak_to_peak - median)))
    threshold = median + threshold_z * 1.4826 * mad

    active = peak_to_peak > threshold
    run_starts = np.flatnonzero(active & np.r_[True, ~active[:-1]])
    run_stops = np.flatnonzero(active & np.r_[~active[1:], True])
    if run_starts.size == 0:
        return np.array([]), np.array([]), np.array([]), threshold

    min_duration_s = min_duration_ms / 1000.0
    max_gap_s = merge_gap_ms / 1000.0
    starts: list[float] = []
    stops: list[float] = []
    peaks: list[float] = []

    cur_start = int(run_starts[0])
    cur_stop = int(run_stops[0])
    merged_runs: list[tuple[int, int]] = []
    for start, stop in zip(run_starts[1:], run_stops[1:]):
        start = int(start)
        stop = int(stop)
        gap_s = (sample_edges[start] - sample_edges[cur_stop + 1]) / sample_rate_hz
        if gap_s <= max_gap_s:
            cur_stop = stop
        else:
            merged_runs.append((cur_start, cur_stop))
            cur_start = start
            cur_stop = stop
    merged_runs.append((cur_start, cur_stop))

    for start_bin, stop_bin in merged_runs:
        start_s = sample_edges[start_bin] / sample_rate_hz
        stop_s = sample_edges[stop_bin + 1] / sample_rate_hz
        if stop_s - start_s < min_duration_s:
            continue
        starts.append(start_s)
        stops.append(stop_s)
        peaks.append(float(peak_to_peak[start_bin : stop_bin + 1].max()))

    return (
        np.asarray(starts, dtype=float),
        np.asarray(stops, dtype=float),
        np.asarray(peaks, dtype=float),
        threshold,
    )


def build_io1_row(
    key: dict[str, str],
    starts: np.ndarray,
    stops: np.ndarray,
    unknown_fraction_limit: float = 0.05,
) -> dict | None:
    if starts.size == 0:
        return None

    durations = stops - starts
    known = (
        ((durations >= 0.05) & (durations <= 0.25))
        | ((durations >= 0.75) & (durations <= 1.25))
        | ((durations >= 1.75) & (durations <= 2.25))
    )
    unknown_fraction = 1.0 - (np.count_nonzero(known) / durations.size)
    if unknown_fraction > unknown_fraction_limit:
        print(
            f"Skipping recovered io1: {unknown_fraction:.1%} "
            "of epochs have unknown duration."
        )
        return None

    timestamps = np.empty(starts.size * 2, dtype=float)
    timestamps[0::2] = starts
    timestamps[1::2] = stops
    return {
        **key,
        "event_name": "io1",
        "event_timestamps": timestamps,
        "event_values": None,
    }


def classify_audio_epochs(
    starts: np.ndarray, stops: np.ndarray
) -> dict[str, np.ndarray]:
    durations = stops - starts
    event_starts = {
        "audio_stim": starts[(durations >= 0.015) & (durations <= 0.05)],
        "go_cue": starts[(durations >= 0.05) & (durations <= 0.25)],
        "punish_wrong": starts[(durations >= 0.75) & (durations <= 1.25)],
        "punish_early": starts[(durations >= 1.75) & (durations <= 2.25)],
    }
    classified = sum(value.size for value in event_starts.values())
    if classified != starts.size:
        raise RuntimeError(
            f"Classified {classified} of {starts.size} recovered audio epochs."
        )
    return event_starts


def insert_io1_if_missing(row: dict, apply: bool, replace_existing_io1: bool) -> None:
    key = {k: row[k] for k in DatasetEvents.Digital.projkeys}
    relation = DatasetEvents.Digital() & key
    if len(relation):
        existing = relation.fetch1()
        existing_count = len(np.asarray(existing["event_timestamps"]))
        if replace_existing_io1:
            if apply:
                relation.delete(force=True)
                DatasetEvents.Digital().insert1(row, allow_direct_insert=True)
            print(
                f"{'Replaced' if apply else 'Would replace'} existing obx:io1 row "
                f"with {existing_count} old timestamps and "
                f"{len(row['event_timestamps'])} new timestamps."
            )
            return
        print(f"Keeping existing obx:io1 row with {existing_count} timestamps.")
        return

    if apply:
        DatasetEvents.Digital().insert1(row, allow_direct_insert=True)
    print(
        f"{'Inserted' if apply else 'Would insert'} obx:io1 with "
        f"{len(row['event_timestamps'])} timestamps "
        f"({len(row['event_timestamps']) // 2} epochs)."
    )


def delete_derived_audio_event_rows(key: dict[str, str], apply: bool) -> None:
    for event_name in AUDIO_ROLE_NAMES:
        relation = DatasetEvents.Digital() & {**key, "event_name": event_name}
        if len(relation):
            count = len(np.asarray(relation.fetch1("event_timestamps")))
            if apply:
                relation.delete(force=True)
            print(
                f"{'Deleted' if apply else 'Would delete'} derived "
                f"{key['stream_name']}:{event_name} row with {count} timestamps."
            )


def set_audio_event_mapping_rows(
    key: dict[str, str],
    apply: bool,
    replace_existing_audio_mappings: bool,
) -> None:
    obsolete_names = ("task_audio", *AUDIO_ROLE_NAMES)
    for event_name in obsolete_names:
        obsolete_key = {
            "subject_name": key["subject_name"],
            "session_name": key["session_name"],
            "event_name": event_name,
        }
        obsolete = EventMapping() & obsolete_key
        if not len(obsolete):
            continue
        if not replace_existing_audio_mappings:
            raise RuntimeError(
                f"Existing EventMapping row `{event_name}` is present. "
                "Rerun with --replace-audio-mappings to delete it."
            )
        if apply:
            obsolete.delete(safemode=False)
        print(f"{'Deleted' if apply else 'Would delete'} EventMapping {event_name}.")

    row = {
        "subject_name": key["subject_name"],
        "session_name": key["session_name"],
        "event_name": "audio",
        "source_dataset_name": key["dataset_name"],
        "source_stream_name": key["stream_name"],
        "source_event_name": "io1",
    }
    relation = EventMapping() & {
        "subject_name": row["subject_name"],
        "session_name": row["session_name"],
        "event_name": row["event_name"],
    }
    exists = len(relation) > 0
    if apply and not exists:
        EventMapping().insert1(row)
    action = "Kept existing" if exists else "Inserted" if apply else "Would insert"
    print(f"{action} EventMapping audio -> {row['source_stream_name']}:io1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recover low-amplitude OBX XA1 task-audio events and seed them as obx:io1."
    )
    parser.add_argument("--subject", default=DEFAULT_SESSION["subject_name"])
    parser.add_argument("--session", default=DEFAULT_SESSION["session_name"])
    parser.add_argument("--dataset", default=DEFAULT_SESSION["dataset_name"])
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Mounted data root. Defaults to the first existing path in /Volumes/data, ~/data.",
    )
    parser.add_argument(
        "--channel-index", type=int, default=1, help="OBX XA channel index."
    )
    parser.add_argument("--bin-ms", type=float, default=10.0)
    parser.add_argument("--threshold-z", type=float, default=10.0)
    parser.add_argument("--merge-gap-ms", type=float, default=30.0)
    parser.add_argument("--min-duration-ms", type=float, default=50.0)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write the recovered io1 row and audio EventMapping row to DataJoint.",
    )
    parser.add_argument(
        "--replace-existing-io1",
        action="store_true",
        help="Delete and reinsert an existing obx:io1 row with corrected timestamps.",
    )
    parser.add_argument(
        "--replace-audio-mappings",
        action="store_true",
        help="Delete older task_audio/audio-role EventMapping rows before inserting the audio-channel mapping.",
    )
    parser.add_argument(
        "--delete-derived-audio-rows",
        action="store_true",
        help="Delete older derived audio_stim/go_cue/punish_* DatasetEvents.Digital rows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root
    if data_root is None:
        data_root = next((root for root in DEFAULT_DATA_ROOTS if root.exists()), None)
    if data_root is None:
        raise FileNotFoundError(f"No default data root found: {DEFAULT_DATA_ROOTS}")

    key = {
        "subject_name": args.subject,
        "session_name": args.session,
        "dataset_name": args.dataset,
        "stream_name": "obx",
    }
    obx_bin_path = resolve_obx_bin_path(
        args.subject,
        args.session,
        args.dataset,
        data_root,
    )
    dat, meta = load_spikeglx_binary(obx_bin_path)
    starts, stops, _, _ = recover_io1_epochs(
        dat,
        meta,
        channel_index=args.channel_index,
        bin_ms=args.bin_ms,
        threshold_z=args.threshold_z,
        merge_gap_ms=args.merge_gap_ms,
        min_duration_ms=args.min_duration_ms,
    )
    row = build_io1_row(key, starts, stops)
    if row is None:
        raise RuntimeError("Could not recover a valid obx:io1 audio event row.")
    insert_io1_if_missing(
        row,
        apply=args.apply,
        replace_existing_io1=args.replace_existing_io1,
    )
    counts = {
        event_name: timestamps.size
        for event_name, timestamps in classify_audio_epochs(starts, stops).items()
    }
    print(f"Parsed audio roles from obx:io1: {counts}")
    if args.delete_derived_audio_rows:
        delete_derived_audio_event_rows(key, apply=args.apply)
    set_audio_event_mapping_rows(
        key,
        apply=args.apply,
        replace_existing_audio_mappings=args.replace_audio_mappings,
    )


if __name__ == "__main__":
    main()
