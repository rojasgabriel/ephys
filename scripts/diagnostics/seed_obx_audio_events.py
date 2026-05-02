from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from labdata.schema import DatasetEvents
from spks.spikeglx_utils import load_spikeglx_binary


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
    bin_samples = int(round(sample_rate_hz * bin_s))
    n_bins = dat.shape[0] // bin_samples
    group_bins = 5000
    amplitudes: list[np.ndarray] = []
    for bin_start in range(0, n_bins, group_bins):
        bin_stop = min(bin_start + group_bins, n_bins)
        samples = dat[
            bin_start * bin_samples : bin_stop * bin_samples,
            channel_index,
        ].reshape(bin_stop - bin_start, bin_samples)
        amplitudes.append(
            samples.max(axis=1).astype(np.float32)
            - samples.min(axis=1).astype(np.float32)
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
    max_gap_bins = int(round((merge_gap_ms / 1000.0) / bin_s))
    starts: list[float] = []
    stops: list[float] = []
    peaks: list[float] = []

    cur_start = int(run_starts[0])
    cur_stop = int(run_stops[0])
    merged_runs: list[tuple[int, int]] = []
    for start, stop in zip(run_starts[1:], run_stops[1:]):
        start = int(start)
        stop = int(stop)
        if start - cur_stop - 1 <= max_gap_bins:
            cur_stop = stop
        else:
            merged_runs.append((cur_start, cur_stop))
            cur_start = start
            cur_stop = stop
    merged_runs.append((cur_start, cur_stop))

    for start_bin, stop_bin in merged_runs:
        start_s = start_bin * bin_s
        stop_s = (stop_bin + 1) * bin_s
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


def insert_io1_if_missing(row: dict, apply: bool) -> None:
    key = {k: row[k] for k in DatasetEvents.Digital.projkeys}
    relation = DatasetEvents.Digital() & key
    if len(relation):
        existing = relation.fetch1()
        existing_count = len(np.asarray(existing["event_timestamps"]))
        print(f"Keeping existing obx:io1 row with {existing_count} timestamps.")
        return

    if apply:
        DatasetEvents.Digital().insert1(row, allow_direct_insert=True)
    print(
        f"{'Inserted' if apply else 'Would insert'} obx:io1 with "
        f"{len(row['event_timestamps'])} timestamps "
        f"({len(row['event_timestamps']) // 2} epochs)."
    )


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
        help="Write the recovered obx:io1 row to DatasetEvents.Digital.",
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
    insert_io1_if_missing(row, apply=args.apply)


if __name__ == "__main__":
    main()
