from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from labdata.schema import DatasetEvents

from ephys.src.utils.double_peak_helpers import resolve_grb006_trial_ts_path
from labdata_plugin.analysisschema import EventMapping

GRB006_KEY = {
    "subject_name": "GRB006",
    "session_name": "20240821_121447",
    "dataset_name": "ephys_g0",
    "stream_name": "nidq",
}

DEFAULT_LOGICAL_EVENT_NAMES = {
    "visual_stim": "io0",
    "trial_start": "io2",
    "frames": "io3",
    "left_port": "io4",
    "center_port": "io5",
    "right_port": "io6",
}

SESSION_MAPPING_SPECS = [
    {
        "subject_name": "GRB006",
        "session_name": "20240821_121447",
        "stream_name": "nidq",
        "source_event_names": {
            "visual_stim": "ai0",
            "trial_start": "2",
            "frames": "1",
            "left_port": "3",
            "center_port": "4",
            "right_port": "5",
        },
    },
    {
        "subject_name": "GRB058",
        "session_name": "20260224_152424",
        "source_event_names": DEFAULT_LOGICAL_EVENT_NAMES,
    },
    {
        "subject_name": "GRB058",
        "session_name": "20260312_134952",
        "source_event_names": DEFAULT_LOGICAL_EVENT_NAMES,
    },
    {
        "subject_name": "GRB058",
        "session_name": "20260319_131303",
        "source_event_names": DEFAULT_LOGICAL_EVENT_NAMES,
    },
]


def fetch_session_rows(subject: str, session: str) -> pd.DataFrame:
    restriction = {"subject_name": subject, "session_name": session}
    rows = pd.DataFrame((DatasetEvents.Digital() & restriction).fetch(as_dict=True))
    if rows.empty:
        raise RuntimeError(
            f"No DatasetEvents.Digital rows found for {subject} {session}"
        )
    return rows


def infer_canonical_stream(rows: pd.DataFrame) -> tuple[str, str]:
    candidates: list[tuple[str, str]] = []
    for stream_name in ("obx", "nidq"):
        mask = (rows["stream_name"] == stream_name) & (rows["event_name"] == "io2")
        if mask.any():
            dataset_name = rows.loc[mask, "dataset_name"].iloc[0]
            candidates.append((str(dataset_name), stream_name))
    if not candidates:
        available = sorted(
            {
                f"{row.dataset_name}:{row.stream_name}:{row.event_name}"
                for row in rows[
                    ["dataset_name", "stream_name", "event_name"]
                ].itertuples(index=False)
            }
        )
        raise RuntimeError(
            "Could not infer a canonical event stream from io2 rows. "
            f"Available rows: {available}"
        )
    return candidates[0]


def load_grb006_visual_onsets(trial_ts_path: Path | None = None) -> np.ndarray:
    trial_ts_path = trial_ts_path or resolve_grb006_trial_ts_path()
    trial_ts = pd.read_pickle(trial_ts_path).reset_index(drop=True)
    if "stim_ts_visual" not in trial_ts.columns:
        raise RuntimeError("trial_ts.pkl does not contain stim_ts_visual")
    visual_stims = [
        np.asarray(stims, dtype=float)
        for stims in trial_ts["stim_ts_visual"].tolist()
        if len(stims)
    ]
    if not visual_stims:
        raise RuntimeError("No visual stim timestamps found in trial_ts.pkl")
    visual_onsets = np.concatenate(visual_stims)
    visual_onsets = visual_onsets[np.isfinite(visual_onsets)]
    if visual_onsets.size == 0:
        raise RuntimeError("No finite visual stim timestamps found in trial_ts.pkl")
    return np.sort(visual_onsets)


def _normalize_event_row(row: dict) -> dict:
    return {
        **row,
        "event_timestamps": np.asarray(row["event_timestamps"], dtype=float),
        "event_values": None
        if row.get("event_values") is None
        else np.asarray(row["event_values"]),
    }


def build_grb006_ai0_row() -> dict:
    visual_onsets = load_grb006_visual_onsets()
    return {
        **GRB006_KEY,
        "event_name": "ai0",
        "event_timestamps": visual_onsets,
        "event_values": np.ones(visual_onsets.shape[0], dtype=np.uint8),
    }


def ensure_grb006_ai0(apply: bool) -> dict:
    row = build_grb006_ai0_row()
    key = {**GRB006_KEY, "event_name": row["event_name"]}
    relation = DatasetEvents.Digital() & key
    if len(relation):
        print(
            f"{'Keeping' if apply else 'Would keep'} existing "
            f"{row['dataset_name']}:{row['stream_name']}:{row['event_name']}"
        )
        return _normalize_event_row((relation).fetch1())
    if apply:
        DatasetEvents.Digital().insert1(row, allow_direct_insert=True)
    print(
        f"{'Inserted' if apply else 'Would insert'} "
        f"{row['dataset_name']}:{row['stream_name']}:{row['event_name']}"
    )
    return _normalize_event_row(row)


def build_event_mapping_rows(
    spec: dict,
    pending_rows: list[dict] | None = None,
) -> list[dict]:
    rows = fetch_session_rows(spec["subject_name"], spec["session_name"]).copy()
    if pending_rows:
        pending_df = pd.DataFrame([_normalize_event_row(row) for row in pending_rows])
        rows = pd.concat([rows, pending_df], ignore_index=True)
        rows = rows.drop_duplicates(
            subset=["dataset_name", "stream_name", "event_name"],
            keep="last",
        ).reset_index(drop=True)
    dataset_name, stream_name = infer_canonical_stream(rows)
    stream_name = spec.get("stream_name", stream_name)
    dataset_name = spec.get("dataset_name", dataset_name)

    mapped_rows: list[dict] = []
    for logical_name, source_event_name in spec["source_event_names"].items():
        mask = (
            (rows["dataset_name"] == dataset_name)
            & (rows["stream_name"] == stream_name)
            & (rows["event_name"] == source_event_name)
        )
        if not mask.any():
            available = sorted(
                {
                    f"{row.dataset_name}:{row.stream_name}:{row.event_name}"
                    for row in rows[
                        ["dataset_name", "stream_name", "event_name"]
                    ].itertuples(index=False)
                }
            )
            raise RuntimeError(
                f"Missing source row for {spec['subject_name']} {spec['session_name']}: "
                f"{dataset_name}:{stream_name}:{source_event_name}. "
                f"Available rows: {available}"
            )
        mapped_rows.append(
            {
                "subject_name": spec["subject_name"],
                "session_name": spec["session_name"],
                "event_name": logical_name,
                "source_dataset_name": dataset_name,
                "source_stream_name": stream_name,
                "source_event_name": source_event_name,
            }
        )
    return mapped_rows


def upsert_event_mapping_rows(rows: list[dict], apply: bool) -> None:
    for row in rows:
        key = {
            "subject_name": row["subject_name"],
            "session_name": row["session_name"],
            "event_name": row["event_name"],
        }
        relation = EventMapping() & key
        if apply and len(relation):
            relation.delete(force=True)
        if apply:
            EventMapping().insert1(row)
        print(
            f"{'Upserted' if apply else 'Would upsert'} EventMapping "
            f"{row['subject_name']} {row['session_name']} {row['event_name']} -> "
            f"{row['source_stream_name']}:{row['source_event_name']}"
        )


def seed_event_mapping(apply: bool) -> None:
    grb006_ai0_row = ensure_grb006_ai0(apply=apply)
    for spec in SESSION_MAPPING_SPECS:
        pending_rows = (
            [grb006_ai0_row]
            if (not apply and spec["subject_name"] == "GRB006")
            else None
        )
        upsert_event_mapping_rows(
            build_event_mapping_rows(spec, pending_rows=pending_rows),
            apply=apply,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seed EventMapping rows and the GRB006 ai0 visual-stim row."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write changes to DataJoint. Without this flag the script only prints the planned actions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_event_mapping(apply=args.apply)


if __name__ == "__main__":
    main()
