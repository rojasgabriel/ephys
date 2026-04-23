# ephys

Analysis code for studying the impact of movements on V1 activity during freely-moving decision making.

## Requirements

- Python 3.10+ (`.python-version` pins the default dev/CI interpreter to 3.11)
- [uv](https://astral.sh/uv)

## Installation

Clone the repository:

    git clone https://github.com/rojasgabriel/ephys.git
    cd ephys

Install dependencies:

    uv sync

Analysis entrypoints are grouped under `scripts/` by role:

- `scripts/maintained/` — canonical figure generators
- `scripts/supporting/` — secondary helpers and reference analyses
- `scripts/diagnostics/` — one-off debug investigations
- `scripts/tools/` — general utilities

Tested on:

- macOS Tahoe 26.3
- Python 3.11
