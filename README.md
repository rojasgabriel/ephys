# ephys

Analysis code for studying the impact of movements on V1 activity during freely-moving decision making.

## Requirements

- [uv](https://astral.sh/uv)

## Installation

Clone the repository:

    git clone https://github.com/rojasgabriel/ephys.git
    cd ephys

Install dependencies and venv:

    uv sync

Analysis entrypoints are grouped under `scripts/` by role:

- `scripts/analyses/` — scripts that answer scientific questions or generate analysis figures
- `scripts/diagnostics/` — one-off checks of data quality, sync, or pipeline assumptions
- `scripts/tools/` — interactive utilities and browsers

Tested on:

- MacBook Pro M2
- macOS Tahoe 26.3.1 (a)
- Python 3.11
