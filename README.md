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

Run the lightweight checks used by CI:

    uvx ruff check .
    uvx ruff format --check .
    uv run python -m unittest discover -s tests

Analysis entrypoints are grouped under `scripts/` by role:

- `scripts/analyses/` — scripts that answer scientific questions or generate analysis figures
- `scripts/diagnostics/` — one-off checks of data quality, sync, or pipeline assumptions
- `scripts/tools/` — interactive utilities and browsers

Tested on:

- macOS Tahoe 26.3
- Python 3.11
