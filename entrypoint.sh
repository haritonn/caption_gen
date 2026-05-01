#!/bin/sh
set -e
uv run python -u install_data.py
exec uv run python -u train.py
