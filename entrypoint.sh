#!/bin/sh
set -e

if [ "$1" = "1" ]; then
        uv run python -u install_data.py
fi
exec uv run python -u train.py
