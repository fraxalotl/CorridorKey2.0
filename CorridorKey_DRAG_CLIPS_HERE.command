#!/usr/bin/env bash
# CorridorKey 2.0 launcher for macOS. Double-click this file in Finder: it
# opens Terminal, then drag a clip or folder into the Terminal window when
# asked. You can also run it from Terminal with clip paths as arguments.
cd "$(dirname "$0")" || exit 1

if [ -x "venv/bin/python" ]; then
    PYTHON="venv/bin/python"
else
    PYTHON="python3"
fi

"$PYTHON" run_interactive.py "$@"

echo
read -r -p "Press Enter to close."
