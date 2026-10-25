#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

export OPENCV_AVFOUNDATION_SKIP_AUTH=1
export PYTHONUTF8=1

if [ -f "$DIR/venv/bin/python" ]; then
    exec "$DIR/venv/bin/python" "$DIR/main.py" "$@"
else
    exec python3 "$DIR/main.py" "$@"
fi
