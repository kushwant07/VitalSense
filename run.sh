#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

export OPENCV_AVFOUNDATION_SKIP_AUTH=1

# Launch VitalSense in the virtual environment
exec "$DIR/venv/bin/python" "$DIR/main.py" "$@"
