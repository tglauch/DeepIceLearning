#!/bin/bash

echo $HOSTNAME
eval `/data/user/tglauch/DeepIceLearning/python_env/bin/activate`
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
SDIR="$(dirname "$DIR")"

source /data/user/tglauch/Software/combo/build/env-shell.sh python "$SDIR/create_data_files.py" $@

