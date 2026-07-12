#!/bin/bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: bash go.sh <pbs_script>" >&2
  exit 1
fi

PBS_SCRIPT="$1"

if [[ ! -f "$PBS_SCRIPT" ]]; then
  echo "pbs script not found: $PBS_SCRIPT" >&2
  exit 1
fi

JOB_BASENAME="$(basename "$PBS_SCRIPT" .pbs)"

rm -f "${JOB_BASENAME}"*
qsub "$PBS_SCRIPT"
sleep 10
qstat
sleep 10 
qstat
sleep 10 
qstat
