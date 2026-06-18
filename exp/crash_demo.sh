#!/usr/bin/env bash
# Demonstrates the system-freeze that occurs when many blis processes are
# launched in parallel on Fedora (observed on F41 and F44).
#
# *** DO NOT RUN ON A MACHINE YOU CAN'T REACH OVER SSH. ***
#
# Symptom: after a few seconds the desktop becomes unresponsive (mouse/kb
# stall, fan ramps to max). Recovery typically requires a hard reboot.
#
# Hypothesis: total RSS of N parallel blis processes exceeds physical RAM.
# Under that pressure the kernel enters a page-reclaim / swap livelock that
# starves interactive processes. systemd-oomd's defaults on Fedora do not
# trigger aggressively enough to kill blis before the desktop locks up.
#
# Mitigations to test (one at a time, after each reboot):
#   1. Install + enable earlyoom:    sudo dnf install earlyoom && sudo systemctl enable --now earlyoom
#   2. Cap N below nproc, e.g. N=2.
#   3. Add a per-process RSS cap:    systemd-run --scope -p MemoryMax=2G ./blis ...
#   4. Set GOMEMLIMIT in the env:    GOMEMLIMIT=1500MiB ./blis ...
#
# Usage:
#   N=2 exp/crash_demo.sh    # try the smallest count first
#   exp/crash_demo.sh        # defaults to nproc — most likely to freeze

set -uo pipefail
cd "$(dirname "$0")/.."

N=${N:-$(nproc)}
LOGDIR="${LOGDIR:-/tmp/blis-crash-demo}"
mkdir -p "$LOGDIR"

echo "Launching $N parallel blis runs. Logs: $LOGDIR"
echo "If the desktop freezes, ssh in and: pkill -9 blis"

for i in $(seq 1 "$N"); do
  ./blis run \
    --model meta-llama/llama-3.3-70b-instruct \
    --workload-spec workloads/inference-perf-interactive-chat.yaml \
    --num-instances 4 --tp 4 --hardware H100 \
    --seed "$i" \
    --post-hoc-detector composite \
    --metrics-path "$LOGDIR/m-$i.json" \
    > "$LOGDIR/run-$i.log" 2>&1 &
done
wait
echo "Done — no freeze this time."
