#!/usr/bin/env bash
# Wait for another tmux session's job to finish, then run a config.
#
#   ./nrun -t gated 'bash scripts/queue_run.sh paper configs/examples/eplb_gated.toml'
#
# Runs occupy every NPU on the node, so a second one has to wait rather than
# contend: a neighbouring job costs ~3% TPOT and ~31% TTFT, which is the size of
# the effects being measured. The wait is bounded, and on timeout this declines
# to launch rather than starting a run that would overlap.
#
# The session is matched exactly ('-t =name'), because tmux's default target
# matching is a prefix match and a waiter called 'gated' would otherwise be able
# to match itself.
set -u

wait_for=${1:?usage: queue_run.sh <tmux-session-to-wait-for> <config> [max-minutes]}
config=${2:?usage: queue_run.sh <tmux-session-to-wait-for> <config> [max-minutes]}
max_minutes=${3:-480}

echo "waiting for tmux session '$wait_for' to finish (up to $max_minutes min)"
waited=0
while tmux has-session -t "=$wait_for" 2>/dev/null; do
    if [ "$waited" -ge "$max_minutes" ]; then
        echo "'$wait_for' is still running after $max_minutes min - not launching $config"
        exit 1
    fi
    sleep 120
    waited=$((waited + 2))
done
echo "'$wait_for' finished after ${waited} min of waiting"

# The gate is a patch on the installed vllm-ascend, and uv sync reverts it. A
# gated run refuses to start without it, but installing it here keeps the
# sequence self-contained; --check then confirms it landed before any NPU time
# is spent.
uv run python scripts/apply_node_patch.py || exit 1
uv run python scripts/apply_node_patch.py --check || exit 1

echo "starting $config"
exec uv run moe-reliability run "$config"
