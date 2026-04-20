#!/bin/bash
# Scouting helper: list this user's RUNNING SLURM GPU jobs + per-node load
# signals, so the operator can pick a node manually for run_post_training_pipeline.sh
# or any other srun --overlap invocation.
#
# This script INTENTIONALLY does NOT auto-pick a node. tmux session count
# alone is misleading (existing session ≠ active work — may be a leftover
# from a finished job). The operator is expected to read the table below
# and decide based on context (which lane is currently training, which step
# was just launched, port shelf usage, etc.).
#
# Usage:
#   bash scripts/eval/list_my_gpu_nodes.sh

set -u

USER_NAME="${USER:-$(whoami)}"

echo "==== GPU jobs for user=$USER_NAME (RUNNING only) ===="
echo

JOBS=$(squeue -u "$USER_NAME" --noheader --states=R -o "%A|%N|%M|%L|%P" 2>/dev/null)
if [[ -z "$JOBS" ]]; then
    echo "  (no running jobs found)"
    echo
    echo "If you need a GPU node, allocate one with:"
    echo "  srun --gres=gpu:H100:1 --mem=100G -t 16:00:00 --pty bash"
    exit 0
fi

printf "  %-9s %-22s %-9s %-9s %-12s %-7s %-7s %-13s\n" \
    JobID Node Elapsed Remaining Partition Tmux Python "GPU%/MiB"
printf "  %-9s %-22s %-9s %-9s %-12s %-7s %-7s %-13s\n" \
    --------- ---------------------- --------- --------- ------------ ------- ------- -------------

# Read JOBS into array first to avoid stdin contention with srun
mapfile -t JOB_LINES <<< "$JOBS"

for line in "${JOB_LINES[@]}"; do
    IFS='|' read -r jobid node elapsed remaining partition <<< "$line"
    [[ -z "$jobid" ]] && continue

    # One srun call per job. Use sentinel marker to filter SLURM prolog noise.
    raw=$(timeout 10 srun --jobid="$jobid" --overlap bash -c '
        echo "===BEGIN==="
        tmux ls 2>/dev/null | wc -l
        pgrep -u "'$USER_NAME'" python 2>/dev/null | wc -l
        nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | head -1
        echo "===END==="
    ' 2>/dev/null)
    info=$(echo "$raw" | sed -n '/===BEGIN===/,/===END===/p' | sed '1d;$d')

    tmux_n=$(echo "$info" | sed -n '1p' | tr -d '[:space:]')
    py_n=$(echo "$info"   | sed -n '2p' | tr -d '[:space:]')
    gpu_line=$(echo "$info" | sed -n '3p' | tr -d ' ')

    [[ -z "$tmux_n"   ]] && tmux_n="?"
    [[ -z "$py_n"     ]] && py_n="?"
    [[ -z "$gpu_line" ]] && gpu_line="?"

    printf "  %-9s %-22s %-9s %-9s %-12s %-7s %-7s %-13s\n" \
        "$jobid" "$node" "$elapsed" "$remaining" "$partition" \
        "$tmux_n" "$py_n" "$gpu_line"
done

echo
echo "Caveats:"
echo "  - Tmux session count: residual sessions from finished jobs still appear; not authoritative."
echo "  - Python proc count: most reliable indicator of CPU/memory contention (training = many procs)."
echo "  - GPU%: 0% = node is idle for compute, even if a python process is running."
echo "  - Remaining < 8h is risky for new training (~14h scratch); fine for evals (~30-60min)."
echo
echo "To inspect a specific node's tmux sessions:"
echo "  srun --jobid=<JobID> --overlap tmux ls"
echo
echo "To kill a stale tmux session on a node:"
echo "  srun --jobid=<JobID> --overlap tmux kill-session -t <session_name>"
