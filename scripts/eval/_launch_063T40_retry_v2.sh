#!/bin/bash
cd /home/hice1/wsun377/Desktop/cs8803drl
exec > /home/hice1/wsun377/Desktop/cs8803drl/docs/experiments/artifacts/slurm-logs/063_T40_retry_v2_$(date +%Y%m%d_%H%M%S).log 2>&1
echo "[$(date)] 063_T40 retry v2 starting, PORT_SEED=77"
PORT_SEED=77 bash scripts/tools/run_with_flags.sh 063_T40_retry -- bash scripts/eval/resume/_resume_063_T40_resume.sh
echo "[$(date)] 063_T40 retry v2 EXIT=$?"
