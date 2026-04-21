#!/bin/bash
cd /home/hice1/wsun377/Desktop/cs8803drl
exec > /home/hice1/wsun377/Desktop/cs8803drl/docs/experiments/artifacts/slurm-logs/056F_retry_v2_$(date +%Y%m%d_%H%M%S).log 2>&1
echo "[$(date)] 056F retry v2 starting, LANE=056F PORT_SEED=71"
LANE=056F PORT_SEED=71 bash scripts/tools/run_with_flags.sh 056F_retry -- bash scripts/eval/resume/_resume_056F_resume.sh
echo "[$(date)] 056F retry v2 EXIT=$?"
