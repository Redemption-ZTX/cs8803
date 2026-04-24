#!/bin/bash
# Usage:
#   SCN=interceptor_subtask M1_CKPT=/path/to/specialist M2=ceia_baseline_agent PORT=62105 LOG=103A_vs_baseline bash run_scenario.sh
#   SCN=defender_subtask M1_CKPT=/path/103B M2_CKPT=/path/1750 PORT=62135 LOG=103B_vs_1750 bash run_scenario.sh
#
# Required env vars:
#   SCN          - scenario name (interceptor_subtask / defender_subtask / dribble_subtask)
#   M1_CKPT      - team0 ckpt path (becomes TRAINED_RAY_CHECKPOINT)
#   M2           - team1 module name (e.g., ceia_baseline_agent)
#                  OR if M2_CKPT is set: M2 defaults to cs8803drl.deployment.trained_team_ray_opponent_agent
#   M2_CKPT      - (optional) team1 ckpt path (becomes TRAINED_TEAM_OPPONENT_CHECKPOINT)
#   PORT         - base port
#   LOG          - log filename suffix
set -e
cd /home/hice1/wsun377/Desktop/cs8803drl
mkdir -p docs/experiments/artifacts/official-evals/scenario-replay
M2=${M2:-ceia_baseline_agent}
if [ -n "$M2_CKPT" ]; then
  M2=cs8803drl.deployment.trained_team_ray_opponent_agent
  export TRAINED_TEAM_OPPONENT_CHECKPOINT=$M2_CKPT
fi
export TRAINED_RAY_CHECKPOINT=$M1_CKPT
exec /home/hice1/wsun377/.venvs/soccertwos_h100/bin/python -m scripts.research.eval_in_scenario \
  --m1 cs8803drl.deployment.trained_team_ray_agent \
  --m2 $M2 \
  --scenario-reset $SCN \
  -n 200 \
  --base-port $PORT \
  --save-log docs/experiments/artifacts/official-evals/scenario-replay/${LOG}.log \
  2>&1 | tee docs/experiments/artifacts/official-evals/scenario-replay/${LOG}_full.log
