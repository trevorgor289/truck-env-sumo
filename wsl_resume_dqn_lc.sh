#!/bin/bash
set -e
export SUMO_HOME=~/truck_env_wsl/.venv/lib/python3.12/site-packages/sumo
source ~/truck_env_wsl/.venv/bin/activate
cd /mnt/c/Users/tgord/SUMO/truck_env
python3 examples/train_neurosymbolic_dqn_learned_combine.py --seed 0 \
  --resume "logs/dqn_neurosymbolic_learnedcombine_seed0_20260819_084534/model_392000_steps.zip"
