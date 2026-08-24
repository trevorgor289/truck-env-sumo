"""
Neurosymbolic DQN training, SCALLOP-ONLY variant: identical to
train_neurosymbolic_dqn.py except the policy is ScallopOnlyDQNPolicy
(neurosymbolic_dqn_policy_scallop_only.py) -- the MLP only produces facts
for Scallop; Scallop's next_action output IS the Q-values directly, no
combination step at all. Only runs under WSL/Linux, since scallopy has no
Windows build.
"""

import argparse
import datetime
import json
import os
import sys

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from truck_env.highway_env import TruckHighwayEnv
from truck_env.neurosymbolic_dqn_policy_scallop_only import ScallopOnlyDQNPolicy
from truck_env.parameters import road_params, sim_params

ROAD_PATH = os.path.join(os.path.dirname(__file__), "..", "road")
LOG_ROOT = os.path.join(os.path.dirname(__file__), "..", "logs")


class EpisodeEventCallback(BaseCallback):
    def __init__(self, log_path):
        super().__init__()
        self.log_path = log_path
        self._episode_idx = 0

    def _on_training_start(self):
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w") as f:
                f.write("timestep,episode,collision,near_collision,outside_road,reached_exit,max_steps\n")

    def _on_step(self):
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            if not done:
                continue
            events = info.get("events", [])
            row = [
                self.num_timesteps, self._episode_idx,
                int("collision" in events), int("near_collision" in events),
                int("outside_road" in events), int("reached_exit" in events),
                int("max_steps" in events),
            ]
            with open(self.log_path, "a") as f:
                f.write(",".join(str(v) for v in row) + "\n")
            self._episode_idx += 1
        return True


def _json_safe(d):
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in d.items()}


def train(timesteps, seed, use_gui, checkpoint_freq=2000, resume=None):
    resuming = resume is not None
    if resuming:
        log_dir = os.path.dirname(resume)
    else:
        start_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(LOG_ROOT, f"dqn_neurosymbolic_scalloponly_seed{seed}_{start_time}")
        os.makedirs(log_dir, exist_ok=True)

        import stable_baselines3
        config = {
            "algo": "dqn_neurosymbolic_scallop_only", "timesteps": timesteps, "seed": seed,
            "use_gui": use_gui, "checkpoint_freq": checkpoint_freq,
            "sb3_version": stable_baselines3.__version__, "sumo_version": "1.27.1",
            "sim_params": _json_safe(sim_params), "road_params": _json_safe(road_params),
            "note": "Policy is ScallopOnlyDQNPolicy: the MLP only produces facts "
                    "(unsafe_gap, leader_close) for Scallop's merge_planner.scl; "
                    "Scallop's next_action output IS the Q-values directly, no "
                    "separate raw-Q pathway and no combination step. Compare "
                    "against dqn_seed0_20260818_093022 (pure DQN baseline), "
                    "dqn_neurosymbolic_seed0_20260819_043657 (fixed log-additive "
                    "combine), and dqn_neurosymbolic_learnedcombine_seed0_20260819_084534 "
                    "(learned combiner).",
        }
        with open(os.path.join(log_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2, default=str)

    env = TruckHighwayEnv(sim_params, road_params, use_gui=use_gui,
                           road_path=ROAD_PATH, name_suffix="dqn_nesy_so")
    check_env(env)
    env = Monitor(env, os.path.join(log_dir, "monitor"), override_existing=not resuming)
    obs, info = env.reset(seed=seed)

    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq, save_path=log_dir, name_prefix="model",
        save_replay_buffer=False, save_vecnormalize=False,
    )
    event_callback = EpisodeEventCallback(os.path.join(log_dir, "events.csv"))
    callback = CallbackList([checkpoint_callback, event_callback])

    if resuming:
        model = DQN.load(resume, env=env, tensorboard_log=log_dir)
        remaining = max(0, timesteps - model.num_timesteps)
        print(f"Resuming from {resume} at timestep {model.num_timesteps}; {remaining} steps remaining.")
        if remaining > 0:
            model.learn(total_timesteps=remaining, callback=callback, reset_num_timesteps=False)
    else:
        model = DQN(ScallopOnlyDQNPolicy, env, verbose=1, seed=seed, tensorboard_log=log_dir)
        model.learn(total_timesteps=timesteps, callback=callback)

    model.save(os.path.join(log_dir, "model_final"))
    env.close()
    print(f"Done. Logs in {log_dir}")
    return log_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--checkpoint-freq", type=int, default=2000)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    train(args.timesteps, args.seed, args.gui, checkpoint_freq=args.checkpoint_freq, resume=args.resume)
