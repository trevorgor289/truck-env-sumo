"""
Phase 3/4 training script for truck_env.

Mirrors the reference repo's training setup (examples/dqn.py, ppo.py, a2c.py in
new_architecture) as closely as possible: SB3 "MlpPolicy" with no hyperparameter
overrides (library defaults), use_gui=False, 1e6 timesteps. The one deliberate
deviation from the paper: SB3 here is 2.9.0, not the paper's pinned 1.6.2, so
"defaults" are today's SB3 defaults, not theirs -- noted in config.json for
every run.

Algorithm is a single CLI flag (--algo), per the plan's requirement that
swapping DQN/PPO/A2C not require touching code (Phase 4).

Checkpoints are saved frequently (every --checkpoint-freq steps, default 2000
-- a few tens of seconds at this env's throughput) since the policy network is
tiny, so an unattended run surviving a crash/shutdown only loses a small
amount of progress. Use --resume <path to a model_*_steps.zip> to continue an
interrupted run from its last checkpoint, appending to the same log files
instead of starting a new experiment.
"""

import argparse
import datetime
import json
import os
import sys

import numpy as np
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from truck_env.highway_env import TruckHighwayEnv
from truck_env.parameters import road_params, sim_params

ALGOS = {"dqn": DQN, "ppo": PPO, "a2c": A2C}
ROAD_PATH = os.path.join(os.path.dirname(__file__), "..", "road")
LOG_ROOT = os.path.join(os.path.dirname(__file__), "..", "logs")


class EpisodeEventCallback(BaseCallback):
    """Logs one row per finished episode: timestep, collision/near-collision/
    outside-road/reached-exit/max-steps -- the event categories our env's
    step() reports -- so collision rate can be computed over training."""

    def __init__(self, log_path):
        super().__init__()
        self.log_path = log_path
        self._episode_idx = 0
        if os.path.exists(self.log_path):
            with open(self.log_path) as f:
                self._episode_idx = max(0, sum(1 for _ in f) - 1)  # resume: continue numbering, skip header

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
                self.num_timesteps,
                self._episode_idx,
                int("collision" in events),
                int("near_collision" in events),
                int("outside_road" in events),
                int("reached_exit" in events),
                int("max_steps" in events),
            ]
            with open(self.log_path, "a") as f:
                f.write(",".join(str(v) for v in row) + "\n")
            self._episode_idx += 1
        return True


def _json_safe(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        else:
            out[k] = v
    return out


def train(algo, timesteps, seed, use_gui, checkpoint_freq=2000, resume=None):
    resuming = resume is not None
    if resuming:
        log_dir = os.path.dirname(resume)
    else:
        start_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(LOG_ROOT, f"{algo}_seed{seed}_{start_time}")
        os.makedirs(log_dir, exist_ok=True)

        import stable_baselines3

        config = {
            "algo": algo,
            "timesteps": timesteps,
            "seed": seed,
            "use_gui": use_gui,
            "checkpoint_freq": checkpoint_freq,
            "sb3_version": stable_baselines3.__version__,
            "sumo_version": "1.27.1",
            "sim_params": _json_safe(sim_params),
            "road_params": _json_safe(road_params),
            "note": "SB3 hyperparameters left at library defaults, matching the "
                    "reference paper's methodology -- but this is SB3 2.9.0's "
                    "defaults, not the paper's pinned 1.6.2.",
        }
        with open(os.path.join(log_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2, default=str)

    env = TruckHighwayEnv(sim_params, road_params, use_gui=use_gui,
                           road_path=ROAD_PATH, name_suffix=f"{algo}_{seed}")
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
        model = ALGOS[algo].load(resume, env=env, tensorboard_log=log_dir)
        remaining = max(0, timesteps - model.num_timesteps)
        print(f"Resuming from {resume} at timestep {model.num_timesteps}; "
              f"{remaining} steps remaining to reach target {timesteps}.")
        if remaining > 0:
            model.learn(total_timesteps=remaining, callback=callback, reset_num_timesteps=False)
    else:
        model = ALGOS[algo]("MlpPolicy", env, verbose=1, seed=seed,
                             tensorboard_log=log_dir)
        model.learn(total_timesteps=timesteps, callback=callback)

    model.save(os.path.join(log_dir, "model_final"))
    env.close()
    print(f"Done. Logs in {log_dir}")
    return log_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=list(ALGOS), default="dqn")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--checkpoint-freq", type=int, default=2000,
                         help="Save a checkpoint every N steps (default 2000, ~30s at this env's throughput).")
    parser.add_argument("--resume", type=str, default=None,
                         help="Path to a model_*_steps.zip checkpoint to resume training from.")
    args = parser.parse_args()

    train(args.algo, args.timesteps, args.seed, args.gui,
          checkpoint_freq=args.checkpoint_freq, resume=args.resume)
