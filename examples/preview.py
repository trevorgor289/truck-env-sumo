"""
Loads the latest checkpoint from a training run's log directory and plays one
episode with the SUMO GUI, so training progress can be watched without
slowing down the actual (headless) training loop.
"""

import argparse
import glob
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stable_baselines3 import A2C, DQN, PPO

from truck_env.highway_env import TruckHighwayEnv
from truck_env.parameters import road_params, sim_params

ALGOS = {"dqn": DQN, "ppo": PPO, "a2c": A2C}
ROAD_PATH = os.path.join(os.path.dirname(__file__), "..", "road")


def latest_checkpoint(log_dir):
    checkpoints = glob.glob(os.path.join(log_dir, "model_*_steps.zip"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {log_dir}")
    return max(checkpoints, key=lambda p: int(re.search(r"model_(\d+)_steps", p).group(1)))


def main(log_dir, algo, deterministic):
    ckpt = latest_checkpoint(log_dir)
    steps = int(re.search(r"model_(\d+)_steps", ckpt).group(1))
    print(f"Previewing checkpoint at {steps} training steps: {ckpt}")

    model = ALGOS[algo].load(ckpt)
    env = TruckHighwayEnv(sim_params, road_params, use_gui=True,
                           road_path=ROAD_PATH, name_suffix="preview")
    try:
        obs, info = env.reset()
        terminated = truncated = False
        episode_reward = 0.0
        episode_steps = 0
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(int(action))
            episode_reward += reward
            episode_steps += 1
        print(f"[preview @ {steps} steps] episode_steps={episode_steps} "
              f"reward={episode_reward:.2f} events={info['events']}")
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir", help="Path to the training run's log directory")
    parser.add_argument("--algo", choices=list(ALGOS), default="dqn")
    parser.add_argument("--stochastic", action="store_true",
                         help="Sample actions instead of taking the greedy/deterministic one")
    args = parser.parse_args()
    main(args.log_dir, args.algo, deterministic=not args.stochastic)
