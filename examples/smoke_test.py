"""
Smoke test for truck_env: confirms the env constructs, a random-action policy
runs an episode to termination, and reset works repeatedly without crashing.

Mirrors the original plan's Phase 1 success criterion, now applied to our own
SUMO-1.27.1-based reimplementation instead of the reference repo's code.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from truck_env.highway_env import TruckHighwayEnv
from truck_env.parameters import sim_params, road_params

ROAD_PATH = os.path.join(os.path.dirname(__file__), "..", "road")
NB_RESETS = 100


def main(use_gui):
    env = TruckHighwayEnv(sim_params, road_params, use_gui=use_gui, road_path=ROAD_PATH)
    try:
        obs, info = env.reset(seed=0)
        assert env.observation_space.contains(obs), "initial observation outside observation_space"

        episode_reward = 0.0
        episode_steps = 0
        terminated = truncated = False
        while not (terminated or truncated):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_steps += 1
        print(f"[episode] steps={episode_steps} reward={episode_reward:.2f} "
              f"terminated={terminated} truncated={truncated} events={info['events']}")

        print(f"Resetting {NB_RESETS} times...")
        for i in range(NB_RESETS):
            obs, info = env.reset(seed=i)
            assert obs.shape == env.observation_space.shape
        print(f"OK: {NB_RESETS}/{NB_RESETS} resets succeeded.")

    finally:
        env.close()


if __name__ == "__main__":
    main(use_gui="--gui" in sys.argv)
