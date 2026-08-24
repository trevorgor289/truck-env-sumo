# Future Work

## True image-based CNN+Scallop+DQN in CARLA

Everything in `truck_env` currently skips the perception problem that the
Scallop paper's own PacMan-Maze reference architecture actually solves:
SUMO/traci hands the policy exact ground-truth telemetry (lane index, gaps,
distances) directly, so the "neural head" in `neurosymbolic_dqn_policy*.py`
is just 1-2 tiny `Linear(1,1)` calibration layers turning already-known
scalars into probabilities -- there's no raw-pixel-to-symbol extraction step
like PacMan-Maze's `EntityExtractor` CNN.

Idea: build a true `CNN -> Scallop -> DQN` pipeline (matching PacMan-Maze's
actual architecture: image -> CNN entity extraction -> Scallop reasoning ->
Scallop's output used directly as Q-values) against CARLA
(https://carla.org) instead of SUMO, using CARLA's rendered camera frames as
input. This would require a real CNN doing real perception (e.g. detecting
other vehicles/lane boundaries from pixels), which is the part of the
neurosymbolic story this repo hasn't actually tested yet -- everything so
far only tests "does Scallop reasoning help," not "does Scallop's
differentiability help train a CNN under sparse RL reward," which is the
more interesting claim from Section 6.3/6.5 of the Scallop paper.

Notably, the Scallop paper's own conclusion (Section 8) names this
direction explicitly: "we intend to integrate it with the CARLA driving
simulator to specify soft temporal constraints for autonomous driving
systems" -- so this would be picking up a thread the original authors flagged
but (as far as the paper shows) never published results on.

Open questions to work out when this gets picked up:
- Whether to keep the `ScallopOnlyDQNPolicy`-style structure (Scallop's
  output IS the Q-values, PacMan-Maze's actual approach) or the parallel
  combine style used elsewhere in this repo.
- What `merge_planner.scl`-equivalent rules make sense for CARLA's driving
  task vs. this repo's highway-merge task.
- Whether DQN or PPO is the better fit once perception (not just the
  symbolic layer) is part of what's being learned from sparse reward --
  this repo's own results so far (dqn_neurosymbolic_scalloponly vs. PPO
  baselines) suggest DQN may be materially less stable than PPO regardless
  of the neurosymbolic architecture, which would likely carry over.
