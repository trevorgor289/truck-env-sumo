"""
Neurosymbolic DQN policy, SCALLOP-ONLY variant.

Unlike neurosymbolic_dqn_policy.py (parallel branches + fixed formula) and
neurosymbolic_dqn_policy_learned_combine.py (parallel branches + learned
combiner), this is a strict pipeline, not two branches: the MLP's only job
is to produce the probabilistic facts Scallop reasons over; Scallop's
next_action output directly IS the Q-values, with nothing recombining it
afterward. There is no separate "raw Q-value" pathway at all.

This matches how the actual Scallop PacMan-Maze reference implementation
works (scallop/experiments/pacman_maze/run.py): its CNN never computes
independent action scores, only entity facts, and the DQN loss is applied
directly to the Scallop-derived next_action output
(`state_action_values_raw.gather(1, action_batch)`), no rescaling.
"""

import os

import torch
import torch.nn as nn
from stable_baselines3.dqn.policies import DQNPolicy, QNetwork

import scallopy

SCL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "merge_planner.scl")

NB_LANES = 3
ACTION_SYMBOLS = [
    "SHORT_GAP", "MEDIUM_GAP", "LONG_GAP", "INC_SPEED",
    "DEC_SPEED", "LANE_LEFT", "LANE_RIGHT", "MAINTAIN",
]
NB_ACTIONS = len(ACTION_SYMBOLS)
NB_FACT_OUTPUTS = NB_LANES + 1  # 3x unsafe_gap + 1x leader_close

EGO_LANE_IDX = 2


class ScallopOnlyQNetwork(QNetwork):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # The MLP's only role: raw observation -> fact logits for Scallop.
        # No separate Q-value head exists in this variant.
        self.fact_mlp = nn.Sequential(
            nn.Linear(self.features_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, NB_FACT_OUTPUTS),
        )

        self.planner = scallopy.Module(
            file=SCL_FILE,
            provenance="difftopkproofs",
            k=3,
            input_mappings={
                "ego_lane": [(l,) for l in range(NB_LANES)],
                "unsafe_gap": [(l,) for l in range(NB_LANES)],
                "leader_close": [(0,)],
            },
            output_mappings={"next_action": list(range(NB_ACTIONS))},
            dispatch="serial",
        )

    def _symbolic_facts(self, obs, features):
        ego_lane = obs[:, EGO_LANE_IDX].round().long().clamp(0, NB_LANES - 1)
        ego_lane_onehot = torch.nn.functional.one_hot(ego_lane, NB_LANES).float()

        fact_logits = self.fact_mlp(features)
        unsafe_gap_prob = torch.sigmoid(fact_logits[:, :NB_LANES])
        leader_close_prob = torch.sigmoid(fact_logits[:, NB_LANES:NB_LANES + 1])

        return ego_lane_onehot, unsafe_gap_prob, leader_close_prob

    def forward(self, obs):
        features = self.extract_features(obs, self.features_extractor)
        ego_lane, unsafe_gap, leader_close = self._symbolic_facts(obs.float(), features)

        result = self.planner(ego_lane=ego_lane, unsafe_gap=unsafe_gap, leader_close=leader_close)
        next_action_scores = result["next_action"] if isinstance(result, dict) else result
        return next_action_scores  # used directly as Q-values, no further transform


class ScallopOnlyDQNPolicy(DQNPolicy):
    def make_q_net(self):
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        return ScallopOnlyQNetwork(**net_args).to(self.device)
