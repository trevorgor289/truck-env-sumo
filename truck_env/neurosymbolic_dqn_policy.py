"""
Neurosymbolic DQN policy: subclasses SB3's QNetwork so its forward() returns
Q-values combined with Scallop's merge_planner.scl next_action distribution,
instead of raw neural Q-values alone.

DQN's architecture makes this integration point cleaner than PPO's: SB3's
DQN routes every use of the Q-function -- action selection (_predict),
the online network's TD loss (self.q_net(obs)), and the target computation
(self.q_net_target(next_obs)) -- through a single QNetwork.forward() call.
Since both self.q_net and self.q_net_target are created via the same
make_q_net() factory, overriding just that one method applies the
neurosymbolic combination consistently to both networks automatically.

Combination: combined_q = neural_q + log(scallop_next_action_prob + eps).
Same log-additive shape as the PPO policy (neurosymbolic_policy.py) -- an
action Scallop rates as fully safe (prob ~1) leaves Q nearly unchanged;
one it rates as unsafe (prob ~0) gets its Q-value crushed toward -inf, so
argmax (and, over training, the learned values themselves) avoid it.
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

# Same observation layout as neurosymbolic_policy.py (see highway_env.py _sensor_model)
NB_EGO_STATES = 6
NB_STATES_PER_VEHICLE = 7
EGO_LANE_IDX = 2
LEADER_DIST_IDX = 5


class MergePlannerQNetwork(QNetwork):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.unsafe_gap_head = nn.Linear(1, 1)
        self.leader_close_head = nn.Linear(1, 1)

        self.planner = scallopy.Module(
            file=SCL_FILE,
            provenance="difftopkproofs",
            k=3,
            input_mappings={
                "ego_lane": [(l,) for l in range(NB_LANES)],
                "unsafe_gap": [(l,) for l in range(NB_LANES)],
                "leader_close": [(0,)],
            },
            output_mappings={"next_action": list(range(len(ACTION_SYMBOLS)))},
            dispatch="serial",
        )

    def _per_lane_min_gap(self, obs):
        batch = obs.shape[0]
        vehicles = obs[:, NB_EGO_STATES:].reshape(batch, -1, NB_STATES_PER_VEHICLE)
        rel_dist = vehicles[:, :, 0].abs()
        lane = vehicles[:, :, 4].round().long()

        min_gap = torch.full((batch, NB_LANES), 1e6, device=obs.device)
        for l in range(NB_LANES):
            mask = (lane == l)
            dist_l = torch.where(mask, rel_dist, torch.full_like(rel_dist, 1e6))
            min_gap[:, l] = dist_l.min(dim=1).values
        return min_gap

    def _symbolic_facts(self, obs):
        ego_lane = obs[:, EGO_LANE_IDX].round().long().clamp(0, NB_LANES - 1)
        ego_lane_onehot = torch.nn.functional.one_hot(ego_lane, NB_LANES).float()

        min_gap = self._per_lane_min_gap(obs)
        unsafe_gap_prob = torch.sigmoid(-self.unsafe_gap_head(min_gap.unsqueeze(-1)).squeeze(-1))

        leader_dist = obs[:, LEADER_DIST_IDX:LEADER_DIST_IDX + 1]
        leader_close_prob = torch.sigmoid(-self.leader_close_head(leader_dist))

        return ego_lane_onehot, unsafe_gap_prob, leader_close_prob

    def forward(self, obs):
        q_values = self.q_net(self.extract_features(obs, self.features_extractor))  # (batch, 8)

        obs_f = obs.float()
        ego_lane, unsafe_gap, leader_close = self._symbolic_facts(obs_f)
        result = self.planner(ego_lane=ego_lane, unsafe_gap=unsafe_gap, leader_close=leader_close)
        next_action_scores = result["next_action"] if isinstance(result, dict) else result
        next_action_probs = torch.softmax(next_action_scores, dim=1)

        return q_values + torch.log(next_action_probs + 1e-6)


class MergePlannerDQNPolicy(DQNPolicy):
    def make_q_net(self):
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        return MergePlannerQNetwork(**net_args).to(self.device)
