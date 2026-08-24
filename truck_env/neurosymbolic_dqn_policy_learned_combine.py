"""
Neurosymbolic DQN policy, LEARNED-COMBINE variant.

Copy of neurosymbolic_dqn_policy.py (MergePlannerQNetwork/MergePlannerDQNPolicy)
with one architectural change: instead of `neural_q + log(scallop_prob + eps)`,
the raw Q-values and Scallop's next_action scores are concatenated and passed
through a small trainable combiner MLP, applied identically inside both
q_net and q_net_target since both are built from the same make_q_net()
factory.

Tradeoff vs. the original: the combined Q-values are no longer guaranteed to
crush a vetoed action toward -inf -- that's now something the combiner has to
learn to do (or not), rather than a fixed mathematical floor. Worth testing
here specifically because the original's hard-ish veto may be interacting
badly with DQN's own instability (see dqn_neurosymbolic_seed0_20260819_043657's
mid-training dip below the original pure-DQN run's peak).
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

NB_EGO_STATES = 6
NB_STATES_PER_VEHICLE = 7
EGO_LANE_IDX = 2
LEADER_DIST_IDX = 5


class LearnedCombineQNetwork(QNetwork):
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
            output_mappings={"next_action": list(range(NB_ACTIONS))},
            dispatch="serial",
        )

        # Learned combiner: [q_values (8), scallop_probs (8)] -> combined_q (8)
        self.combiner = nn.Sequential(
            nn.Linear(NB_ACTIONS * 2, 32),
            nn.ReLU(),
            nn.Linear(32, NB_ACTIONS),
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

        combiner_input = torch.cat([q_values, next_action_probs], dim=1)
        return self.combiner(combiner_input)


class LearnedCombineDQNPolicy(DQNPolicy):
    def make_q_net(self):
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        return LearnedCombineQNetwork(**net_args).to(self.device)
