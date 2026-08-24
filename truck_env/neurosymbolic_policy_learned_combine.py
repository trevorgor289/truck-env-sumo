"""
Neurosymbolic PPO policy, LEARNED-COMBINE variant.

Copy of neurosymbolic_policy.py (MergePlannerPolicy) with one architectural
change: instead of a fixed `neural_logits + log(scallop_prob + eps)` formula,
the neural actor's raw logits and Scallop's next_action scores are
concatenated and passed through a small trainable "combiner" MLP that learns
how to merge them, rather than a hand-picked rule.

Tradeoff vs. the original: this gains flexibility (the network can learn a
context-dependent way to weigh the two signals) but loses the original's
structural guarantee that a Scallop-vetoed action's score is driven toward
-inf no matter what -- that veto is now emergent, not mathematically forced,
since the combiner's weights are free to route around it if that's what
gradient descent finds easier.

merge_planner.scl itself is unchanged and still fixed/non-learned; only the
combination step differs from the original.
"""

import os

import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from torch.distributions import Categorical

import scallopy

SCL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "merge_planner.scl")

NB_LANES = 3
ACTION_SYMBOLS = [
    "SHORT_GAP", "MEDIUM_GAP", "LONG_GAP", "INC_SPEED",
    "DEC_SPEED", "LANE_LEFT", "LANE_RIGHT", "MAINTAIN",
]
NB_ACTIONS = len(ACTION_SYMBOLS)

# Observation layout (see highway_env.py _sensor_model): index 2 = ego lane,
# index 5 = distance to leading vehicle. Per-surrounding-vehicle block starts
# at NB_EGO_STATES=6, stride NB_STATES_PER_VEHICLE=7, column 0 = relative
# longitudinal distance, column 4 = lane index.
NB_EGO_STATES = 6
NB_STATES_PER_VEHICLE = 7
EGO_LANE_IDX = 2
LEADER_DIST_IDX = 5


class LearnedCombinePolicy(ActorCriticPolicy):
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
            dispatch="serial",  # scallopy's default "parallel" dispatch panics under Python 3.12's GIL model
        )

        # Learned combiner: [neural_logits (8), scallop_probs (8)] -> combined_logits (8)
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

    def _combined_distribution(self, obs, latent_pi):
        neural_logits = self.action_net(latent_pi)  # (batch, 8)

        ego_lane, unsafe_gap, leader_close = self._symbolic_facts(obs.float())
        result = self.planner(ego_lane=ego_lane, unsafe_gap=unsafe_gap, leader_close=leader_close)
        next_action_scores = result["next_action"] if isinstance(result, dict) else result
        next_action_probs = torch.softmax(next_action_scores, dim=1)  # scallop's output isn't pre-normalized

        combiner_input = torch.cat([neural_logits, next_action_probs], dim=1)
        combined_logits = self.combiner(combiner_input)
        return Categorical(logits=combined_logits)

    def forward(self, obs, deterministic=False):
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        distribution = self._combined_distribution(obs, latent_pi)
        actions = distribution.mode if deterministic else distribution.sample()
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return actions, values, log_prob

    def evaluate_actions(self, obs, actions):
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        distribution = self._combined_distribution(obs, latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return values, log_prob, distribution.entropy()

    def get_distribution(self, obs):
        features = self.extract_features(obs)
        latent_pi, _ = self.mlp_extractor(features)
        return self._combined_distribution(obs, latent_pi)
