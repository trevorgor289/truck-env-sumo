"""
Neurosymbolic PPO policy: wraps SB3's ActorCriticPolicy so the final action
distribution is a differentiable blend of the neural actor's logits and a
Scallop program's `next_action` relation, reasoning over lane-safety facts
derived from the observation. Modeled on the Scallop paper's PacMan-Maze
architecture (Section 4.3): a small neural head turns continuous state into
probabilistic relational facts, `merge_planner.scl` reasons over them, and
gradients flow back through Scallop's differentiable provenance
(`diff-top-k-proofs`) into that head, end to end with the rest of PPO's loss.

NOTE: this is a first draft written against the scallopy API as documented in
the paper's PacMan-Maze example; exact parameter names should be verified
against the installed scallopy package and its bundled examples before
trusting this to run unmodified.
"""

import os

import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from torch.distributions import Categorical

import scallopy

SCL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "merge_planner.scl")

# Action indices, matching highway_env.py's semantic action constants
NB_LANES = 3
ACTION_SYMBOLS = [
    "SHORT_GAP", "MEDIUM_GAP", "LONG_GAP", "INC_SPEED",
    "DEC_SPEED", "LANE_LEFT", "LANE_RIGHT", "MAINTAIN",
]

# Observation layout (see highway_env.py _sensor_model): index 2 = ego lane,
# index 5 = distance to leading vehicle. Per-surrounding-vehicle block starts
# at NB_EGO_STATES=6, stride NB_STATES_PER_VEHICLE=7, column 0 = relative
# longitudinal distance, column 4 = lane index.
NB_EGO_STATES = 6
NB_STATES_PER_VEHICLE = 7
EGO_LANE_IDX = 2
LEADER_DIST_IDX = 5


class MergePlannerPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Small learned heads mapping continuous gap/distance features to
        # probabilistic symbolic facts -- the differentiable seam between
        # the neural and symbolic components.
        self.unsafe_gap_head = nn.Linear(1, 1)   # per-lane min gap -> P(unsafe_gap(lane))
        self.leader_close_head = nn.Linear(1, 1)  # leader distance -> P(leader_close)

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
            dispatch="serial",  # scallopy's default "parallel" dispatch panics under Python 3.12's GIL model
        )

    def _per_lane_min_gap(self, obs):
        """For each lane, the smallest |relative longitudinal distance| among
        surrounding vehicles currently in that lane (large default if none)."""
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

        min_gap = self._per_lane_min_gap(obs)  # (batch, NB_LANES)
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

        combined_logits = neural_logits + torch.log(next_action_probs + 1e-6)
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
