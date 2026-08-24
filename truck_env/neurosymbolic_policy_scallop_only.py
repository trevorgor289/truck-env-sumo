"""
Neurosymbolic PPO policy, SCALLOP-ONLY variant.

Unlike neurosymbolic_policy.py (parallel branches + fixed formula) and
neurosymbolic_policy_learned_combine.py (parallel branches + learned
combiner), this is a strict pipeline for the actor: the MLP's only job is to
produce the probabilistic facts Scallop reasons over; Scallop's next_action
output directly becomes the action logits, with nothing recombining it
afterward -- no separate "raw actor logits" pathway exists.

One necessary carve-out: PPO still needs a value function (critic) for
advantage estimation, and Scallop has no notion of expected return, only
action preference. So the critic (value_net, fed by the standard
mlp_extractor) stays a normal learned component -- only the actor becomes
MLP -> Scallop -> action.
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
NB_FACT_OUTPUTS = NB_LANES + 1  # 3x unsafe_gap + 1x leader_close

EGO_LANE_IDX = 2


class ScallopOnlyPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # The actor's only learned component: raw features -> fact logits for Scallop.
        # No separate actor-logit head exists in this variant.
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

    def _action_distribution(self, obs, features):
        ego_lane, unsafe_gap, leader_close = self._symbolic_facts(obs.float(), features)
        result = self.planner(ego_lane=ego_lane, unsafe_gap=unsafe_gap, leader_close=leader_close)
        next_action_scores = result["next_action"] if isinstance(result, dict) else result
        return Categorical(logits=next_action_scores)  # used directly as logits, no further transform

    def forward(self, obs, deterministic=False):
        features = self.extract_features(obs)
        _, latent_vf = self.mlp_extractor(features)  # critic still needed; actor latent unused
        distribution = self._action_distribution(obs, features)
        actions = distribution.mode if deterministic else distribution.sample()
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return actions, values, log_prob

    def evaluate_actions(self, obs, actions):
        features = self.extract_features(obs)
        _, latent_vf = self.mlp_extractor(features)
        distribution = self._action_distribution(obs, features)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return values, log_prob, distribution.entropy()

    def get_distribution(self, obs):
        features = self.extract_features(obs)
        return self._action_distribution(obs, features)
