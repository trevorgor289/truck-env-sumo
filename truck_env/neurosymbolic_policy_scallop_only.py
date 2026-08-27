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

HISTORY, for context on the two design changes below:

1. Originally, merge_planner.scl (the file every other policy here still
   uses) only gave 3 of the 8 actions (LANE_LEFT, LANE_RIGHT, INC_SPEED) any
   safety precondition; the other 5 were `rel next_action(X) = ego_lane(_)`,
   unconditionally true, tag exactly 1.0 with zero gradient w.r.t. any
   trainable parameter -- the model could never learn a preference among
   those 5, full stop. Confirmed against a live checkpoint at the time: raw
   next_action tags were exactly [1.0, 1.0, 1.0, 0.26, 1.0, 0.0, 0.66, 1.0].

2. This policy now points at merge_planner_weighted.scl instead (a separate
   file -- the shared merge_planner.scl and every policy still using it are
   untouched). Every rule there, including all 8 next_action(...) rules, is
   ANDed with its own learned, randomly-initialized confidence weight
   (self.rule_weights below). That incidentally fixes point 1: each of the
   previously-dead 5 actions now has its own trainable w_next_* weight
   multiplying its tag, so there's a real gradient path for all 8 actions,
   not just 3. lane_exists is still left a hard/certain fact (a genuinely
   certain structural property of the road), and the existing safety logic
   (safe_lane, can_change_left/right, can_increase_speed) is unchanged --
   the learned weights multiply an additional confidence on top of that
   logic, they don't replace it.
"""

import os

import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from torch.distributions import Categorical

import scallopy

SCL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "merge_planner_weighted.scl")

NB_LANES = 3
ACTION_SYMBOLS = [
    "SHORT_GAP", "MEDIUM_GAP", "LONG_GAP", "INC_SPEED",
    "DEC_SPEED", "LANE_LEFT", "LANE_RIGHT", "MAINTAIN",
]
NB_ACTIONS = len(ACTION_SYMBOLS)
NB_FACT_OUTPUTS = NB_LANES + 1  # 3x unsafe_gap + 1x leader_close

EGO_LANE_IDX = 2

# One learned confidence weight per rule in merge_planner_weighted.scl, in the
# same order as that file's w_* type declarations. Each is ANDed into its
# rule's body as an auxiliary input fact -- see merge_planner_weighted.scl's
# header comment for why a *learned* rule weight has to be supplied this way
# rather than as a `rel 0.9 :: head = body` compile-time tag.
RULE_WEIGHT_NAMES = [
    "w_safe_lane", "w_can_change_left", "w_can_change_right", "w_can_increase_speed",
    "w_next_short_gap", "w_next_medium_gap", "w_next_long_gap", "w_next_inc_speed",
    "w_next_dec_speed", "w_next_lane_left", "w_next_lane_right", "w_next_maintain",
]
NB_RULE_WEIGHTS = len(RULE_WEIGHT_NAMES)


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
                **{name: [(0,)] for name in RULE_WEIGHT_NAMES},
            },
            output_mappings={"next_action": list(range(NB_ACTIONS))},
            dispatch="serial",
        )

        # One learned weight per rule, randomly initialized (not a fixed
        # constant): torch.randn gives each an independent draw from a
        # standard normal, so sigmoid(weight) starts spread out around 0.5
        # rather than all identical. Trained like any other parameter here --
        # picked up by the optimizer rebuild below since it's declared before
        # that runs.
        self.rule_weights = nn.Parameter(torch.randn(NB_RULE_WEIGHTS))

        # Scallop's next_action tags live in [0,1] -- too narrow a range to use
        # directly as Categorical logits (softmax([1,1,1,.26,1,0,.66,1]) is
        # nearly uniform: 15/15/15/7/15/6/11/15%, so even a mathematically
        # impossible action at tag=0 still draws ~6% probability). A learned,
        # positive temperature lets training sharpen this scale.
        #
        # CAUTION, found by direct measurement: scaling raw tags by
        # `tag * temperature` (uncentered) makes an UNTRAINED network already
        # look "successful" for the wrong reason. unsafe_gap/leader_close come
        # from fresh sigmoid(Linear) heads, which sit near 0.5 (genuine
        # uncertainty) before any training. At temperature=10, tag=0.5 becomes
        # logit=5 vs the always-safe actions' logit=10 -- an e^5≈148x
        # suppression from pure random init, not from anything learned. That
        # made the untrained policy default to never attempting LANE_LEFT/
        # LANE_RIGHT/INC_SPEED at all, and this environment turns out to be
        # solvable that way (verified: an untrained model scored 99.3% success
        # over 150 episodes, statistically identical to the "trained" run's
        # 98-99% -- i.e. the earlier number reflected architecture, not
        # learning). Centering at 0.5 fixes this: uncertainty (tag=0.5) now
        # maps to logit=0 (neutral, comparable to a lower but non-negligible
        # probability against the always-safe actions' fixed +temperature/2),
        # instead of being crushed as if it were confidently unsafe. Only
        # actual training-driven movement away from 0.5 sharpens the
        # preference, in either direction.
        self._log_temperature = nn.Parameter(torch.log(torch.tensor(6.0)))

        # BUG FIX: ActorCriticPolicy.__init__() ends by calling self._build(),
        # which constructs self.optimizer = Adam(self.parameters(), ...) --
        # that already ran inside super().__init__() above, BEFORE fact_mlp,
        # planner, and _log_temperature existed. So none of them were ever in
        # the optimizer: verified directly (parameter values were bit-for-bit
        # identical before/after real PPO gradient updates that did change
        # value_net). Rebuilding the optimizer here, now that every parameter
        # actually exists, is the fix -- same optimizer_class/kwargs/lr SB3
        # already chose, just constructed at the right time.
        current_lr = self.optimizer.param_groups[0]["lr"]
        self.optimizer = self.optimizer_class(self.parameters(), lr=current_lr, **self.optimizer_kwargs)

    def _symbolic_facts(self, obs, features):
        ego_lane = obs[:, EGO_LANE_IDX].round().long().clamp(0, NB_LANES - 1)
        ego_lane_onehot = torch.nn.functional.one_hot(ego_lane, NB_LANES).float()

        fact_logits = self.fact_mlp(features)
        unsafe_gap_prob = torch.sigmoid(fact_logits[:, :NB_LANES])
        leader_close_prob = torch.sigmoid(fact_logits[:, NB_LANES:NB_LANES + 1])

        return ego_lane_onehot, unsafe_gap_prob, leader_close_prob

    def _action_distribution(self, obs, features):
        ego_lane, unsafe_gap, leader_close = self._symbolic_facts(obs.float(), features)
        batch = obs.shape[0]
        rule_weight_probs = torch.sigmoid(self.rule_weights)
        rule_weight_kwargs = {
            name: rule_weight_probs[i].expand(batch, 1)
            for i, name in enumerate(RULE_WEIGHT_NAMES)
        }
        result = self.planner(
            ego_lane=ego_lane, unsafe_gap=unsafe_gap, leader_close=leader_close,
            **rule_weight_kwargs,
        )
        next_action_scores = result["next_action"] if isinstance(result, dict) else result
        scaled_logits = (next_action_scores - 0.5) * torch.exp(self._log_temperature)
        return Categorical(logits=scaled_logits)

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
