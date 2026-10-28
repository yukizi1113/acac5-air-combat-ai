"""Submission entry point (M-mode: one Agent instance controls the 4 fighters).

args.json keys
  "policy"   : "rule" (rule-based commander) | "nn" (learned high-level policy, weights in policy.pt)
  "overrides": dict of modelConfig overrides (tactical parameters)
  "greedy"   : true -> argmax actions for the NN policy (default true at evaluation)
"""
import os
import json
from ASRCAISim1.policy import StandalonePolicy  # type: ignore

_HERE = os.path.dirname(__file__)
# Training-time hook: a trainer can register policy objects here, selected by args["policy_slot"].
POLICY_REGISTRY = {}


def getUserAgentClass(args={}):
    from .agent import TacAgent
    return TacAgent


def getUserAgentModelConfig(args={}):
    cfg = json.load(open(os.path.join(_HERE, "config.json"), "r"))
    args = args or {}
    for k, v in args.get("overrides", {}).items():
        cfg[k] = v
    if args.get("policy", "rule") == "nn" or args.get("policy_slot") is not None:
        cfg["rl_mode"] = True
    return cfg


def isUserAgentSingleAsset(args={}):
    return False


class RulePolicy(StandalonePolicy):
    """The rule-based commander lives inside the Agent; returning None selects it."""
    def step(self, observation, reward, done, info, agentFullName, observation_space, action_space):
        return None


class NNPolicy(StandalonePolicy):
    """Learned option-selection policy. Evaluated only on decision ticks; the last action is held."""
    def __init__(self, path, greedy=True):
        super().__init__()
        import torch
        from .nn_model import EntityNet
        torch.set_num_threads(1)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.net = EntityNet(**ck.get("arch", {}))
        self.net.load_state_dict(ck["model"])
        self.net.eval()
        self.greedy = greedy
        self.last = {}

    def reset(self):
        self.last = {}

    def step(self, observation, reward, done, info, agentFullName, observation_space, action_space):
        import numpy as np
        import torch
        from .nn_model import obs_to_tensors
        if observation is None or not isinstance(observation, dict):
            return None
        if observation["decide"][0] < 0.5 and agentFullName in self.last:
            return self.last[agentFullName]
        with torch.no_grad():
            lo, lf, _ = self.net(*obs_to_tensors([observation]))
            if self.greedy:
                ao = lo[0].argmax(-1).numpy(); af = lf[0].argmax(-1).numpy()
            else:
                ao = torch.distributions.Categorical(logits=lo[0]).sample().numpy()
                af = torch.distributions.Categorical(logits=lf[0]).sample().numpy()
        a = np.concatenate([ao, af]).astype(np.int64)
        self.last[agentFullName] = a
        return a


def getUserPolicy(args={}):
    args = args or {}
    slot = args.get("policy_slot")
    if slot is not None and slot in POLICY_REGISTRY:
        return POLICY_REGISTRY[slot]
    if args.get("policy", "rule") == "nn":
        path = os.path.join(_HERE, args.get("weights", "policy.pt"))
        if os.path.exists(path):
            return NNPolicy(path, greedy=args.get("greedy", True))
    return RulePolicy()
