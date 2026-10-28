import os, json
from ASRCAISim1.policy import StandalonePolicy
def getUserAgentClass(args={}):
    from .agent import LabAgent
    return LabAgent
def getUserAgentModelConfig(args={}):
    cfg = json.load(open(os.path.join(os.path.dirname(__file__), '..', 'Tac', 'config.json')))
    cfg.update((args or {}).get('overrides', {}))
    cfg['use_pk_model'] = False
    return cfg
def isUserAgentSingleAsset(args={}):
    return False
class P(StandalonePolicy):
    def step(self, *a): return None
def getUserPolicy(args={}):
    return P()
