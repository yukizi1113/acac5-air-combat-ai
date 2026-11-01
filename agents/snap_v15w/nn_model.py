"""Entity-attention actor-critic for option selection.

  tokens = [global, friend x4, enemy x8, escort x2] -> type-specific linear embeddings
  -> L x TransformerEncoder (pre-norm, masked)  -> heads
     option head (per friend token): N_OPT logits
     fire-mode head (per friend token): N_FIRE logits  (hold / normal / aggressive)
     value head: masked mean-pool + global token -> V(s)
Permutation-equivariant over enemies and friends, so it handles variable numbers of tracks and
losses naturally; parameter count ~60k so CPU inference is < 2 ms.
"""
import torch
import torch.nn as nn
from .rl_obs import G_DIM, F_DIM, E_DIM, S_DIM, MAX_F, MAX_E, N_OPT, N_FIRE


class EntityNet(nn.Module):
    def __init__(self, d=64, layers=2, heads=4, ff=128):
        super().__init__()
        self.eg = nn.Linear(G_DIM, d)
        self.ef = nn.Linear(F_DIM, d)
        self.ee = nn.Linear(E_DIM, d)
        self.es = nn.Linear(S_DIM, d)
        self.type_emb = nn.Parameter(torch.zeros(4, d))
        layer = nn.TransformerEncoderLayer(d, heads, ff, dropout=0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.opt_head = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, N_OPT))
        self.fire_head = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, N_FIRE))
        self.v_head = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, 1))

    def forward(self, g, f, e, s, fmask, emask):
        B = g.shape[0]
        tg = self.eg(g).unsqueeze(1) + self.type_emb[0]
        tf = self.ef(f) + self.type_emb[1]
        te = self.ee(e) + self.type_emb[2]
        ts = self.es(s) + self.type_emb[3]
        x = torch.cat([tg, tf, te, ts], 1)
        valid = torch.cat([torch.ones(B, 1, device=g.device), torch.ones_like(fmask), emask,
                           torch.ones(B, 2, device=g.device)], 1) > 0.5
        h = self.norm(self.enc(x, src_key_padding_mask=~valid))
        hg = h[:, 0]
        hf = h[:, 1:1 + MAX_F]
        w = valid.float().unsqueeze(-1)
        pooled = (h * w).sum(1) / w.sum(1).clamp(min=1.0)
        ctx = pooled.unsqueeze(1).expand(-1, MAX_F, -1)
        z = torch.cat([hf, ctx], -1)
        return self.opt_head(z), self.fire_head(z), self.v_head(torch.cat([hg, pooled], -1)).squeeze(-1)


def obs_to_tensors(obs_list, device='cpu'):
    import numpy as np
    keys = ['g', 'f', 'e', 's', 'fmask', 'emask']
    return [torch.as_tensor(np.stack([o[k] for o in obs_list]), dtype=torch.float32, device=device) for k in keys]
