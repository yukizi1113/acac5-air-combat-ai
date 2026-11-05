"""Shared actor-critic for the decentralised fighters (set / entity transformer, permutation-invariant over entities).

tokens = [self, friends x4, enemies x9, missiles x8, mws x4]  -> type-specific linear embedding (+ masks)
       -> L pre-norm transformer blocks (explicit multi-head attention, so the numpy mirror is exact)
       -> z = [h_self, masked-mean(h)] -> heads: option (6), altitude band (3), fire mode (3), value (1)
The same weights run for every fighter on its own ego observation (decentralised execution, parameter sharing).
Inference in the match uses net2_np.py (numpy) with weights exported by export_npz(); torch is only for training.
"""
import numpy as np
import torch
import torch.nn as nn
from .rl2 import S_DIM, F_DIM, E_DIM, M_DIM, W_DIM, N_OPT, N_ALT, N_FIRE

ARCH = dict(d=64, heads=4, layers=2, ff=128)


class Block(nn.Module):
    def __init__(self, d, heads, ff):
        super().__init__()
        self.h = heads; self.dk = d // heads
        self.ln1 = nn.LayerNorm(d); self.ln2 = nn.LayerNorm(d)
        self.q = nn.Linear(d, d); self.k = nn.Linear(d, d); self.v = nn.Linear(d, d); self.o = nn.Linear(d, d)
        self.f1 = nn.Linear(d, ff); self.f2 = nn.Linear(ff, d)

    def forward(self, x, mask):                       # x [B,T,d], mask [B,T] (1 valid)
        B, T, d = x.shape
        h = self.ln1(x)
        q = self.q(h).view(B, T, self.h, self.dk).transpose(1, 2)
        k = self.k(h).view(B, T, self.h, self.dk).transpose(1, 2)
        v = self.v(h).view(B, T, self.h, self.dk).transpose(1, 2)
        s = q @ k.transpose(-1, -2) / (self.dk ** 0.5)
        s = s + (1.0 - mask)[:, None, None, :] * -1e9
        a = torch.softmax(s, -1) @ v
        x = x + self.o(a.transpose(1, 2).reshape(B, T, d))
        x = x + self.f2(torch.relu(self.f1(self.ln2(x))))
        return x


def _head(i, d, o):
    return nn.Sequential(nn.Linear(i, d), nn.ReLU(), nn.Linear(d, o))


class Net2(nn.Module):
    def __init__(self, d=64, heads=4, layers=2, ff=128):
        super().__init__()
        self.es = nn.Linear(S_DIM, d); self.ef = nn.Linear(F_DIM, d); self.ee = nn.Linear(E_DIM, d)
        self.em = nn.Linear(M_DIM, d); self.ew = nn.Linear(W_DIM, d)
        self.blocks = nn.ModuleList([Block(d, heads, ff) for _ in range(layers)])
        self.lnf = nn.LayerNorm(d)
        self.h_opt = _head(2 * d, d, N_OPT); self.h_alt = _head(2 * d, d, N_ALT); self.h_fire = _head(2 * d, d, N_FIRE)
        self.h_v = _head(2 * d, d, 1)

    def forward(self, s, fr, frm, en, enm, ms, msm, mw, mwm):
        x = torch.cat([self.es(s)[:, None], self.ef(fr), self.ee(en), self.em(ms), self.ew(mw)], 1)
        mask = torch.cat([torch.ones_like(s[:, :1]), frm, enm, msm, mwm], 1)
        for b in self.blocks:
            x = b(x, mask)
        x = self.lnf(x)
        pooled = (x * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
        z = torch.cat([x[:, 0], pooled], -1)
        return self.h_opt(z), self.h_alt(z), self.h_fire(z), self.h_v(z)[:, 0]


KEYS = ('self', 'fr', 'frm', 'en', 'enm', 'ms', 'msm', 'mw', 'mwm')


def batch_tensors(samples):
    """samples: list of per-fighter dicts (key -> np array without the fighter axis)."""
    return [torch.as_tensor(np.stack([x[k] for x in samples]), dtype=torch.float32) for k in KEYS]


def export_npz(net, path, extra=None):
    d = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in net.state_dict().items()}
    if extra:
        for k, v in extra.items():
            d['__' + k] = np.asarray(v)
    np.savez_compressed(path, **d)


def load_from_npz(net, path):
    z = np.load(path)
    sd = {k: torch.as_tensor(z[k]) for k in z.files if not k.startswith('__')}
    net.load_state_dict(sd)
    return net
