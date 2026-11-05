"""Numpy forward pass of net2.Net2 (no torch needed in the match process). Weights: npz from net2.export_npz."""
import numpy as np


def _ln(x, w, b, eps=1e-5):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * w + b


def _lin(W, p, x):
    return x @ W[p + '.weight'].T + W[p + '.bias']


class Net2NP:
    def __init__(self, path):
        z = np.load(path)
        self.W = {k: z[k].astype(np.float64) for k in z.files if not k.startswith('__')}
        self.meta = {k[2:]: z[k] for k in z.files if k.startswith('__')}
        self.layers = sum(1 for k in self.W if k.startswith('blocks.') and k.endswith('.ln1.weight'))
        self.d = self.W['es.weight'].shape[0]
        self.heads = int(self.meta.get('heads', 4))

    def _head(self, p, z):
        W = self.W
        return _lin(W, p + '.2', np.maximum(0.0, _lin(W, p + '.0', z)))

    def forward(self, s, fr, frm, en, enm, ms, msm, mw, mwm):
        """Batched over the leading axis. Returns (opt_logits, alt_logits, fire_logits, value)."""
        W = self.W
        x = np.concatenate([_lin(W, 'es', s)[:, None], _lin(W, 'ef', fr), _lin(W, 'ee', en), _lin(W, 'em', ms),
                            _lin(W, 'ew', mw)], 1)
        mask = np.concatenate([np.ones_like(s[:, :1]), frm, enm, msm, mwm], 1)
        B, T, d = x.shape
        H = self.heads; dk = d // H
        for i in range(self.layers):
            p = f'blocks.{i}.'
            h = _ln(x, W[p + 'ln1.weight'], W[p + 'ln1.bias'])
            q = _lin(W, p + 'q', h).reshape(B, T, H, dk).transpose(0, 2, 1, 3)
            k = _lin(W, p + 'k', h).reshape(B, T, H, dk).transpose(0, 2, 1, 3)
            v = _lin(W, p + 'v', h).reshape(B, T, H, dk).transpose(0, 2, 1, 3)
            sc = q @ k.transpose(0, 1, 3, 2) / np.sqrt(dk) + (1.0 - mask)[:, None, None, :] * -1e9
            sc = sc - sc.max(-1, keepdims=True)
            a = np.exp(sc); a = a / a.sum(-1, keepdims=True)
            o = (a @ v).transpose(0, 2, 1, 3).reshape(B, T, d)
            x = x + _lin(W, p + 'o', o)
            h2 = _ln(x, W[p + 'ln2.weight'], W[p + 'ln2.bias'])
            x = x + _lin(W, p + 'f2', np.maximum(0.0, _lin(W, p + 'f1', h2)))
        x = _ln(x, W['lnf.weight'], W['lnf.bias'])
        pooled = (x * mask[..., None]).sum(1) / mask.sum(1, keepdims=True)
        z = np.concatenate([x[:, 0], pooled], -1)
        return self._head('h_opt', z), self._head('h_alt', z), self._head('h_fire', z), self._head('h_v', z)[:, 0]


def logsoftmax(l):
    l = l - l.max(-1, keepdims=True)
    return l - np.log(np.exp(l).sum(-1, keepdims=True))
