"""Runtime inference of the learned hit-probability model (numpy only, LightGBM optional)."""
import json
import numpy as np


def featurize(x):
    return np.array([x['rn'], x['d'] / 1000, x['rh'] / 1000, x['rt'] / 1000, x['aspect'], x['ata'],
                     x['alt_s'] / 1000, x['alt_t'] / 1000, x['V_s'], x['V_t'], x['closing'],
                     x['thrust'] / 30017.9989, x['kind'], x['dz'] / 1000, x.get('rhe', 0.0) / 1000, 0.0], dtype=np.float64)


class PkModel:
    def __init__(self, path):
        d = json.load(open(path))
        self.kind = d['type']
        self.nfeat = len(d.get('feats', [])) or 15
        if self.kind == 'lgbm':
            import lightgbm as lgb
            self.booster = lgb.Booster(model_str=d['model'])
        else:
            self.mean = np.array(d['mean']); self.scale = np.array(d['scale'])
            if self.kind == 'logreg':
                self.coef = np.array(d['coef']); self.b0 = d['intercept']
            else:
                self.W = [np.array(w) for w in d['W']]; self.b = [np.array(b) for b in d['b']]

    def predict(self, x):
        z = featurize(x)[:self.nfeat]
        if self.kind == 'lgbm':
            return float(self.booster.predict(z[None, :])[0])
        z = (z - self.mean) / self.scale
        if self.kind == 'logreg':
            s = z @ self.coef + self.b0
        else:
            h = z
            for i, (W, b) in enumerate(zip(self.W, self.b)):
                h = h @ W + b
                if i < len(self.W) - 1:
                    h = np.maximum(h, 0.0)
            s = h[0]
        return float(1.0 / (1.0 + np.exp(-s)))
