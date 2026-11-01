"""Fit the strategy-selection policy (option-level decision at arm_t) from counterfactual rollouts.

For every (opponent, seed, colour) the match is identical up to arm_t for all arms (deterministic simulator),
so results/arm{a}_{opp}.json give the outcome of each arm from the *same* situation.  Features at arm_t come from
results/armfeat.jsonl (own sensors / datalink only).

Policy: per-arm ridge regression of the match score on standardised features; act = argmax.
Evaluated with leave-one-opponent-out CV (generalisation to an opponent type never seen in training).
  python tools/fit_arms.py [--out agents/TacB/arm_policy.json]
"""
import json, glob, os, sys, argparse
import numpy as np

FEAT = ['n_in', 'n_trk', 'x_mean', 'x_min', 'alt_med', 'V_med', 'd_esc_min', 'kills', 'lost', 'max_in']


def load(arms, opps):
    feat = {}
    for l in open('results/armfeat.jsonl'):
        r = json.loads(l)
        tag, seed, color = r['match'].split('.json_')[0], *r['match'].split('.json_')[1].split('_')[:2]
        opp = tag.split('_', 1)[1]
        feat[(opp, int(seed), color)] = [r[k] for k in FEAT]
    Y = {}
    for a in arms:
        for o in opps:
            p = f'results/arm{a}_{o}.json'
            if not os.path.exists(p):
                continue
            for l in open(p):
                r = json.loads(l)
                if 'score' in r:
                    Y.setdefault((o, r['seed'], r['color']), {})[a] = r['score']
    keys = [k for k in Y if len(Y[k]) == len(arms) and k in feat]
    X = np.array([feat[k] for k in keys], float)
    S = np.array([[Y[k][a] for a in arms] for k in keys], float)
    return keys, X, S


def fit(X, S, lam=3.0):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    A = np.hstack([Z, np.ones((len(Z), 1))])
    W = np.linalg.solve(A.T @ A + lam * np.diag([1] * Z.shape[1] + [0]), A.T @ S)   # (F+1, n_arms)
    return mu, sd, W


def act(model, X):
    mu, sd, W = model
    Z = (X - mu) / sd
    return np.argmax(np.hstack([Z, np.ones((len(Z), 1))]) @ W, axis=1)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--arms', default='0,1,2,3,4')
    ap.add_argument('--opps', default='p2,sn,scr,v7,mir,low')
    ap.add_argument('--lam', type=float, default=3.0)
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    arms = [int(x) for x in a.arms.split(',')]; opps = a.opps.split(',')
    keys, X, S = load(arms, opps)
    print('matches with all arms:', len(keys))
    opp_of = np.array([k[0] for k in keys])
    print('mean score per arm (rows = opponent):')
    for o in opps:
        m = opp_of == o
        if m.any():
            print(f'  {o:>4} n={m.sum():2d} ' + ' '.join(f'a{arms[j]}={S[m, j].mean():.3f}' for j in range(len(arms))))
    print('  all  ' + ' '.join(f'a{arms[j]}={S[:, j].mean():.3f}' for j in range(len(arms))),
          f'| per-match oracle={S.max(1).mean():.3f}')
    # leave-one-opponent-out
    got = np.zeros(len(keys))
    for o in opps:
        te = opp_of == o
        if not te.any() or te.all():
            continue
        mdl = fit(X[~te], S[~te], a.lam)
        got[te] = S[te, act(mdl, X[te])]
    print('LOOO-CV policy score:', round(got.mean(), 3), ' by opponent:',
          {o: round(got[opp_of == o].mean(), 3) for o in opps if (opp_of == o).any()})
    mdl = fit(X, S, a.lam)
    ch = act(mdl, X)
    print('in-sample policy score:', round(S[np.arange(len(S)), ch].mean(), 3), ' arm usage:', np.bincount(ch, minlength=len(arms)).tolist())
    if a.out:
        mu, sd, W = mdl
        pol = {'feat': FEAT, 'mu': mu.tolist(), 'sd': sd.tolist(), 'W': W[:-1].T.tolist(), 'b': W[-1].tolist(), 'arms': arms}
        json.dump(pol, open(a.out, 'w'))
        print('wrote', a.out)
