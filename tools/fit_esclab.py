"""Fit a simple escort-shot Pk model on the escort-strike lab and print Pk(range) curves per altitude/aspect.

  python tools/fit_esclab.py results/esclab1.jsonl data/shots_esclab1.jsonl
"""
import sys, json, math
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


def load(res, shots):
    out = {}
    for l in open(res):
        r = json.loads(l)
        for m in r.get('godlog') or []:
            out[(r['match_id'], m['name'])] = (m['hit'], r)
    X, y, S = [], [], []
    for l in open(shots):
        s = json.loads(l)
        k = (s['match'], s['msl'])
        if k in out:
            X.append(s); y.append(int(out[k][0])); S.append(out[k][1]['meta']['S'])
    return X, np.array(y), S


def feats(s, S):
    return [s['rn'], s['rn'] ** 2, s['d'] / 1e5, math.cos(s['aspect']), s['alt_s'] / 1e4, s['V_s'] / 500,
            s['thrust'] / 30018.0, 1.0 if S == 'crank' else 0.0, 1.0 if S == 'drag' else 0.0]


def main(res, shots):
    X, y, S = load(res, shots)
    print('n', len(y), 'hit rate', y.mean())
    rn = np.array([s['rn'] for s in X])
    for lo, hi in [(-1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.1), (1.1, 3)]:
        k = (rn >= lo) & (rn < hi)
        if k.sum():
            print(f'  rn {lo:4.1f}-{hi:3.1f}: n={k.sum():3d} hit={y[k].mean():.2f}   d={np.median([X[i]["d"] for i in np.where(k)[0]])/1000:.0f}km')
    Z = np.array([feats(s, q) for s, q in zip(X, S)])
    clf = LogisticRegression(C=3.0, max_iter=2000)
    print('cv logloss', -cross_val_score(clf, Z, y, cv=5, scoring='neg_log_loss').mean(),
          'cv auc', cross_val_score(clf, Z, y, cv=5, scoring='roc_auc').mean())
    clf.fit(Z, y)
    print('coef', dict(zip(['rn', 'rn2', 'd', 'cosasp', 'alt', 'V', 'thr', 'crank', 'drag'], np.round(clf.coef_[0], 2))))
    # crank support, empirical rh/rt from samples: report Pk vs rn for typical case
    for rnv in [0.2, 0.4, 0.6, 0.8, 1.0]:
        row = []
        for alt in [11000, 13000, 15000]:
            z = [rnv, rnv ** 2, 0.8, 0.0, alt / 1e4, 1.05, 1.0, 1.0, 0.0]
            row.append(f'{clf.predict_proba([z])[0, 1]:.2f}')
        print(f'  rn={rnv}: Pk(alt 11/13/15 km, crank) =', ' '.join(row))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
