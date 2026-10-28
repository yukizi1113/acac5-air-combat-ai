"""Paired comparison on identical seeds/colors: python tools/paired.py BASE CAND [opp...]"""
import sys, json, os
import numpy as np
base, cand = sys.argv[1], sys.argv[2]
opps = sys.argv[3:] or ['v7', 'sn', 'vb', 'bm']
def load(n, o):
    p = f'results/col_pan_{n}_{o}.jsonl'
    if not os.path.exists(p): return {}
    d = {}
    for l in open(p):
        r = json.loads(l)
        if 'score' in r: d[(r['seed'], r['color'])] = r['score']
    return d
alld = []
for o in opps:
    a, b = load(base, o), load(cand, o)
    ks = sorted(set(a) & set(b))
    if not ks: continue
    d = np.array([b[k] - a[k] for k in ks])
    same = np.mean(np.abs(d) < 1e-9)
    alld += list(d)
    print(f'{o}: n={len(ks)} diff={d.mean():+.3f}±{d.std()/np.sqrt(len(d)):.3f} identical={same:.2f}')
if alld:
    alld = np.array(alld); print(f'ALL: n={len(alld)} diff={alld.mean():+.3f}±{alld.std()/np.sqrt(len(alld)):.3f}')
