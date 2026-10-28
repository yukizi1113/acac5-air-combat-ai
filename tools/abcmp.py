"""Compare AB-queue variants against v9b baselines: python tools/abcmp.py Ax Bx Cx"""
import sys, json, os, numpy as np
def load(f):
    if not os.path.exists(f): return {}
    return {(r['seed'], r['color']): r['score'] for r in map(json.loads, open(f)) if 'score' in r}
base = {'s1': load('results/col_pan_p2a14m_v7.jsonl'), 's2': load('results/col_fin_AM_v7.jsonl')}
for nm in sys.argv[1:]:
    out = []; ds = []
    for key, f in [('s1', f'results/col_pan_{nm}_v7.jsonl'), ('s2', f'results/col_fin_{nm}_v7.jsonl')]:
        b = load(f); a = base[key]; ks = sorted(set(a) & set(b))
        if ks:
            d = [b[k] - a[k] for k in ks]; ds += d
            out.append(f'v7{key}: {np.mean([b[k] for k in ks]):.3f} d={np.mean(d):+.3f} (n={len(ks)})')
    v = list(load(f'results/col_pan_{nm}_vb.jsonl').values())
    if ds: out.append(f'v7 all d={np.mean(ds):+.3f}±{np.std(ds)/np.sqrt(len(ds)):.3f}')
    if v: out.append(f'vs v9b: {np.mean(v):.3f}±{np.std(v)/np.sqrt(len(v)):.3f} (n={len(v)})')
    print(nm, ' | '.join(out))
