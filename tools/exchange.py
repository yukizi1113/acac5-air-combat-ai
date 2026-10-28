"""Fighter-war exchange metric per config: mean (enemy fighters killed - own fighters lost) per match, by opponent.
  python tools/exchange.py e1 e2 p2a14m ..."""
import sys, json, os
import numpy as np
for name in sys.argv[1:]:
    out = []
    for o in ['v7', 'sn', 'p2', 'vb', 'bm']:
        p = f'results/col_pan_{name}_{o}.jsonl'
        if not os.path.exists(p):
            continue
        d = []
        for l in open(p):
            r = json.loads(l)
            if 'score' not in r: continue
            me = r['color']
            kills = sum(1 for m in r['godlog'] if m['team'] == me and m['hit'] and not m['target'].endswith('E'))
            loss = sum(1 for m in r['godlog'] if m['team'] != me and m['hit'] and not m['target'].endswith('E'))
            d.append(kills - loss)
        if d:
            out.append(f"{o}: {np.mean(d):+.2f}±{np.std(d)/np.sqrt(len(d)):.2f}")
    print(f'{name:10s}', '  '.join(out))
