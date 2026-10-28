"""python tools/panel_summary.py NAME [NAME ...] -> per-opponent mean score and panel mean"""
import sys, json, os
import numpy as np
OPPS = ['v7', 'sn', 'ru', 'p2', 'vb', 'bm']
for name in sys.argv[1:]:
    line, means = [], []
    for o in OPPS:
        p = f'results/col_pan_{name}_{o}.jsonl'
        if not os.path.exists(p):
            line.append(f'{o}: -'); continue
        rows = [json.loads(l) for l in open(p)]
        rows = [r for r in rows if 'score' in r]
        if not rows:
            line.append(f'{o}: -'); continue
        sc = np.mean([r['score'] for r in rows]); w = sum(r['result'] == 'win' for r in rows)
        means.append(sc)
        line.append(f"{o}: {sc:.3f} ({w}/{len(rows)})")
    print(f'{name:14s} panel={np.mean(means) if means else float("nan"):.3f}  ' + '  '.join(line))
