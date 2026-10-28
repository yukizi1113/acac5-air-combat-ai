"""python tools/quick_sum.py results/a.json [results/b.json ...] : score / W-L-D / hits / launches per file"""
import json, sys, numpy as np
for p in sys.argv[1:]:
    R = [json.loads(l) for l in open(p) if l.strip()]
    R = [r for r in R if 'score' in r]
    if not R:
        print(p, 'no results'); continue
    s = np.array([r['score'] for r in R]); res = [r['result'] for r in R]
    print(f"{p}: n={len(R)} score={s.mean():.3f}±{s.std()/np.sqrt(len(s)):.3f} W/L/D={res.count('win')}/{res.count('loss')}/{res.count('draw')} "
          f"launch={np.mean([r['launched'] for r in R]):.1f} hits={np.mean([r['hits'] for r in R]):.1f} "
          f"opp_launch={np.mean([r['opp_launched'] for r in R]):.1f} opp_hits={np.mean([r['opp_hits'] for r in R]):.1f} T={np.median([r['time'] for r in R]):.0f}")
