"""Summarise engagement-lab results: hit rate by evasion, post-launch support and launch range."""
import json, glob, sys
import numpy as np


def load(shots_glob, res_glob):
    out = {}
    for rf in glob.glob(res_glob):
        for l in open(rf):
            r = json.loads(l)
            for m in r.get('godlog') or []:
                out[(r['match_id'], m['name'])] = m
    rows = []
    for sf in glob.glob(shots_glob):
        for l in open(sf):
            s = json.loads(l)
            m = out.get((s.get('match'), s['msl']))
            if m is None:
                continue
            s['hit'] = int(m['hit']); s['final_mode'] = m['traj'][-1][-1] if m['traj'] else ''
            rows.append(s)
    return rows


def table(rows, key, bins=((0, 15), (15, 25), (25, 35), (35, 50), (50, 90))):
    keys = sorted(set(r.get(key, '') for r in rows))
    hdr = f"{key:>12} | " + ' | '.join(f'{lo}-{hi}km' for lo, hi in bins) + ' | all'
    print(hdr)
    for k in keys:
        cells = []
        for lo, hi in bins:
            sel = [r['hit'] for r in rows if r.get(key) == k and lo <= r['d'] / 1000 < hi]
            cells.append(f"{np.mean(sel):.2f}({len(sel)})" if sel else '   -   ')
        al = [r['hit'] for r in rows if r.get(key) == k]
        print(f"{k:>12} | " + ' | '.join(cells) + f" | {np.mean(al):.2f}({len(al)})")


if __name__ == '__main__':
    sg = sys.argv[1] if len(sys.argv) > 1 else 'data/shots_col_lab*.jsonl'
    rg = sys.argv[2] if len(sys.argv) > 2 else 'results/col_lab*.jsonl'
    rows = load(sg, rg)
    print('shots', len(rows), 'hit rate', np.mean([r['hit'] for r in rows]).round(3))
    print('\n== by target evasion (E)'); table(rows, 'E')
    print('\n== by shooter post-launch behaviour (S)'); table(rows, 'S')
    print('\n== evasion x support (all ranges)')
    for E in sorted(set(r['E'] for r in rows)):
        print(f"{E:>12}: " + '  '.join(f"{S}={np.mean([r['hit'] for r in rows if r['E']==E and r['S']==S]):.2f}"
                                       for S in sorted(set(r['S'] for r in rows)) if any(r['E']==E and r['S']==S for r in rows)))
    asp = lambda r: 'head' if r['aspect'] < 0.8 else ('beam' if r['aspect'] < 2.3 else 'tail')
    for r in rows:
        r['asp'] = asp(r)
    print('\n== by target aspect at launch'); table(rows, 'asp')
