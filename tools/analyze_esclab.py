"""Summarise the escort-strike lab: hit rate and time of flight by launch altitude / range / aspect / support.

  python tools/analyze_esclab.py results/esclab1.jsonl
"""
import sys, json
import numpy as np


def load(paths):
    rows = []
    for p in paths:
        for l in open(p):
            r = json.loads(l)
            if r.get('D') is not None:
                rows.append(r)
    return rows


def table(rows, key, bins, label):
    print(f'-- by {label}')
    v = np.array([key(r) for r in rows], float)
    h = np.array([r['hit'] for r in rows], bool)
    tof = np.array([r['tof'] for r in rows], float)
    for lo, hi in zip(bins[:-1], bins[1:]):
        k = (v >= lo) & (v < hi)
        if k.sum():
            print(f'  {lo:>7.1f}-{hi:<7.1f} n={k.sum():3d} hit={h[k].mean():.2f}  tof(hit)={np.median(tof[k & h]) if (k & h).any() else float("nan"):5.0f}')


def main(paths):
    rows = load(paths)
    print('episodes with a shot:', len(rows))
    table(rows, lambda r: r['D'] / 1000, [40, 60, 70, 80, 90, 100, 110, 130], 'range km')
    table(rows, lambda r: r['alt_s'] / 1000, [9, 11, 12, 13, 14, 15, 17], 'shooter alt km')
    table(rows, lambda r: r['asp'], [0, 30, 60, 90, 120, 150, 181], 'aspect deg (0=escort flying at shooter)')
    for S in sorted({r['meta']['S'] for r in rows}):
        sub = [r for r in rows if r['meta']['S'] == S]
        print(f'  support={S}: n={len(sub)} hit={np.mean([r["hit"] for r in sub]):.2f}')
    # 2-D: range x altitude
    print('-- hit rate: range (rows) x altitude (cols 10-12,12-14,14-17)')
    for lo, hi in [(40, 70), (70, 85), (85, 100), (100, 130)]:
        line = []
        for alo, ahi in [(9, 12), (12, 14), (14, 17)]:
            k = [r for r in rows if lo <= r['D'] / 1000 < hi and alo <= r['alt_s'] / 1000 < ahi]
            line.append(f'{np.mean([r["hit"] for r in k]):.2f}({len(k)})' if k else '   -   ')
        print(f'  {lo}-{hi}km  ' + '  '.join(line))
    miss = [r for r in rows if not r['hit']]
    if miss:
        print('misses: end mode', {m: sum(1 for r in miss if r['mode_end'] == m) for m in {r['mode_end'] for r in miss}},
              ' median end speed', np.median([r['vend'] for r in miss if r['vend'] is not None]),
              ' tof>=148:', sum(1 for r in miss if r['tof'] >= 148))


if __name__ == '__main__':
    main(sys.argv[1:])
