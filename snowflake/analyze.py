"""Merge result shards and apply the stop rules of SNOWFLAKE_PLAN.md.   python analyze.py out/phase2_*.jsonl

h2h_v17_v15  : mean score of v17 vs v15 with 95% CI.
v15_X / v17_X: same seeds and colours for both, so the paired difference (v17 - v15) per opponent X is reported.
Stop rule (checked after each tranche): every CI either excludes 0 or is narrower than +-0.05 -> stop.
"""
import sys, json, glob, collections, math


def load(paths):
    rs = []
    for p in paths:
        for f in glob.glob(p):
            for l in open(f):
                r = json.loads(l)
                if 'score' in r and not r.get('error'):
                    rs.append(r)
    return rs


def ci(xs):
    n = len(xs)
    if n < 2:
        return (sum(xs) / max(n, 1), float('nan'))
    m = sum(xs) / n
    sd = (sum((x - m) ** 2 for x in xs) / (n - 1)) ** 0.5
    return m, 1.96 * sd / math.sqrt(n)


def main(paths):
    rs = load(paths)
    by = collections.defaultdict(dict)
    for r in rs:
        by[r['tag']][(r['seed'], r['color'])] = r['score']
    print(f'{len(rs)} results')
    decided = []
    if 'h2h_v17_v15' in by:
        m, h = ci(list(by['h2h_v17_v15'].values()))
        print(f'h2h v17 vs v15: n={len(by["h2h_v17_v15"])}  {m:.3f} +- {h:.3f}')
        decided.append(abs(m - 0.5) > h or h < 0.05)
    opps = sorted({t.split('_', 1)[1] for t in by if t.startswith(('v15_', 'v17_'))})
    for o in opps:
        a, b = by.get(f'v15_{o}', {}), by.get(f'v17_{o}', {})
        keys = sorted(set(a) & set(b))
        if not keys:
            continue
        d = [b[k] - a[k] for k in keys]
        m, h = ci(d)
        print(f'vs {o:4s}: n={len(keys):3d}  v15 {sum(a[k] for k in keys)/len(keys):.3f}  v17 {sum(b[k] for k in keys)/len(keys):.3f}'
              f'  diff {m:+.3f} +- {h:.3f}')
        decided.append(abs(m) > h or h < 0.05)
    print('STOP (all decided)' if decided and all(decided) else 'continue')


if __name__ == '__main__':
    main(sys.argv[1:] or ['out/*.jsonl'])
