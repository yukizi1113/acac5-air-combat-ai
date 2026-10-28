"""Markdown table of experiment results with bootstrap 95% CI of the mean match score.

  python tools/summarize_exps.py v3def v3novt v3nodef ...
"""
import sys, json
import numpy as np


def load(tag):
    rows = []
    for l in open(f'results/col_{tag}.jsonl'):
        r = json.loads(l)
        if 'score' in r:
            rows.append(r)
    return rows


def ci(x, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    bs = [rng.choice(x, len(x)).mean() for _ in range(n)]
    return np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def main(tags):
    print('| 設定 | 試合 | 勝 | 敗 | 分 | 平均得点 [95%CI] | 命中率 | 発射/試合 | 勝利時刻[s] | 自陣生存数* | 敵陣生存数* |')
    print('|---|---|---|---|---|---|---|---|---|---|---|')
    for t in tags:
        try:
            rows = load(t)
        except FileNotFoundError:
            continue
        if not rows:
            continue
        sc = [r['score'] for r in rows]
        lo, hi = ci(sc)
        w = sum(r['result'] == 'win' for r in rows); l = sum(r['result'] == 'loss' for r in rows)
        H = sum(r['hits'] for r in rows); Lc = sum(r['launched'] for r in rows)
        tw = [r['time'] for r in rows if r['result'] == 'win']
        al = np.mean([r['alive'] for r in rows])   # incl. escort
        oal = np.mean([r['opp_alive'] for r in rows])
        print(f"| {t} | {len(rows)} | {w} | {l} | {len(rows)-w-l} | {np.mean(sc):.3f} [{lo:.3f}, {hi:.3f}] | "
              f"{H/Lc if Lc else float('nan'):.2f} | {Lc/len(rows):.1f} | {np.mean(tw) if tw else float('nan'):.0f} | {al:.2f} | {oal:.2f} |")


if __name__ == '__main__':
    main(sys.argv[1:])
