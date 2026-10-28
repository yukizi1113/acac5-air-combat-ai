"""Self-play parameter search for the rule commander (successive halving on paired seeds).

  python tools/search.py --space space.json --opp agents/snap_v7 --me agents/snap_v8d --cands 8 --n0 10 --rounds 3

space.json: {"param": [low, high] | [choice1, choice2, ...] (lists of non-numbers or len>2 = choices)}
Each round evaluates the surviving candidates on the SAME seeds (variance reduction), keeps the top half by mean
match score, and doubles the number of new matches.  Candidate 0 is always the unmodified base (control).
Results: results/search_<tag>.jsonl (one line per candidate per round).
"""
import os, sys, json, argparse, subprocess, random
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def sample(space, rng):
    c = {}
    for k, v in space.items():
        if isinstance(v, list) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v):
            lo, hi = v
            c[k] = int(rng.integers(lo, hi + 1)) if all(isinstance(x, int) for x in v) else float(rng.uniform(lo, hi))
        else:
            c[k] = v[int(rng.integers(len(v)))]
    return c


def evaluate(me, opp, opp_args, cand, n, seed, tag, workers):
    out = f'results/col_srch_{tag}.jsonl'
    cmd = [sys.executable, 'tools/run_matches.py', '--me', me, '--opp', opp, '--n', str(n), '--workers', str(workers),
           '--seed', str(seed), '--out', out, '--me-args', json.dumps({'overrides': cand}), '--opp-args', opp_args]
    env = dict(os.environ, GOD_LOG='1', SHOT_LOG=os.path.join(ROOT, 'data', f'shots_col_srch_{tag}.jsonl'))
    subprocess.run(cmd, cwd=ROOT, capture_output=True, env=env)
    rows = [json.loads(l) for l in open(os.path.join(ROOT, out))]
    return [r['score'] for r in rows if 'score' in r]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--space', required=True)
    ap.add_argument('--me', default='agents/snap_v8d')
    ap.add_argument('--base', default='{}', help='overrides applied to every candidate')
    ap.add_argument('--opp', default='agents/snap_v7')
    ap.add_argument('--opp-args', default='{}')
    ap.add_argument('--opps', default='', help='JSON list of [module, args-json] pairs; overrides --opp')
    ap.add_argument('--cands', type=int, default=8)
    ap.add_argument('--n0', type=int, default=10)
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--seed', type=int, default=20000)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--tag', default='s1')
    a = ap.parse_args()
    space = json.load(open(a.space)) if os.path.exists(a.space) else json.loads(a.space)
    base = json.loads(a.base)
    rng = np.random.default_rng(a.seed)
    cands = [dict(base)] + [dict(base, **sample(space, rng)) for _ in range(a.cands - 1)]
    scores = {i: [] for i in range(len(cands))}
    alive = list(range(len(cands)))
    n, seed = a.n0, a.seed
    log = open(os.path.join(ROOT, f'results/search_{a.tag}.jsonl'), 'a')
    for rd in range(a.rounds):
        opps = json.loads(a.opps) if a.opps else [[a.opp, a.opp_args]]
        for i in alive:
            sc = []
            for k, (om, oa) in enumerate(opps):
                sc += evaluate(a.me, om, oa if isinstance(oa, str) else json.dumps(oa), cands[i], n, seed, f'{a.tag}_r{rd}_c{i}_o{k}', a.workers)
            scores[i] += sc
            rec = {'round': rd, 'cand': i, 'params': cands[i], 'n_total': len(scores[i]), 'mean': float(np.mean(scores[i])),
                   'round_mean': float(np.mean(sc)) if sc else None}
            log.write(json.dumps(rec) + '\n'); log.flush()
            print(f"[r{rd}] cand {i} mean={rec['mean']:.3f} (n={rec['n_total']}) {cands[i]}", flush=True)
        alive.sort(key=lambda i: -np.mean(scores[i]))
        alive = alive[:max(1, len(alive) // 2)]
        seed += 1000
        n *= 2
    best = alive[0]
    print('BEST', best, round(float(np.mean(scores[best])), 3), json.dumps(cands[best]))


if __name__ == '__main__':
    main()
