"""Evaluate saved checkpoints against fixed opponents and pick the best one.

  python rl/eval_ckpts.py --run runs/ppo --n 12 --workers 6
Writes <run>/eval.csv (checkpoint, opponent, mean score, win rate, hit rate) and prints the best checkpoint
by mean score over opponents (benchmark, rule commander, rusher).
"""
import os, sys, glob, json, csv, argparse, subprocess

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OPPS = [
    ('bench', 'Agents/BenchMark', '{"type":"Fixed"}'),
    ('rule', 'agents/snap_v7', '{}'),
    ('rusher', 'agents/snap_v3', '{"overrides":{"n_defenders_max":0,"engage_radius":0.0,"use_vthreat":false,"k_fighter":-1.0,"k_escort":0.6}}'),
]


def summarize(path):
    rows = [json.loads(l) for l in open(path)]
    rows = [r for r in rows if 'score' in r]
    if not rows:
        return 0, 0, 0
    sc = sum(r['score'] for r in rows) / len(rows)
    wr = sum(r['result'] == 'win' for r in rows) / len(rows)
    hr = [r['hits'] / r['launched'] for r in rows if r['launched'] > 0]
    return sc, wr, (sum(hr) / len(hr) if hr else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--n', type=int, default=12)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--pattern', default='ckpt_*.pt')
    a = ap.parse_args()
    ckpts = sorted(glob.glob(os.path.join(ROOT, a.run, a.pattern)))
    out = open(os.path.join(ROOT, a.run, 'eval.csv'), 'a', newline='')
    w = csv.writer(out)
    best = None
    for ck in ckpts:
        rel = os.path.relpath(ck, os.path.join(ROOT, 'agents', 'Tac'))
        me_args = json.dumps({'policy': 'nn', 'weights': rel})
        tot = 0.0
        for name, mod, oargs in OPPS:
            res = os.path.join('results', f'eval_{os.path.basename(a.run)}_{os.path.basename(ck)[:-3]}_{name}.jsonl')
            subprocess.run([sys.executable, 'tools/run_matches.py', '--me', 'agents/Tac', '--opp', mod, '--opp-args', oargs,
                            '--me-args', me_args, '--n', str(a.n), '--workers', str(a.workers), '--seed', '90000',
                            '--out', res], cwd=ROOT, capture_output=True)
            sc, wr, hr = summarize(os.path.join(ROOT, res))
            w.writerow([os.path.basename(ck), name, round(sc, 4), round(wr, 3), round(hr, 3)]); out.flush()
            print(f'{os.path.basename(ck)} vs {name}: score={sc:.3f} win={wr:.2f} hit={hr:.2f}', flush=True)
            tot += sc
        tot /= len(OPPS)
        if best is None or tot > best[1]:
            best = (ck, tot)
    if best:
        print('BEST', best[0], round(best[1], 4))


if __name__ == '__main__':
    main()
