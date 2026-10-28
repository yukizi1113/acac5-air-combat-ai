"""Parallel match runner with detailed stats.

Usage (from project root):
  python tools/run_matches.py --me agents/Tac --opp Agents/BenchMark --n 8 --workers 2 --out results/x.jsonl
Each match alternates colors. Writes one JSON line per match.
"""
import os, sys, json, time, argparse, multiprocessing as mp, traceback

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def _agent_info(agent_id, module_path, args_override=None):
    module_name = '.'.join(module_path.strip('/').split('/'))
    ap = os.path.join(ROOT, module_path, 'args.json')
    args = json.load(open(ap)) if os.path.exists(ap) else {}
    if args_override:
        args.update(args_override)
    return {'userModelID': agent_id, 'userModuleID': module_name, 'args': args}


def worker(job):
    os.chdir(ROOT)
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    import numpy as np
    from tools.utils import make_match_info
    from tools.fight_ext import Fight
    idx, me, opp, color, seed, me_args, opp_args, time_out, replay, outdir = job
    np.random.seed(seed)
    os.environ['MATCH_ID'] = f"{os.environ.get('RUN_TAG','run')}_{seed}_{color}_{idx}" 
    mine = _agent_info('0', me, me_args)
    theirs = _agent_info('1', opp, opp_args)
    red, blue = (mine, theirs) if color == 'Red' else (theirs, mine)
    mi = make_match_info(red=red, blue=blue, replay=replay, movie_dir=outdir, time_out=time_out,
                         common_dir='common', control_init='random', youth=False, make_log=False,
                         visualize=False)
    mi['seed'] = seed
    mi['god_log'] = bool(os.environ.get('GOD_LOG'))
    t0 = time.time()
    try:
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            f = Fight(mi)
            f.make_matchInfo()
            f.set_match()
            winner, d = f.run(False)
        out = buf.getvalue()
    except Exception:
        return {'idx': idx, 'error': traceback.format_exc()}
    opp_color = 'Blue' if color == 'Red' else 'Red'
    res = {
        'idx': idx, 'seed': seed, 'color': color, 'winner': d.get('winner'),
        'result': 'win' if d.get('winner') == color else ('loss' if d.get('winner') == opp_color else d.get('winner')),
        'endReason': d['endReason'], 'time': d['spentTime'],
        'score': d['scores'][color], 'oppscore': d['scores'][opp_color],
        'hits': d['hits'].get(color, 0), 'launched': d['launched'].get(color, 0),
        'opp_hits': d['hits'].get(opp_color, 0), 'opp_launched': d['launched'].get(opp_color, 0),
        'alive': d['numAlives'][color], 'opp_alive': d['numAlives'][opp_color],
        'wall': time.time() - t0, 'error': d['error'],
        'err_msg': (d['red_error_message'] + d['blue_error_message'])[-3000:],
        'timeouts': out.count('TimeOut'),
        'events': d.get('events', []),
        'godlog': d.get('godlog'),
        'match_id': os.environ.get('MATCH_ID'),
    }
    # agent-side debug lines
    dbg = [l for l in out.splitlines() if l.startswith('[DBG]')]
    if dbg:
        res['dbg'] = dbg[-200:]
    return res


def summarize(rows):
    import numpy as np
    rows = [r for r in rows if 'score' in r]
    if not rows:
        return 'no results'
    n = len(rows)
    w = sum(r['result'] == 'win' for r in rows)
    l = sum(r['result'] == 'loss' for r in rows)
    sc = np.mean([r['score'] for r in rows])
    hr = [r['hits'] / r['launched'] for r in rows if r['launched'] > 0]
    tw = [r['time'] for r in rows if r['result'] == 'win']
    reasons = {}
    for r in rows:
        reasons[r['endReason']] = reasons.get(r['endReason'], 0) + 1
    return (f"n={n} win={w} loss={l} draw={n-w-l} meanScore={sc:.3f} "
            f"hitRate={np.mean(hr) if hr else float('nan'):.2f} launched/match={np.mean([r['launched'] for r in rows]):.1f} "
            f"winTime={np.mean(tw) if tw else float('nan'):.0f}s alive={np.mean([r['alive'] for r in rows]):.2f} "
            f"oppAlive={np.mean([r['opp_alive'] for r in rows]):.2f} reasons={reasons} "
            f"timeouts={sum(r['timeouts'] for r in rows)} errors={sum(bool(r['error']) for r in rows)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--me', default='agents/Tac')
    ap.add_argument('--opp', default='Agents/BenchMark')
    ap.add_argument('--n', type=int, default=4)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--seed', type=int, default=1000)
    ap.add_argument('--out', default='results/run.jsonl')
    ap.add_argument('--me-args', default='{}')
    ap.add_argument('--opp-args', default='{}')
    ap.add_argument('--time-out', type=float, default=3.0)
    ap.add_argument('--replay', type=int, default=0)
    a = ap.parse_args()
    os.environ['RUN_TAG'] = os.path.basename(a.out).replace('.jsonl', '')
    os.makedirs(os.path.dirname(os.path.join(ROOT, a.out)) or '.', exist_ok=True)
    jobs = [(i, a.me, a.opp, 'Red' if i % 2 == 0 else 'Blue', a.seed + i // 2,
             json.loads(a.me_args), json.loads(a.opp_args), a.time_out, a.replay,
             os.path.join(ROOT, 'results', 'replay')) for i in range(a.n)]
    rows = []
    ctx = mp.get_context('spawn')
    t0 = time.time()
    with ctx.Pool(a.workers, maxtasksperchild=1) as pool, open(os.path.join(ROOT, a.out), 'w') as fo:
        for r in pool.imap_unordered(worker, jobs):
            rows.append(r)
            fo.write(json.dumps(r) + '\n'); fo.flush()
            if 'error' in r and isinstance(r['error'], str):
                print('ERROR', r['idx'], r['error'][-2000:])
            else:
                print(f"#{r['idx']} {r['color']} {r['result']} {r['endReason']} t={r['time']:.0f} "
                      f"score={r['score']:.3f} hit={r['hits']}/{r['launched']} opp={r['opp_hits']}/{r['opp_launched']} "
                      f"alive={r['alive']:.0f}/{r['opp_alive']:.0f} wall={r['wall']:.0f}s"
                      + (f" ERR {r['err_msg'][-800:]}" if r['error'] else ''), flush=True)
    print(summarize(rows))
    print(f'total wall {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
