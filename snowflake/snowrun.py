"""Batch match runner for a remote (Snowflake notebook) node: one job list, all cores, hard time budget, resumable.

  python snowrun.py --jobs jobs/phase1.jsonl --out out/phase1.jsonl --workers 6 --budget-min 20 [--maxtasks 10]

Each line of the job file: {"tag", "me", "me_ov", "opp", "opp_ov", "seed", "color", "idx"}.
- Results are appended line by line (flush per match), so an interrupted run loses at most the matches in flight.
- Jobs already present in --out (same tag/seed/color) are skipped: re-running the same command resumes.
- After --budget-min minutes no new match is started; the pool is terminated and a summary is printed.
- One process per match (--maxtasks 1): reusing a process re-registers agent classes and changes results.
"""
import os, sys, json, time, argparse, multiprocessing as mp, collections

ROOT = os.path.abspath(os.path.dirname(__file__))
# One match = one process = one core. Library thread pools (OpenMP in lightgbm/numpy) would otherwise start
# nproc threads per match that spin-wait against each other (Snowflake CPU node: 25% sys time, matches 3x slower).
# Spawned children inherit these.
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
# Same BLAS kernel on every machine: OpenBLAS picks a kernel per CPU (local sandbox: Prescott, Snowflake Xeon 8375C:
# SkylakeX) and the different rounding makes matches diverge. Forcing Prescott reproduces local results bit for bit
# (verified on a Snowflake CPU node, det_v17_v15 seed 41000).
# Measured cost: none that matters (Phase 2 on Snowflake ran ~175 matches/h per CPU_X64_S node with Prescott, same as
# SkylakeX). Set ACAC_EXACT_BLAS=0 to let OpenBLAS pick its own kernel.
if os.environ.get('ACAC_EXACT_BLAS', '1') == '1':
    os.environ.setdefault('OPENBLAS_CORETYPE', 'Prescott')
sys.path.insert(0, ROOT)


def usable_cpus():
    """CPUs this container may use: the cgroup quota (cpu.max) caps it below nproc on Snowflake nodes."""
    n = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)
    try:
        q, per = open('/sys/fs/cgroup/cpu.max').read().split()
        if q != 'max':
            n = min(n, max(1, int(int(q) / int(per))))
    except Exception:
        pass
    return n


def _run(job):
    os.chdir(ROOT)
    from tools.run_matches import worker
    os.environ['RUN_TAG'] = job['tag']
    j = (job['idx'], job['me'], job['opp'], job['color'], job['seed'], {'overrides': job['me_ov']},
         {'overrides': job['opp_ov']}, 1e9, False, '')
    r = worker(j)
    r['tag'] = job['tag']
    r.pop('events', None); r.pop('godlog', None)
    return r


def summary(path):
    by = collections.defaultdict(list)
    for l in open(path):
        r = json.loads(l)
        if 'score' in r:
            by[r['tag']].append(r)
    out = {}
    for tag, rs in sorted(by.items()):
        s = [r['score'] for r in rs]
        res = [r['result'] for r in rs]
        n = len(s); m = sum(s) / n
        sd = (sum((x - m) ** 2 for x in s) / max(1, n - 1)) ** 0.5
        out[tag] = dict(n=n, score=round(m, 3), se=round(sd / n ** 0.5, 3), win=res.count('win'), loss=res.count('loss'),
                        draw=res.count('Draw') + res.count('draw'), wall=round(sum(r['wall'] for r in rs) / n, 1),
                        simt=round(sum(r['time'] for r in rs) / n))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=0, help='0 = usable CPUs (cgroup quota)')
    ap.add_argument('--budget-min', type=float, default=30.0)
    ap.add_argument('--maxtasks', type=int, default=1)
    ap.add_argument('--start', default='spawn', choices=['spawn', 'forkserver'],
                    help='forkserver: simulator imported once, each match still gets a fresh process')
    ap.add_argument('--shard', default='0/1', help='i/k: run every k-th job starting at i (one shard per node)')
    a = ap.parse_args()
    if a.workers <= 0:
        a.workers = usable_cpus()
    jobs = [json.loads(l) for l in open(a.jobs) if l.strip()]
    si, sk = map(int, a.shard.split('/'))
    jobs = jobs[si::sk]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            try:
                r = json.loads(l); done.add((r['tag'], r['seed'], r['color']))
            except Exception:
                pass
    todo = [j for j in jobs if (j['tag'], j['seed'], j['color']) not in done]
    print(f'jobs {len(jobs)}, already done {len(jobs) - len(todo)}, to run {len(todo)}, workers {a.workers}', flush=True)
    t0 = time.time(); deadline = t0 + a.budget_min * 60
    ctx = mp.get_context(a.start)
    if a.start == 'forkserver':
        ctx.set_forkserver_preload(['numpy', 'lightgbm', 'tools.utils', 'tools.fight_ext'])
    pool = ctx.Pool(a.workers, maxtasksperchild=a.maxtasks)
    pending = collections.deque(todo); running = []; n_done = 0
    with open(a.out, 'a') as fo:
        while pending or running:
            while pending and len(running) < a.workers and time.time() < deadline:
                running.append(pool.apply_async(_run, (pending.popleft(),)))
            still = []
            for h in running:
                if h.ready():
                    try:
                        r = h.get()
                    except Exception as ex:
                        r = {'error': repr(ex)}
                    fo.write(json.dumps(r) + '\n'); fo.flush(); n_done += 1
                    if n_done % 10 == 0:
                        el = time.time() - t0
                        print(f'{n_done} done, {el / 60:.1f} min, {3600 * n_done / el:.0f} matches/h', flush=True)
                else:
                    still.append(h)
            running = still
            if time.time() >= deadline and not running:
                break
            if time.time() >= deadline + 600:        # matches in flight get 10 more minutes at most
                print('hard stop: terminating matches in flight', flush=True)
                break
            time.sleep(0.5)
    pool.terminate()
    el = time.time() - t0
    print(f'finished {n_done} matches in {el / 60:.1f} min ({3600 * n_done / max(el, 1):.0f}/h), {len(pending)} not started')
    print(json.dumps(summary(a.out), indent=1))


if __name__ == '__main__':
    main()
