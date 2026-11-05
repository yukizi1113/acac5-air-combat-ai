"""RL pipeline for agents/TacR (decentralised ego-centric option policy).

  python rl2/train.py collect --mode teacher --episodes 96 --workers 4 --out data/rl2/bc          # BC data (v17)
  python rl2/train.py bc --data 'data/rl2/bc/*.pkl.gz' --out runs/rl2                                # -> policy_0000
  python rl2/train.py ppo --run runs/rl2 --iters 200 --eps 48 --workers 4                            # synchronous
  python rl2/train.py eval --weights runs/rl2/policy_0010.npz --n 16 --workers 4 [--opps v17,v15]
Distributed (Snowflake: one learner node, many actor nodes sharing a folder that is synced with a stage):
  python rl2/train.py learner --run SHARED/run --eps 48        # waits for rollouts in SHARED/run/inbox
  python rl2/train.py actor   --run SHARED/run --workers 4     # plays with the latest policy, writes rollouts
League (PFSP, Vinyals et al. 2019 / digest sec. 3): rule agents of every style seen in the field logs + frozen
snapshots of the policy + mirror self-play; opponents we score badly against are sampled more often.
"""
import os
import sys
import glob
import json
import time
import argparse
import random
import multiprocessing as mp
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)


def _play(job):
    os.chdir(ROOT)
    sys.path.insert(0, ROOT)
    for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ.setdefault(v, '1')
    from rl2.rollout import play
    return play(job)


def pool(workers):
    return mp.get_context('spawn').Pool(workers, maxtasksperchild=1)


class League:
    def __init__(self, path, base=None, mirror=0.2, snap_keep=6):
        from rl2.common import LEAGUE
        self.path = path
        self.rule = dict(LEAGUE)
        if base:
            self.rule = {k: v for k, v in self.rule.items() if k in base}
        self.mirror = mirror; self.snap_keep = snap_keep
        self.state = {'score': {}, 'n': {}, 'snaps': []}
        if os.path.exists(path):
            self.state.update(json.load(open(path)))

    def names(self):
        return list(self.rule) + [f'snap{v}' for v, _ in self.state['snaps']]

    def weight(self, nm):
        s = self.state['score'].get(nm, 0.5)
        return (1.0 - s) ** 2 + 0.02                     # PFSP "hard" weighting

    def sample(self, rng):
        if rng.random() < self.mirror:
            return 'self', {'mirror': True, 'name': 'self'}
        names = self.names()
        w = np.array([self.weight(n) for n in names]); w /= w.sum()
        nm = names[rng.choice(len(names), p=w)]
        if nm.startswith('snap'):
            path = dict((f'snap{v}', p) for v, p in self.state['snaps'])[nm]
            return nm, {'snapshot': path, 'name': nm}
        o = dict(self.rule[nm]); o['name'] = nm
        return nm, o

    def record(self, nm, score, alpha=0.1):
        s = self.state['score'].get(nm, 0.5)
        self.state['score'][nm] = (1 - alpha) * s + alpha * score
        self.state['n'][nm] = self.state['n'].get(nm, 0) + 1

    def add_snapshot(self, version, path):
        self.state['snaps'] = (self.state['snaps'] + [(version, path)])[-self.snap_keep:]

    def save(self):
        json.dump(self.state, open(self.path + '.tmp', 'w'), indent=1); os.replace(self.path + '.tmp', self.path)


def make_jobs(league, weights, version, n, seed0, rng):
    jobs = []
    for i in range(n):
        nm, opp = league.sample(rng)
        jobs.append(dict(seed=seed0 + i // 2, color='Red' if i % 2 == 0 else 'Blue', mode='policy', weights=weights,
                         opp=opp, version=version, tag=nm))
    return jobs


def latest(run):
    p = os.path.join(run, 'latest.json')
    if not os.path.exists(p):
        return None, None
    j = json.load(open(p))
    return j['version'], os.path.join(run, j['path'])


def log_to(run):
    os.makedirs(run, exist_ok=True)
    f = open(os.path.join(run, 'log.txt'), 'a')

    def log(msg):
        line = time.strftime('%H:%M:%S ') + msg
        print(line, flush=True); f.write(line + '\n'); f.flush()
    return log


# ---------------------------------------------------------------- commands
def cmd_collect(a):
    from rl2.common import LEAGUE
    from rl2.rio import save_rollout
    os.makedirs(a.out, exist_ok=True)
    opps = a.opps.split(',') if a.opps else list(LEAGUE)
    jobs = []
    for i in range(a.episodes):
        nm = opps[(i // 2) % len(opps)]
        o = dict(LEAGUE[nm]); o['name'] = nm
        jobs.append(dict(seed=a.seed + i // 2, color='Red' if i % 2 == 0 else 'Blue', mode='teacher', opp=o, tag=nm))
    t0 = time.time(); n = 0
    with pool(a.workers) as p:
        for r in p.imap_unordered(_play, jobs):
            if 'error' in r:
                print('ERR', r['error'][-500:]); continue
            save_rollout(r, os.path.join(a.out, f"bc_{r['seed']}_{r['color']}.pkl.gz")); n += 1
            if n % 8 == 0:
                print(f'{n} episodes, {3600 * n / (time.time() - t0):.0f}/h', flush=True)
    print(f'collected {n} in {(time.time() - t0) / 60:.1f} min')


def cmd_bc(a):
    from rl2.learner import make_net, train_bc, save
    from rl2.rio import load_rollout
    log = log_to(a.out)
    trajs = []
    for p in sorted(glob.glob(a.data)):
        trajs += load_rollout(p)['trajs']
    log(f'[bc] {len(trajs)} trajectories from {a.data}')
    net = make_net()
    train_bc(net, trajs, epochs=a.epochs, log=log)
    path = save(net, a.out, 0)
    log(f'[bc] saved {path}')


def _summ(results):
    by = {}
    for r in results:
        by.setdefault(r['tag'], []).append(r['score'])
    return ' '.join(f'{k}:{np.mean(v):.2f}({len(v)})' for k, v in sorted(by.items()))


def cmd_ppo(a):
    """Synchronous loop on one machine (local tests and single-node runs)."""
    from rl2.learner import make_net, PPO, save
    log = log_to(a.run)
    v, w = latest(a.run)
    if v is None:
        raise SystemExit('no policy in run dir (run bc first)')
    net = make_net(os.path.join(a.run, f'ckpt_{v:04d}.pt'))
    ppo = PPO(net, json.loads(a.hp) if a.hp else None); ppo.it = v
    league = League(os.path.join(a.run, 'league.json'), base=a.opps.split(',') if a.opps else None)
    rng = np.random.default_rng(a.seed + v)
    with pool(a.workers) as p:
        for it in range(a.iters):
            t0 = time.time()
            jobs = make_jobs(league, w, v, a.eps, a.seed + 1000 * (v + 1), rng)
            res = [r for r in p.imap_unordered(_play, jobs)]
            ok = [r for r in res if 'error' not in r]
            for r in res:
                if 'error' in r:
                    log('ERR ' + r['error'][-300:])
            for r in ok:
                league.record(r['tag'], r['score'])
            trajs = [t for r in ok for t in r['trajs']]
            t1 = time.time()
            st = ppo.update(trajs, log=log)
            v += 1
            w = save(net, a.run, v)
            if v % a.snap_every == 0:
                league.add_snapshot(v, w)
            league.save()
            log(f'[it {v}] score={np.mean([r["score"] for r in ok]):.3f} {_summ(ok)} play={t1 - t0:.0f}s learn={time.time() - t1:.0f}s')


def cmd_eval(a):
    from rl2.common import LEAGUE
    opps = a.opps.split(',') if a.opps else [k for k in LEAGUE if k != 'bm']
    jobs = []
    for nm in opps:
        for i in range(a.n):
            o = dict(LEAGUE[nm]); o['name'] = nm
            jobs.append(dict(seed=a.seed + i // 2, color='Red' if i % 2 == 0 else 'Blue',
                             mode='teacher' if a.weights == 'rule' else 'policy', weights=a.weights,
                             greedy=not a.sample, opp=o, tag=nm))
    res = []
    with pool(a.workers) as p:
        for r in p.imap_unordered(_play, jobs):
            if 'error' in r:
                print('ERR', r['error'][-300:]); continue
            r.pop('trajs', None); res.append(r)
    by = {}
    for r in res:
        by.setdefault(r['tag'], []).append(r['score'])
    allm = []
    for k, s in sorted(by.items()):
        m = np.mean(s); allm.append(m)
        print(f'{k:5s} n={len(s):3d} score={m:.3f} +- {1.96 * np.std(s) / max(1, len(s)) ** 0.5:.3f}')
    print(f'mean over opponents {np.mean(allm):.3f}')
    if a.out:
        with open(a.out, 'a') as f:
            for r in res:
                f.write(json.dumps({k: v for k, v in r.items()}) + '\n')


def cmd_learner(a):
    """Distributed learner: consumes rollout files from RUN/inbox (written by actors), publishes policies."""
    from rl2.learner import make_net, PPO, save
    from rl2.rio import load_rollout
    log = log_to(a.run)
    inbox = os.path.join(a.run, 'inbox'); os.makedirs(inbox, exist_ok=True)
    v, w = latest(a.run)
    net = make_net(os.path.join(a.run, f'ckpt_{v:04d}.pt'))
    ppo = PPO(net, json.loads(a.hp) if a.hp else None); ppo.it = v
    league = League(os.path.join(a.run, 'league.json'), base=a.opps.split(',') if a.opps else None); league.save()
    t_end = time.time() + a.budget_min * 60
    while time.time() < t_end and v < a.max_version:
        t0 = time.time(); batch = []; used = []
        while len(batch) < a.eps and time.time() < t_end:
            for p in sorted(glob.glob(os.path.join(inbox, '*.pkl.gz'))):
                if p in used:
                    continue
                try:
                    r = load_rollout(p)
                except Exception:
                    continue
                used.append(p)
                if 'error' in r:
                    log('ERR ' + r['error'][-300:]); continue
                if r['version'] < v - ppo.hp['max_lag']:
                    continue                                   # too stale
                batch.append(r)
            if len(batch) < a.eps:
                time.sleep(2.0)
        if len(batch) < max(4, a.eps // 4):
            break
        for p in used:
            os.replace(p, p.replace('/inbox/', '/used/')) if os.path.isdir(os.path.join(a.run, 'used')) else os.remove(p)
        for r in batch:
            league.record(r['tag'], r['score'])
        t1 = time.time()
        ppo.update([t for r in batch for t in r['trajs']], log=log)
        v += 1
        w = save(net, a.run, v)
        if v % a.snap_every == 0:
            league.add_snapshot(v, os.path.basename(w))
        league.save()
        lags = [v - 1 - r['version'] for r in batch]
        log(f'[it {v}] score={np.mean([r["score"] for r in batch]):.3f} {_summ(batch)} wait={t1 - t0:.0f}s '
            f'learn={time.time() - t1:.0f}s lag={np.mean(lags):.2f}')


def cmd_actor(a):
    """Distributed actor: plays episodes with the latest published policy, writes rollouts to RUN/inbox."""
    from rl2.rio import save_rollout
    inbox = os.path.join(a.run, 'inbox'); os.makedirs(inbox, exist_ok=True)
    rng = np.random.default_rng(a.seed + os.getpid())
    t_end = time.time() + a.budget_min * 60
    node = a.node or os.uname().nodename[:8]
    n = 0; t0 = time.time()
    with pool(a.workers) as p:
        pending = []
        while time.time() < t_end:
            while len(pending) < a.workers and time.time() < t_end:
                v, w = latest(a.run)
                if v is None:
                    time.sleep(5); continue
                league = League(os.path.join(a.run, 'league.json'))
                if league.state['snaps']:
                    league.state['snaps'] = [(sv, os.path.join(a.run, sp)) for sv, sp in league.state['snaps']]
                job = make_jobs(league, w, v, 1, int(rng.integers(1, 2 ** 30)), rng)[0]
                job['color'] = 'Red' if rng.random() < 0.5 else 'Blue'
                pending.append(p.apply_async(_play, (job,)))
            still = []
            for h in pending:
                if h.ready():
                    r = h.get(); n += 1
                    save_rollout(r, os.path.join(inbox, f'{node}_{n:06d}.pkl.gz'))
                else:
                    still.append(h)
            pending = still
            time.sleep(0.5)
    print(f'actor {node}: {n} episodes in {(time.time() - t0) / 60:.1f} min')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd')
    ap.add_argument('--mode', default='teacher'); ap.add_argument('--episodes', type=int, default=48)
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2)))
    ap.add_argument('--out', default=''); ap.add_argument('--data', default='')
    ap.add_argument('--run', default='runs/rl2'); ap.add_argument('--iters', type=int, default=10)
    ap.add_argument('--eps', type=int, default=48); ap.add_argument('--seed', type=int, default=70000)
    ap.add_argument('--epochs', type=int, default=6); ap.add_argument('--hp', default='')
    ap.add_argument('--opps', default=''); ap.add_argument('--weights', default='')
    ap.add_argument('--n', type=int, default=16); ap.add_argument('--sample', action='store_true')
    ap.add_argument('--snap-every', dest='snap_every', type=int, default=10)
    ap.add_argument('--budget-min', dest='budget_min', type=float, default=60.0)
    ap.add_argument('--max-version', dest='max_version', type=int, default=10 ** 6)
    ap.add_argument('--node', default='')
    a = ap.parse_args()
    {'collect': cmd_collect, 'bc': cmd_bc, 'ppo': cmd_ppo, 'eval': cmd_eval, 'learner': cmd_learner,
     'actor': cmd_actor}[a.cmd](a)


if __name__ == '__main__':
    main()
