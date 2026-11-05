"""Snowflake notebook side of the RL run (runs in the notebook kernel: Python 3.12 with torch + Ray).

Cell:
    exec(open('/tmp/boot/rl2_boot.py').read())          # downloads code, installs on the head node, imports this
    R = SF(session, nodes=6)                              # scale the notebook to 6 nodes, install on every node
    R.collect_bc(240); R.bc(); R.evaluate('bc', 24); R.ppo(iters=15, eps=48, budget_min=50)
Everything important is copied to the stage (acac5/rl2/<run>/) as it is produced.
"""
import os
import sys
import io
import json
import time
import glob
import tarfile
import numpy as np

STAGE = '@SNOWFLAKE_LEARNING_DB.CURSOR_MCP.CASMI_STAGE/acac5'
W = '/tmp/acac5'
ACAC = W + '/acac'
RUNS = '/tmp/rl2run'


class SF:
    def __init__(self, session, nodes=6, run='run1', log_name='log.txt'):
        self.s = session
        self.run_name = run
        self.run = os.path.join(RUNS, run)
        os.makedirs(self.run, exist_ok=True)
        os.environ['LEARNER_THREADS'] = '2'
        os.environ.setdefault('ACAC_ROOT', ACAC)
        sys.path.insert(0, ACAC)
        self.logf = open(os.path.join(self.run, log_name), 'a')
        self.nodes = nodes
        self._cluster(nodes)

    # ------------------------------------------------------------------ infra
    def log(self, msg):
        line = time.strftime('%H:%M:%S ') + str(msg)
        print(line, flush=True)
        self.logf.write(line + '\n'); self.logf.flush()

    def put(self, path, sub=''):
        try:
            self.s.file.put(path, f'{STAGE}/rl2/{self.run_name}/{sub}', auto_compress=False, overwrite=True)
        except Exception as ex:
            self.log(f'stage put failed {path}: {ex}')

    def _cluster(self, n):
        import ray
        from rl2.driver import RayExec, setup_ray_nodes
        if n > 1:
            from snowflake.ml.runtime_cluster import scale_cluster
            t = time.time()
            ok = scale_cluster(n, options={'block_until_min_cluster_size': max(2, n - 1)})
            self.log(f'scale_cluster({n}) -> {ok} in {time.time() - t:.0f}s')
        if not ray.is_initialized():
            ray.init(address='auto', ignore_reinit_error=True)
        pack = None
        if os.path.exists(W + '/pack.tar'):
            pack = open(W + '/pack.tar', 'rb').read()
        code = open(W + '/code.tar.gz', 'rb').read()
        res = setup_ray_nodes(pack, code)
        self.log(f'nodes: {res}')
        self.ex = RayExec(reserve_head=1)
        self.log(f'cluster CPUs {ray.cluster_resources().get("CPU")}, episode slots {self.ex.slots}')

    # ------------------------------------------------------------------ phases
    def collect_bc(self, n=240, seed=90000):
        from rl2.common import LEAGUE
        from rl2.driver import run_fixed
        from rl2.rio import save_rollout
        d = os.path.join(RUNS, 'bc_data'); os.makedirs(d, exist_ok=True)
        opps = [k for k in LEAGUE]
        jobs = []
        for i in range(n):
            nm = opps[(i // 2) % len(opps)]
            o = dict(LEAGUE[nm]); o['name'] = nm
            jobs.append(dict(seed=seed + i // 2, color='Red' if i % 2 == 0 else 'Blue', mode='teacher', opp=o, tag=nm))
        t = time.time()

        def keep(r):
            if 'error' not in r:
                save_rollout(r, os.path.join(d, f"bc_{r['seed']}_{r['color']}.pkl.gz"))
            return {k: v for k, v in r.items() if k != 'trajs'}
        out = run_fixed(self.ex, jobs, {}, log=self.log, on_result=keep)
        err = [r for r in out if 'error' in r]
        self.log(f'BC data: {len(out) - len(err)} episodes ({len(err)} errors) in {(time.time() - t) / 60:.1f} min; '
                 f'teacher score {np.mean([r["score"] for r in out if "score" in r]):.3f}')
        for r in err[:3]:
            self.log('ERR ' + str(r['error'])[-400:])

    def bc(self, epochs=12):
        from rl2.learner import make_net, train_bc, save
        from rl2.rio import load_rollout
        trajs = []
        for p in sorted(glob.glob(os.path.join(RUNS, 'bc_data', '*.pkl.gz'))):
            trajs += load_rollout(p)['trajs']
        self.log(f'BC on {len(trajs)} trajectories')
        net = make_net()
        train_bc(net, trajs, epochs=epochs, log=self.log)
        p = save(net, self.run, 0)
        for f in (p, os.path.join(self.run, 'ckpt_0000.pt'), os.path.join(self.run, 'latest.json')):
            self.put(f)
        self.log(f'BC policy saved {p}')

    def evaluate(self, which='latest', n=24, opps=None, seed=95000, tag=''):
        """Greedy policy vs rule agents on the same seeds as the rule teacher (paired comparison)."""
        from rl2.common import LEAGUE
        from rl2.driver import run_fixed
        from rl2.train import latest
        if which == 'latest':
            v, wp = latest(self.run)
        elif which == 'bc':
            wp = os.path.join(self.run, 'policy_0000.npz')
        else:
            wp = which
        wname = os.path.basename(wp)
        files = {wname: open(wp, 'rb').read()}
        opps = opps or [k for k in LEAGUE if k != 'bm']
        jobs = []
        for nm in opps:
            o = dict(LEAGUE[nm]); o['name'] = nm
            for i in range(n):
                base = dict(seed=seed + i // 2, color='Red' if i % 2 == 0 else 'Blue', opp=o, tag=nm)
                jobs.append(dict(base, mode='policy', weights_name=wname, greedy=True, version=-1))
                jobs.append(dict(base, mode='teacher', version=-2))
        drop = lambda r: {k: v for k, v in r.items() if k != 'trajs'}
        out = [r for r in run_fixed(self.ex, jobs, files, log=self.log, on_result=drop) if 'error' not in r]
        pol = {(r['tag'], r['seed'], r['color']): r['score'] for r in out if r['mode'] == 'policy'}
        rule = {(r['tag'], r['seed'], r['color']): r['score'] for r in out if r['mode'] == 'teacher'}
        rows = []
        for nm in opps:
            ks = [k for k in pol if k[0] == nm and k in rule]
            if not ks:
                continue
            a = np.array([pol[k] for k in ks]); b = np.array([rule[k] for k in ks]); dlt = a - b
            rows.append((nm, len(ks), a.mean(), b.mean(), dlt.mean(), 1.96 * dlt.std() / max(1, len(ks)) ** 0.5))
            self.log(f'eval {tag or wname} vs {nm:4s} n={len(ks):3d} policy {a.mean():.3f} rule(v17) {b.mean():.3f} '
                     f'diff {dlt.mean():+.3f} +- {rows[-1][5]:.3f}')
        if rows:
            self.log(f'eval {tag or wname} MEAN policy {np.mean([r[2] for r in rows]):.3f} rule {np.mean([r[3] for r in rows]):.3f} '
                     f'diff {np.mean([r[4] for r in rows]):+.3f}')
        json.dump(out, open(os.path.join(self.run, f'eval_{tag or wname}.json'), 'w'))
        self.put(os.path.join(self.run, f'eval_{tag or wname}.json'))
        return rows

    def ppo(self, iters=15, eps=48, budget_min=50.0, hp=None, put_every=5, snap_every=10):
        from rl2.driver import train

        def on_version(v, run):
            if v % put_every == 0:
                for f in (f'policy_{v:04d}.npz', f'ckpt_{v:04d}.pt', 'latest.json', 'league.json', 'log.txt'):
                    p = os.path.join(run, f)
                    if os.path.exists(p):
                        self.put(p)
        v = train(self.ex, self.run, iters=iters, eps=eps, budget_min=budget_min, hp=hp, log=self.log,
                  on_version=on_version, snap_every=snap_every)
        on_version(v - v % put_every if v % put_every else v, self.run)
        for f in (f'policy_{v:04d}.npz', f'ckpt_{v:04d}.pt', 'latest.json', 'league.json', 'log.txt'):
            p = os.path.join(self.run, f)
            if os.path.exists(p):
                self.put(p)
        self.log(f'PPO stopped at version {v}')
        return v

    def shutdown(self):
        try:
            from snowflake.ml.runtime_cluster import scale_cluster
            scale_cluster(1)
            self.log('scaled back to 1 node')
        except Exception as ex:
            self.log(f'scale down failed: {ex}')
