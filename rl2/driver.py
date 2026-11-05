"""Training driver that runs in the notebook kernel (Snowflake: Python 3.12 + torch + Ray) or locally.

The simulator needs Python 3.10, so every episode runs as `python3.10 rl2/play_cli.py job.json out.pkl.gz` in a
subprocess on whichever node Ray schedules it; the driver (learner) only needs torch.
Executors:
  LocalExec : process pool on this machine (tests)
  RayExec   : Ray tasks over all nodes of a multi-node notebook (scale_cluster); one CPU per episode
Asynchronous actor-learner (IMPALA-style, but PPO): episodes keep flowing while the learner updates; each rollout
carries the policy version it was played with, the PPO ratio uses the recorded behaviour log-probs, and rollouts more
than HP['max_lag'] versions old are dropped (docs/RL_DESIGN_DIGEST.md, item 4).
"""
import os
import sys
import json
import time
import glob
import subprocess
import tempfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
SIM_ROOT = os.environ.get('ACAC_ROOT', ROOT)               # where agents/, rl2/, tools/ live on every node
SIM_PY = os.environ.get('ACAC_PY', '/tmp/acenv/bin/python')
FILES_DIR = os.environ.get('ACAC_FILES', '/tmp/rl2files')


def _sim_env():
    env = dict(os.environ)
    ld = '/tmp/acenv/ldpath.sh'
    if os.path.exists(ld):
        line = open(ld).read().strip()
        if 'LD_LIBRARY_PATH=' in line:
            val = line.split('LD_LIBRARY_PATH=', 1)[1].split(':$LD_LIBRARY_PATH')[0]
            env['LD_LIBRARY_PATH'] = val + ':' + env.get('LD_LIBRARY_PATH', '')
    for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        env[v] = '1'
    env.setdefault('OPENBLAS_CORETYPE', 'Prescott')
    return env


def run_episode(job, files, timeout=2400):
    """Runs on a node: materialise policy files, play one episode in the 3.10 env, return the gz-pickled result."""
    if SIM_ROOT not in sys.path:
        sys.path.insert(0, SIM_ROOT)          # Ray worker processes do not inherit the driver's sys.path
    os.makedirs(FILES_DIR, exist_ok=True)
    for name, data in files.items():
        p = os.path.join(FILES_DIR, name)
        if not os.path.exists(p):
            tmp = p + f'.{os.getpid()}.tmp'
            open(tmp, 'wb').write(data); os.replace(tmp, p)
    job = dict(job)
    if job.get('weights_name'):
        job['weights'] = os.path.join(FILES_DIR, job.pop('weights_name'))
    if job.get('opp', {}).get('snapshot_name'):
        job['opp'] = dict(job['opp']); job['opp']['snapshot'] = os.path.join(FILES_DIR, job['opp'].pop('snapshot_name'))
    d = tempfile.mkdtemp(prefix='ep_')
    jp, op = os.path.join(d, 'job.json'), os.path.join(d, 'out.pkl.gz')
    json.dump(job, open(jp, 'w'))
    try:
        r = subprocess.run([SIM_PY, os.path.join(SIM_ROOT, 'rl2', 'play_cli.py'), jp, op], cwd=SIM_ROOT,
                           env=_sim_env(), capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0 or not os.path.exists(op):
            from rl2.rio import dumps
            return dumps({'error': f'rc={r.returncode} ' + (r.stderr or '')[-1500:], 'job': job})
        return open(op, 'rb').read()
    except subprocess.TimeoutExpired:
        from rl2.rio import dumps
        return dumps({'error': 'timeout', 'job': job})
    finally:
        for p in (jp, op):
            if os.path.exists(p):
                os.remove(p)
        os.rmdir(d)


class LocalExec:
    def __init__(self, workers):
        import multiprocessing as mp
        self.pool = mp.get_context('spawn').Pool(workers)
        self.slots = workers

    def submit(self, job, files):
        return self.pool.apply_async(run_episode, (job, files))

    def ready(self, handles, timeout=1.0):
        t = time.time() + timeout
        while True:
            done = [h for h in handles if h.ready()]
            if done or time.time() > t:
                return done
            time.sleep(0.2)

    def get(self, h):
        return h.get()


class RayExec:
    def __init__(self, reserve_head=1):
        import ray
        self.ray = ray
        if not ray.is_initialized():
            ray.init(address='auto', ignore_reinit_error=True)
        import ray.cloudpickle as cp
        cp.register_pickle_by_value(sys.modules[__name__])    # ship this module's code to the workers by value
        self.remote = ray.remote(num_cpus=1, max_retries=0)(run_episode)
        cpus = int(ray.cluster_resources().get('CPU', 1))
        self.slots = max(1, cpus - reserve_head)

    def submit(self, job, files):
        return self.remote.remote(job, files)

    def ready(self, handles, timeout=1.0):
        if not handles:
            time.sleep(timeout); return []
        done, _ = self.ray.wait(handles, num_returns=len(handles), timeout=timeout)
        return done

    def get(self, h):
        return self.ray.get(h)


def setup_ray_nodes(pack_bytes, code_bytes, setup_sh='bash /tmp/acac5/acac/node_setup.sh'):
    """Install the offline bundle + current code on every Ray node (one pinned task per node)."""
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    @ray.remote(num_cpus=0)
    def _setup(pack, code):
        import io, tarfile, socket
        os.makedirs('/tmp/acac5', exist_ok=True)
        if pack is not None and not os.path.exists('/tmp/acac5/acac/nb_driver.py'):
            tarfile.open(fileobj=io.BytesIO(pack)).extractall('/tmp/acac5')
        tarfile.open(fileobj=io.BytesIO(code)).extractall('/tmp/acac5/acac')      # current code (incl. fixed node_setup.sh)
        if not os.path.exists('/tmp/acenv/.ok'):
            r = subprocess.run(setup_sh + ' && touch /tmp/acenv/.ok', shell=True, capture_output=True, text=True)
            if r.returncode != 0:
                return socket.gethostname(), 'setup failed: ' + (r.stdout + r.stderr)[-800:]
        return socket.gethostname(), f'ok nproc={os.cpu_count()}'

    pref = ray.put(pack_bytes); cref = ray.put(code_bytes)
    nodes = [n for n in ray.nodes() if n.get('Alive')]
    hs = [_setup.options(scheduling_strategy=NodeAffinitySchedulingStrategy(n['NodeID'], soft=False)).remote(pref, cref)
          for n in nodes]
    return ray.get(hs)


# ------------------------------------------------------------------------------------------------ jobs
def league_job(league, rng, version, wname, seed, files_cache):
    nm, opp = league.sample(rng)
    files = {wname: files_cache[wname]}
    if opp.get('snapshot'):
        sn = os.path.basename(opp['snapshot'])
        if sn not in files_cache:
            files_cache[sn] = open(opp['snapshot'], 'rb').read()
        files[sn] = files_cache[sn]
        opp = {'snapshot_name': sn, 'name': nm}
    job = dict(seed=int(seed), color='Red' if rng.random() < 0.5 else 'Blue', mode='policy', weights_name=wname,
               opp=opp, version=version, tag=nm)
    return job, files


def train(ex, run, iters=100, eps=48, budget_min=60.0, opps=None, snap_every=10, hp=None, log=print,
          on_version=None, seed0=80000):
    """Asynchronous PPO. run/ must contain ckpt_XXXX.pt + latest.json (from BC)."""
    sys.path.insert(0, ROOT)
    from rl2.learner import make_net, PPO, save
    from rl2.train import League, latest
    from rl2.rio import loads
    v, wpath = latest(run)
    net = make_net(os.path.join(run, f'ckpt_{v:04d}.pt'))
    ppo = PPO(net, hp); ppo.it = v
    league = League(os.path.join(run, 'league.json'), base=opps)
    rng = np.random.default_rng(seed0 + v)
    cache = {os.path.basename(wpath): open(wpath, 'rb').read()}
    wname = os.path.basename(wpath)
    inflight, batch = [], []
    t_end = time.time() + budget_min * 60
    t_it = time.time(); n_err = 0; seed = seed0 + 100000 * (v + 1)
    v_end = v + iters
    while v < v_end and time.time() < t_end:
        while len(inflight) < ex.slots:
            job, files = league_job(league, rng, v, wname, seed, cache); seed += 1
            inflight.append(ex.submit(job, files))
        for h in ex.ready(inflight, timeout=2.0):
            inflight.remove(h)
            r = loads(ex.get(h))
            if 'error' in r:
                n_err += 1; log('ERR ' + str(r['error'])[-300:]); continue
            league.record(r['tag'], r['score'])
            if r['version'] >= v - ppo.hp['max_lag']:
                batch.append(r)
        if len(batch) >= eps:
            t1 = time.time()
            st = ppo.update([t for r in batch for t in r['trajs']], log=log)
            v += 1
            wpath = save(net, run, v); wname = os.path.basename(wpath)
            cache = {k: b for k, b in cache.items() if k.startswith('policy_') is False or k == wname}
            cache[wname] = open(wpath, 'rb').read()
            if v % snap_every == 0:
                league.add_snapshot(v, wpath)
            league.save()
            by = {}
            for r in batch:
                by.setdefault(r['tag'], []).append(r['score'])
            lag = np.mean([v - 1 - r['version'] for r in batch])
            log(f"[it {v}] score={np.mean([r['score'] for r in batch]):.3f} "
                + ' '.join(f'{k}:{np.mean(s):.2f}({len(s)})' for k, s in sorted(by.items()))
                + f" it_time={time.time() - t_it:.0f}s learn={time.time() - t1:.0f}s lag={lag:.2f} err={n_err}")
            t_it = time.time(); batch = []
            if on_version is not None:
                on_version(v, run)
    for h in inflight:              # let in-flight episodes finish? no: budget is over, drop them
        pass
    return v


def run_fixed(ex, jobs, files, log=print, on_result=None):
    """Plays a fixed job list (BC collection, evaluation). Returns results without trajectories unless kept."""
    from rl2.rio import loads
    pending = list(jobs); inflight = []; out = []
    t0 = time.time()
    while pending or inflight:
        while pending and len(inflight) < ex.slots:
            inflight.append(ex.submit(pending.pop(0), files))
        for h in ex.ready(inflight, timeout=2.0):
            inflight.remove(h)
            r = loads(ex.get(h))
            if on_result is not None:
                r = on_result(r)
            out.append(r)
            if len(out) % 20 == 0:
                log(f'{len(out)}/{len(jobs)} done, {3600 * len(out) / (time.time() - t0):.0f}/h')
    return out
