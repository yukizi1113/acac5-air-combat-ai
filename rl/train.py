"""Trainer for the high-level option policy.

  Stage 1  bc   : behaviour cloning of the rule-based commander (teacher labels recorded in obs)
  Stage 2  ppo  : PPO (clipped, GAE) with league self-play + PFSP opponent sampling and a decaying
                  KL-to-teacher regulariser (keeps the policy close to the rule commander early on)

Examples (run from project root):
  python rl/train.py bc  --workers 6 --episodes 120 --out runs/bc
  python rl/train.py ppo --workers 6 --init runs/bc/policy.pt --iters 300 --eps-per-iter 12 --out runs/ppo
Outputs: <out>/policy.pt (latest, submission format), <out>/ckpt_XXXX.pt, <out>/log.csv
"""
import os
import sys
import csv
import json
import time
import copy
import random
import argparse
import multiprocessing as mp
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

ARCH = dict(d=64, layers=2, heads=4, ff=128)

# --------------------------------------------------------------- opponent league
BASE_OPPONENTS = [
    {'name': 'bench', 'opp_module': 'Agents.BenchMark', 'opp_args': {'type': 'Fixed'}},
    {'name': 'bench_rand', 'opp_module': 'Agents.BenchMark', 'opp_args': {'type': 'Random'}},
    # frozen rule commander (v7) - the teacher itself
    {'name': 'rule', 'opp_module': 'agents.snap_v7', 'opp_args': {}},
    # previous generations, for diversity
    {'name': 'rule_v4', 'opp_module': 'agents.snap_v4', 'opp_args': {'overrides': {'pk_thr_fighter': 0.65}}},
    # escort rusher: ignores fighters, never drags, all four go for the escort
    {'name': 'rusher', 'opp_module': 'agents.snap_v3', 'opp_args': {'overrides': {
        'n_defenders_max': 0, 'engage_radius': 0.0, 'use_vthreat': False, 'k_fighter': -1.0, 'k_escort': 0.6}}},
    # turtle: two goalkeepers, aggressive against fighters
    {'name': 'turtle', 'opp_module': 'agents.snap_v7', 'opp_args': {'overrides': {
        'n_goalkeepers': 2, 'pk_thr_fighter': 0.45}}},
]


def make_net(state=None):
    import torch
    from agents.Tac.nn_model import EntityNet
    net = EntityNet(**ARCH)
    if state is not None:
        net.load_state_dict(state)
    return net


def save_policy(net, path, extra=None):
    import torch
    d = {'arch': ARCH, 'model': net.state_dict()}
    if extra:
        d.update(extra)
    torch.save(d, path)


def load_policy(path):
    import torch
    ck = torch.load(path, map_location='cpu', weights_only=False)
    return ck['model']


def run_jobs(pool, jobs):
    from rl.rollout import play_episode
    out = []
    for r in pool.imap_unordered(play_episode, jobs):
        out.append(r)
    return out


# --------------------------------------------------------------- batching
def flatten(results, teams_key='trajs'):
    """-> list of trajectories: each dict(obs list, a_opt, a_fire, alive, logp, v, rew)"""
    trajs = []
    for r in results:
        for tm, tr in r[teams_key].items():
            if len(tr['steps']) == 0:
                continue
            trajs.append(tr)
    return trajs


def gae(rew, val, gamma, lam):
    T = len(rew)
    adv = np.zeros(T, np.float32)
    last = 0.0
    for t in reversed(range(T)):
        nv = val[t + 1] if t + 1 < T else 0.0
        delta = rew[t] + gamma * nv - val[t]
        last = delta + gamma * lam * last
        adv[t] = last
    return adv, adv + np.asarray(val, np.float32)


def collate(steps, idx):
    import torch
    from agents.Tac.nn_model import obs_to_tensors
    obs = [steps[i]['obs'] for i in idx]
    T = obs_to_tensors(obs)
    teacher = torch.as_tensor(np.stack([o['teacher'] for o in obs]), dtype=torch.long)
    ao = torch.as_tensor(np.stack([steps[i]['a_opt'] for i in idx]), dtype=torch.long)
    af = torch.as_tensor(np.stack([steps[i]['a_fire'] for i in idx]), dtype=torch.long)
    alive = torch.as_tensor(np.stack([steps[i]['alive'] for i in idx]), dtype=torch.float32)
    return T, teacher, ao, af, alive


# --------------------------------------------------------------- BC
def cmd_bc(a):
    import torch
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(0)
    net = make_net()
    ctx = mp.get_context('spawn')
    data = []
    with ctx.Pool(a.workers, maxtasksperchild=1) as pool:
        jobs = []
        for i in range(a.episodes):
            opp = BASE_OPPONENTS[i % len(BASE_OPPONENTS)]
            jobs.append({'seed': 10000 + i, 'color': 'Red' if i % 2 == 0 else 'Blue', 'mode': 'teacher',
                         'arch': ARCH, 'opp_module': opp['opp_module'], 'opp_args': opp['opp_args'],
                         'opp_name': opp['name']})
        t0 = time.time()
        for k, r in enumerate(pool.imap_unordered(_play, jobs)):
            for tm, tr in r['trajs'].items():
                data += tr['steps']
            print(f"[bc-collect] {k+1}/{len(jobs)} opp={r['opp']} score={r['score']:.2f} steps={len(data)} {time.time()-t0:.0f}s", flush=True)
    import pickle
    pickle.dump(data, open(os.path.join(a.out, 'bc_data.pkl'), 'wb'))
    train_bc(net, data, a)
    save_policy(net, os.path.join(a.out, 'policy.pt'))
    print('saved', os.path.join(a.out, 'policy.pt'))


def train_bc(net, data, a):
    import torch
    import torch.nn.functional as Fn
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    n = len(data)
    idx_all = np.arange(n)
    rng = np.random.default_rng(0)
    rng.shuffle(idx_all)
    nval = max(1, n // 10)
    val, tr = idx_all[:nval], idx_all[nval:]
    for ep in range(a.bc_epochs):
        rng.shuffle(tr)
        net.train()
        tot = 0.0
        for s in range(0, len(tr), 256):
            T, teacher, ao, af, alive = collate(data, tr[s:s + 256])
            lo, lf, v = net(*T)
            ce = Fn.cross_entropy(lo.reshape(-1, lo.shape[-1]), ao.reshape(-1), reduction='none').reshape(ao.shape)
            cf = Fn.cross_entropy(lf.reshape(-1, lf.shape[-1]), af.reshape(-1), reduction='none').reshape(af.shape)
            loss = ((ce + 0.3 * cf) * alive).sum() / alive.sum().clamp(min=1)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            tot += loss.item() * len(ao)
        net.eval()
        with torch.no_grad():
            T, teacher, ao, af, alive = collate(data, val)
            lo, lf, v = net(*T)
            acc = (((lo.argmax(-1) == ao).float() * alive).sum() / alive.sum()).item()
        print(f'[bc] epoch {ep} loss={tot/len(tr):.4f} val_acc={acc:.3f}', flush=True)


def _play(job):
    from rl.rollout import play_episode
    try:
        return play_episode(job)
    except Exception:
        import traceback
        return {'error': traceback.format_exc(), 'trajs': {}, 'score': 0.0, 'opp': job.get('opp_name'), 'winner': None,
                'time': 0, 'wall': 0, 'hits': 0, 'launched': 0, 'color': job['color']}


# --------------------------------------------------------------- PPO
class League:
    def __init__(self, base, max_snaps=8, p=2.0):
        self.base = [dict(o) for o in base]
        self.snaps = []        # dicts with name, opp_net_state
        self.stats = {}        # name -> [score_sum, n]
        self.max_snaps = max_snaps
        self.p = p

    def all(self):
        return self.base + self.snaps

    def add_snapshot(self, name, state):
        self.snaps.append({'name': name, 'opp_net_state': copy.deepcopy(state)})
        if len(self.snaps) > self.max_snaps:
            self.snaps.pop(0)

    def mean_score(self, name):
        s, n = self.stats.get(name, (0.0, 0))
        return s / n if n else 0.5

    def sample(self, rng):
        opps = self.all()
        # PFSP: weight ~ (1 - our mean score vs opp)^p, floor so nobody is forgotten
        w = np.array([max(0.05, 1.0 - self.mean_score(o['name'])) ** self.p for o in opps])
        w = w / w.sum()
        return opps[rng.choice(len(opps), p=w)]

    def update(self, name, score, decay=0.98):
        s, n = self.stats.get(name, (0.0, 0))
        self.stats[name] = (s * decay + score, n * decay + 1)


def cmd_ppo(a):
    import torch
    import torch.nn.functional as Fn
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    net = make_net(load_policy(a.init) if a.init else None)
    teacher_net = None
    opt = torch.optim.Adam(net.parameters(), lr=a.lr, eps=1e-5)
    league = League(BASE_OPPONENTS, max_snaps=a.max_snaps)
    logf = open(os.path.join(a.out, 'log.csv'), 'a', newline='')
    lw = csv.writer(logf)
    if os.path.getsize(os.path.join(a.out, 'log.csv')) == 0:
        lw.writerow(['iter', 'time', 'episodes', 'steps', 'mean_score', 'win_rate', 'hit_rate', 'pi_loss', 'v_loss',
                     'entropy', 'kl_teacher', 'approx_kl', 'clipfrac', 'beta', 'per_opp'])
    ctx = mp.get_context('spawn')
    seed_ctr = a.seed * 100000
    t_start = time.time()
    with ctx.Pool(a.workers, maxtasksperchild=a.max_tasks) as pool:
        for it in range(a.start_iter, a.start_iter + a.iters):
            beta = a.kl_teacher * max(0.0, 1.0 - it / max(1, a.kl_decay_iters))
            state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            jobs = []
            for e in range(a.eps_per_iter):
                o = league.sample(rng)
                seed_ctr += 1
                j = {'seed': int(seed_ctr), 'color': 'Red' if seed_ctr % 2 else 'Blue', 'arch': ARCH,
                     'net_state': state, 'opp_name': o['name'], 'cfg': json.loads(a.reward_cfg)}
                if 'opp_net_state' in o:
                    j['opp_net_state'] = o['opp_net_state']; j['record_opp'] = False
                else:
                    j['opp_module'] = o['opp_module']; j['opp_args'] = o['opp_args']
                jobs.append(j)
            results = list(pool.imap_unordered(_play, jobs))
            errs = [r for r in results if 'error' in r]
            for r in errs:
                print('[worker error]', r['error'][-1500:], flush=True)
            results = [r for r in results if 'error' not in r]
            per_opp = {}
            for r in results:
                league.update(r['opp'], r['score'])
                per_opp.setdefault(r['opp'], []).append(r['score'])
            # ---------- build training batch
            steps = []
            for r in results:
                for tm, tr in r['trajs'].items():
                    st = tr['steps']; rw = tr['rewards']
                    if len(st) == 0:
                        continue
                    val = [s['v'] for s in st]
                    adv, ret = gae(rw, val, a.gamma, a.lam)
                    for s, ad, rt in zip(st, adv, ret):
                        s['adv'] = float(ad); s['ret'] = float(rt)
                    steps += st
            if len(steps) < 16:
                continue
            advs = np.array([s['adv'] for s in steps], np.float32)
            amean, astd = advs.mean(), advs.std() + 1e-8
            stats = {'pi': [], 'v': [], 'ent': [], 'klt': [], 'akl': [], 'cf': []}
            net.train()
            idx = np.arange(len(steps))
            for ep in range(a.epochs):
                rng.shuffle(idx)
                for s0 in range(0, len(idx), a.minibatch):
                    mb = idx[s0:s0 + a.minibatch]
                    T, teacher, ao, af, alive = collate(steps, mb)
                    old_logp = torch.as_tensor([steps[i]['logp'] for i in mb], dtype=torch.float32)
                    old_v = torch.as_tensor([steps[i]['v'] for i in mb], dtype=torch.float32)
                    adv = (torch.as_tensor([steps[i]['adv'] for i in mb], dtype=torch.float32) - amean) / astd
                    ret = torch.as_tensor([steps[i]['ret'] for i in mb], dtype=torch.float32)
                    lo, lf, v = net(*T)
                    do = torch.distributions.Categorical(logits=lo)
                    df = torch.distributions.Categorical(logits=lf)
                    logp = ((do.log_prob(ao) + df.log_prob(af)) * alive).sum(-1)
                    ratio = torch.exp(logp - old_logp)
                    s1 = ratio * adv; s2 = torch.clamp(ratio, 1 - a.clip, 1 + a.clip) * adv
                    pi_loss = -torch.min(s1, s2).mean()
                    vc = old_v + torch.clamp(v - old_v, -a.vclip, a.vclip)
                    v_loss = 0.5 * torch.max((v - ret) ** 2, (vc - ret) ** 2).mean()
                    ent = ((do.entropy() + df.entropy()) * alive).sum(-1).mean() / 4.0
                    nF = ao.shape[1]
                    klt = (Fn.cross_entropy(lo.reshape(-1, lo.shape[-1]), teacher[:, :nF].reshape(-1), reduction='none').reshape(ao.shape) * alive).sum(-1).mean() / 4.0
                    loss = pi_loss + a.vcoef * v_loss - a.ent * ent + beta * klt
                    opt.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), a.max_grad)
                    opt.step()
                    with torch.no_grad():
                        stats['pi'].append(pi_loss.item()); stats['v'].append(v_loss.item())
                        stats['ent'].append(ent.item()); stats['klt'].append(klt.item())
                        stats['akl'].append((old_logp - logp).mean().item())
                        stats['cf'].append(((ratio - 1).abs() > a.clip).float().mean().item())
            net.eval()
            scores = [r['score'] for r in results]
            wins = [1.0 if r['winner'] == r['color'] else 0.0 for r in results]
            hr = [r['hits'] / r['launched'] for r in results if r['launched'] > 0]
            po = {k: round(float(np.mean(v)), 3) for k, v in per_opp.items()}
            row = [it, round(time.time() - t_start), len(results), len(steps), round(float(np.mean(scores)), 4),
                   round(float(np.mean(wins)), 3), round(float(np.mean(hr)) if hr else 0.0, 3)] + \
                  [round(float(np.mean(stats[k])), 4) for k in ('pi', 'v', 'ent', 'klt', 'akl', 'cf')] + [round(beta, 4), json.dumps(po)]
            lw.writerow(row); logf.flush()
            print(f"[ppo] it={it} score={row[4]:.3f} win={row[5]:.2f} hit={row[6]:.2f} steps={len(steps)} ent={row[9]:.3f} "
                  f"klT={row[10]:.3f} akl={row[11]:.4f} beta={beta:.3f} opp={po} t={row[1]}s", flush=True)
            save_policy(net, os.path.join(a.out, 'policy.pt'), {'iter': it})
            if (it + 1) % a.snap_every == 0:
                save_policy(net, os.path.join(a.out, f'ckpt_{it+1:04d}.pt'), {'iter': it})
                league.add_snapshot(f'snap{it+1}', net.state_dict())
            json.dump({k: list(v) for k, v in league.stats.items()}, open(os.path.join(a.out, 'league.json'), 'w'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['bc', 'ppo'])
    ap.add_argument('--workers', type=int, default=max(1, mp.cpu_count() - 1))
    ap.add_argument('--out', default='runs/exp')
    ap.add_argument('--seed', type=int, default=1)
    # bc
    ap.add_argument('--episodes', type=int, default=60)
    ap.add_argument('--bc-epochs', type=int, default=30)
    # ppo
    ap.add_argument('--init', default='')
    ap.add_argument('--iters', type=int, default=100)
    ap.add_argument('--start-iter', type=int, default=0)
    ap.add_argument('--eps-per-iter', type=int, default=8)
    ap.add_argument('--epochs', type=int, default=4)
    ap.add_argument('--minibatch', type=int, default=256)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--gamma', type=float, default=0.995)
    ap.add_argument('--lam', type=float, default=0.95)
    ap.add_argument('--clip', type=float, default=0.2)
    ap.add_argument('--vclip', type=float, default=0.5)
    ap.add_argument('--vcoef', type=float, default=0.5)
    ap.add_argument('--ent', type=float, default=0.01)
    ap.add_argument('--max-grad', type=float, default=0.5)
    ap.add_argument('--kl-teacher', type=float, default=0.3)
    ap.add_argument('--kl-decay-iters', type=int, default=150)
    ap.add_argument('--snap-every', type=int, default=20)
    ap.add_argument('--max-snaps', type=int, default=6)
    ap.add_argument('--max-tasks', type=int, default=1)
    ap.add_argument('--reward-cfg', default='{}')
    a = ap.parse_args()
    if a.cmd == 'bc':
        cmd_bc(a)
    else:
        cmd_ppo(a)


if __name__ == '__main__':
    main()
