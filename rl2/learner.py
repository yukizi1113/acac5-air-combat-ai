"""Learner for the decentralised option policy (agents/TacR): behaviour cloning, then PPO (actor-critic, GAE).

Design choices and their sources are in docs/RL_DESIGN_DIGEST.md:
  - per-fighter samples, factorised action (option x altitude band x fire mode), joint log-prob per fighter
  - PPO clip 0.2, GAE lambda 0.95, gamma 0.995 per 5 s decision, 2-3 epochs, approx-KL early stop
  - value recomputed with the current critic before GAE (handles policy lag); rollouts older than 2 versions dropped
  - BC term on the rule commander's labels kept for the whole run (PPO-ptx analogue; weight decays to a floor),
    entropy bonus per head
"""
import os
import glob
import time
import json
import pickle
import gzip
import numpy as np
import torch
import torch.nn.functional as Fn

FKEYS = ('self', 'fr', 'frm', 'en', 'enm', 'ms', 'msm', 'mw', 'mwm')
HP = dict(gamma=0.995, lam=0.95, clip=0.2, epochs=3, minibatch=2048, lr=3e-4, ent=0.01, vf=0.5, max_grad=0.5,
          target_kl=0.02, bc0=0.5, bc_floor=0.05, bc_decay_iters=100, max_lag=2)


def make_net(init=None):
    import sys
    from agents.TacR.net2 import Net2, ARCH
    torch.set_num_threads(int(os.environ.get('LEARNER_THREADS', '4')))
    net = Net2(**ARCH)
    if init:
        if init.endswith('.pt'):
            net.load_state_dict(torch.load(init, map_location='cpu'))
        else:
            from agents.TacR.net2 import load_from_npz
            load_from_npz(net, init)
    return net


def save(net, out_dir, version, extra=None):
    from agents.TacR.net2 import export_npz
    os.makedirs(out_dir, exist_ok=True)
    torch.save(net.state_dict(), f'{out_dir}/ckpt_{version:04d}.pt')
    tmp = f'{out_dir}/.policy_{version:04d}.npz'
    export_npz(net, tmp, dict({'heads': 4, 'version': version}, **(extra or {})))
    os.replace(tmp, f'{out_dir}/policy_{version:04d}.npz')
    tmp2 = f'{out_dir}/.latest.json'
    json.dump({'version': version, 'path': f'policy_{version:04d}.npz', 't': time.time()}, open(tmp2, 'w'))
    os.replace(tmp2, f'{out_dir}/latest.json')
    return f'{out_dir}/policy_{version:04d}.npz'


def flatten(trajs):
    """List of per-fighter trajectories -> dict of concatenated arrays + segment boundaries."""
    cat = {k: np.concatenate([t[k] for t in trajs]).astype(np.float32) for k in FKEYS}
    for k in ('act', 'teacher'):
        cat[k] = np.concatenate([t[k] for t in trajs]).astype(np.int64)
    for k in ('logp', 'val', 'rew'):
        cat[k] = np.concatenate([t[k] for t in trajs]).astype(np.float32)
    lens = np.array([t['n'] for t in trajs])
    cat['seg'] = np.repeat(np.arange(len(trajs)), lens)
    cat['lens'] = lens
    return cat


def forward_all(net, d, idx=None, bs=8192):
    outs = [[], [], [], []]
    n = len(d['self']) if idx is None else len(idx)
    for s in range(0, n, bs):
        sl = slice(s, s + bs) if idx is None else idx[s:s + bs]
        xs = [torch.as_tensor(d[k][sl]) for k in FKEYS]
        o = net(*xs)
        for j in range(4):
            outs[j].append(o[j])
    return [torch.cat(o) for o in outs]


def gae(rew, val, lens, gamma, lam):
    adv = np.zeros_like(rew); ret = np.zeros_like(rew)
    o = 0
    for L in lens:
        r = rew[o:o + L]; v = val[o:o + L]
        last = 0.0
        for t in range(L - 1, -1, -1):
            nv = v[t + 1] if t + 1 < L else 0.0          # trajectory end = match end or death (MC tail inside r)
            delta = r[t] + gamma * nv - v[t]
            last = delta + gamma * lam * last
            adv[o + t] = last
        ret[o:o + L] = adv[o:o + L] + v
        o += L
    return adv, ret


def logp_ent(lo, la, lf, act):
    ls = [Fn.log_softmax(x, -1) for x in (lo, la, lf)]
    lp = sum(l.gather(1, act[:, j:j + 1])[:, 0] for j, l in enumerate(ls))
    ent = sum(-(l.exp() * l).sum(-1) for l in ls)
    return lp, ent, ls


def bc_loss(ls, teacher, smooth=(0.05, 0.1, 0.1)):
    tot = 0.0
    for j, l in enumerate(ls):
        n = l.shape[-1]
        tgt = Fn.one_hot(teacher[:, j], n).float() * (1 - smooth[j]) + smooth[j] / n
        tot = tot + -(tgt * l).sum(-1).mean()
    return tot


def train_bc(net, trajs, epochs=6, lr=1e-3, gamma=0.995, log=print):
    d = flatten(trajs)
    # Monte-Carlo returns to pre-train the critic
    ret = np.zeros_like(d['rew']); o = 0
    for L in d['lens']:
        g = 0.0
        for t in range(L - 1, -1, -1):
            g = d['rew'][o + t] + gamma * g; ret[o + t] = g
        o += L
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    n = len(ret); teacher = torch.as_tensor(d['teacher']); R = torch.as_tensor(ret)
    for ep in range(epochs):
        perm = np.random.permutation(n); tl = tv = acc = 0.0; nb = 0
        for s in range(0, n, 1024):
            idx = perm[s:s + 1024]
            lo, la, lf, v = net(*[torch.as_tensor(d[k][idx]) for k in FKEYS])
            ls = [Fn.log_softmax(x, -1) for x in (lo, la, lf)]
            lb = bc_loss(ls, teacher[idx]); lv = ((v - R[idx]) ** 2).mean()
            loss = lb + 0.5 * lv
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            tl += float(lb); tv += float(lv); nb += 1
            acc += float((ls[0].argmax(-1) == teacher[idx, 0]).float().mean())
        log(f'[bc] epoch {ep} bc={tl/nb:.4f} v={tv/nb:.4f} opt_acc={acc/nb:.4f} n={n}')
    return net


class PPO:
    def __init__(self, net, hp=None):
        self.net = net
        self.hp = dict(HP); self.hp.update(hp or {})
        self.opt = torch.optim.Adam(net.parameters(), lr=self.hp['lr'], eps=1e-5)
        self.it = 0

    def update(self, trajs, log=print):
        hp = self.hp
        d = flatten(trajs)
        with torch.no_grad():
            _, _, _, v = forward_all(self.net, d)
        val = v.numpy()
        adv, ret = gae(d['rew'], val, d['lens'], hp['gamma'], hp['lam'])
        advn = (adv - adv.mean()) / (adv.std() + 1e-8)
        A = torch.as_tensor(advn); R = torch.as_tensor(ret); LP0 = torch.as_tensor(d['logp'])
        act = torch.as_tensor(d['act']); teacher = torch.as_tensor(d['teacher'])
        bc_w = max(hp['bc_floor'], hp['bc0'] * (1 - self.it / max(1, hp['bc_decay_iters'])))
        n = len(ret); stats = {}
        stop = False
        for ep in range(hp['epochs']):
            perm = np.random.permutation(n)
            for s in range(0, n, hp['minibatch']):
                idx = perm[s:s + hp['minibatch']]
                lo, la, lf, vv = self.net(*[torch.as_tensor(d[k][idx]) for k in FKEYS])
                lp, ent, ls = logp_ent(lo, la, lf, act[idx])
                ratio = torch.exp(lp - LP0[idx])
                s1 = ratio * A[idx]; s2 = torch.clamp(ratio, 1 - hp['clip'], 1 + hp['clip']) * A[idx]
                lpi = -torch.min(s1, s2).mean()
                lv = ((vv - R[idx]) ** 2).mean()
                lb = bc_loss(ls, teacher[idx])
                loss = lpi + hp['vf'] * lv - hp['ent'] * ent.mean() + bc_w * lb
                self.opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), hp['max_grad']); self.opt.step()
                with torch.no_grad():
                    akl = float(((ratio - 1) - torch.log(ratio)).mean())
                    clipf = float(((ratio - 1).abs() > hp['clip']).float().mean())
                stats = dict(pi=float(lpi), v=float(lv), bc=float(lb), ent=float(ent.mean()), akl=akl, clipfrac=clipf)
                if akl > 1.5 * hp['target_kl']:
                    stop = True
                    break
            if stop:
                break
        self.it += 1
        with torch.no_grad():
            lo, la, lf, _ = forward_all(self.net, d)
            dev = float((lo.argmax(-1) != teacher[:, 0]).float().mean())
            altdev = float((la.argmax(-1) != 0).float().mean()); firedev = float((lf.argmax(-1) != 1).float().mean())
        stats.update(it=self.it, n=n, epochs_done=ep + 1, bc_w=bc_w, adv_std=float(adv.std()),
                     ret_mean=float(ret.mean()), dev_opt=dev, dev_alt=altdev, dev_fire=firedev)
        log('[ppo] ' + ' '.join(f'{k}={v:.4g}' if isinstance(v, float) else f'{k}={v}' for k, v in stats.items()))
        return stats


from rl2.rio import save_rollout, load_rollout  # noqa: E402,F401
