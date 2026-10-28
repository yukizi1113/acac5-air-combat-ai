"""Episode rollout for training (runs inside a worker process).

One episode = one full match. Our side is agents.Tac in RL mode with a TrainPolicy that samples
options from the current network and records SMDP transitions on decision ticks. The opponent is
any registered module (benchmark, rule variants, frozen snapshots of the network).

Reward (per decision step, computed from god-view events; training-time only)
  r = w_hit*hits_on_fighters - w_miss*missed_missiles - w_loss*own_fighters_lost + w_kill*enemy_fighters_killed
      + (gamma*Phi(s') - Phi(s))                                    potential-based shaping
  terminal: + w_term * 2*(match_score - 0.5)                        exact competition score in [-1, 1]
  Phi(s) = -c1 * d(own armed fighters, enemy escort)/250km + c2 * d(enemy fighters, own escort)/250km
"""
import os
import sys
import time
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

REWARD_CFG = dict(w_hit=0.10, w_miss=0.06, w_loss=0.15, w_kill=0.10, w_term=1.0, c1=0.15, c2=0.10, gamma=0.995,
                  w_escort_hit=0.0)


class TrainPolicy:
    """Samples actions from the network on decision ticks and records transitions."""
    def __init__(self, net, sample=True, record=True, temperature=1.0):
        self.net = net
        self.sample = sample
        self.record = record
        self.temp = temperature
        self.steps = []          # dicts: obs, a_opt, a_fire, logp, v, alive, teacher, t
        self.last = None

    def reset(self):
        self.steps = []
        self.last = None

    def step(self, observation, reward, done, info, agentFullName, observation_space, action_space):
        import torch
        sys.path.insert(0, ROOT)
        from agents.Tac.nn_model import obs_to_tensors
        if not isinstance(observation, dict):
            return None
        if observation['decide'][0] < 0.5 and self.last is not None:
            return self.last
        with torch.no_grad():
            lo, lf, v = self.net(*obs_to_tensors([observation]))
            lo = lo[0] / self.temp; lf = lf[0] / self.temp
            do = torch.distributions.Categorical(logits=lo)
            df = torch.distributions.Categorical(logits=lf)
            if self.sample:
                ao, af = do.sample(), df.sample()
            else:
                ao, af = lo.argmax(-1), lf.argmax(-1)
            alive = torch.as_tensor(observation['fmask'])
            logp = ((do.log_prob(ao) + df.log_prob(af)) * alive).sum().item()
        a = np.concatenate([ao.numpy(), af.numpy()]).astype(np.int64)
        if self.record:
            self.steps.append({'obs': {k: np.array(vv, copy=True) for k, vv in observation.items()},
                               'a_opt': ao.numpy().copy(), 'a_fire': af.numpy().copy(), 'logp': logp,
                               'v': float(v[0]), 'alive': observation['fmask'].copy()})
        self.last = a
        return a


class TeacherRecorder:
    """Lets the rule commander act (returns None) and records observations + teacher labels for BC."""
    def __init__(self):
        self.steps = []
        self.record = True

    def reset(self):
        self.steps = []

    def step(self, observation, reward, done, info, agentFullName, observation_space, action_space):
        if isinstance(observation, dict) and observation['decide'][0] > 0.5:
            n = len(observation['fmask'])
            t = observation['teacher']
            self.steps.append({'obs': {k: np.array(vv, copy=True) for k, vv in observation.items()},
                               'a_opt': t[:n].astype(np.int64), 'a_fire': t[n:2 * n].astype(np.int64),
                               'logp': 0.0, 'v': 0.0, 'alive': observation['fmask'].copy()})
        return None


class GodTracker:
    """Training-only bookkeeping from the simulator's ground truth."""
    def __init__(self, manager, team):
        from ASRCAISim1.core import Fighter
        self.m = manager
        self.team = team
        self.hits = []      # (t, wpn_team, tgt_name)
        self.Fighter = Fighter
        self.msl_state = {}  # name -> alive
        self.missed = {'Red': 0, 'Blue': 0}

    def on_hit(self, args):
        from ASRCAISim1.core import Entity
        wpn = Entity.from_json(args['wpn']); tgt = Entity.from_json(args['tgt'])
        self.hits.append((self.m.getElapsedTime(), wpn.getTeam(), tgt.getFullName(), wpn.getFullName()))

    def snapshot(self):
        """Counts and positions used for rewards/potential."""
        from ASRCAISim1.core import Missile
        s = {'alive': {}, 'pos': {}, 'escort': {}}
        for wa in self.m.getAssets(lambda a: isinstance(a, self.Fighter)):
            a = wa()
            nm = a.getFullName()
            s['alive'][nm] = a.isAlive()
            if a.isAlive():
                s['pos'][nm] = (np.array(a.motion.pos()), int(a.remMsls) if hasattr(a, 'remMsls') else 1)
        hit_names = set(h[3] for h in self.hits)
        for wm in self.m.getAssets(lambda a: isinstance(a, Missile)):
            mm = wm()
            if not mm.hasLaunched:
                continue
            nm = mm.getFullName()
            prev = self.msl_state.get(nm, True)
            al = mm.isAlive()
            if prev and not al and nm not in hit_names:
                self.missed[mm.getTeam()] += 1
            self.msl_state[nm] = al
        return s


def potential(snap, team, cfg):
    opp = 'Blue' if team == 'Red' else 'Red'
    mine = [(p, r) for n, (p, r) in snap['pos'].items() if n.startswith(team + '/') and not n.endswith('E')]
    ee = [p for n, (p, r) in snap['pos'].items() if n == f'{opp}/{opp}E']
    oe = [p for n, (p, r) in snap['pos'].items() if n == f'{team}/{team}E']
    theirs = [p for n, (p, r) in snap['pos'].items() if n.startswith(opp + '/') and not n.endswith('E')]
    phi = 0.0
    armed = [p for p, r in mine if r > 0]
    if ee and armed:
        d = min(np.linalg.norm((p - ee[0])[:2]) for p in armed)
        phi -= cfg['c1'] * min(1.0, d / 250000.0)
    if oe and theirs:
        d = min(np.linalg.norm((p - oe[0])[:2]) for p in theirs)
        phi += cfg['c2'] * min(1.0, d / 250000.0)
    return phi


def match_score(winner, end_reason, team, hits, launched, t):
    opp = 'Blue' if team == 'Red' else 'Red'
    if winner not in ('Red', 'Blue'):
        return 0.5
    h = hits.get(winner, 0) / launched[winner] if launched.get(winner, 0) > 0 else 0.0
    ws = 0.6 + 0.3 * h + 0.1 * (1.0 if t < 900 else max(0.0, 2.0 - t / 900.0))
    return ws if winner == team else 1.0 - ws


def play_episode(job):
    """job: dict(seed, color, me_args, opp_module, opp_args, net_state, arch, opp_net_state, cfg, record_opp)"""
    os.chdir(ROOT)
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    import io, contextlib, torch
    torch.set_num_threads(1)
    from tools.utils import make_match_info
    from tools.fight_ext import Fight
    import agents.Tac as TacPkg
    from agents.Tac.nn_model import EntityNet
    cfg = dict(REWARD_CFG); cfg.update(job.get('cfg', {}))
    np.random.seed(job['seed'])
    torch.manual_seed(job['seed'])
    net = EntityNet(**job.get('arch', {}))
    if job.get('net_state') is not None:
        net.load_state_dict(job['net_state'])
    net.eval()
    if job.get('mode') == 'teacher':
        me_pol = TeacherRecorder()
    else:
        me_pol = TrainPolicy(net, sample=job.get('sample', True), record=True, temperature=job.get('temperature', 1.0))
    TacPkg.POLICY_REGISTRY['me'] = me_pol
    me_args = dict(job.get('me_args', {})); me_args['policy_slot'] = 'me'
    mine = {'userModelID': '0', 'userModuleID': 'agents.Tac', 'args': me_args}
    opp_pol = None
    if job.get('opp_net_state') is not None:
        onet = EntityNet(**job.get('arch', {}))
        onet.load_state_dict(job['opp_net_state']); onet.eval()
        opp_pol = TrainPolicy(onet, sample=True, record=bool(job.get('record_opp')))
        TacPkg.POLICY_REGISTRY['opp'] = opp_pol
        oargs = dict(job.get('opp_args', {})); oargs['policy_slot'] = 'opp'
        theirs = {'userModelID': '1', 'userModuleID': 'agents.Tac', 'args': oargs}
    else:
        theirs = {'userModelID': '1', 'userModuleID': job['opp_module'], 'args': job.get('opp_args', {})}
    color = job['color']
    red, blue = (mine, theirs) if color == 'Red' else (theirs, mine)
    mi = make_match_info(red=red, blue=blue, replay=False, movie_dir='', time_out=1e9, common_dir='common',
                         control_init='random', youth=False, make_log=False, visualize=False)
    mi['seed'] = job['seed']
    buf = io.StringIO()
    t0 = time.time()
    with contextlib.redirect_stdout(buf):
        fight = Fight(mi)
        fight.make_matchInfo(); fight.set_match()
        env = fight.env
        policies = {'Policy_' + info['userModelID']: fight.importedUserModules[info['userModuleID']].getUserPolicy(info['args'])
                    for team, info in fight.match_config['teams'].items()}
        obs, info = env.reset()
        for p in policies.values():
            p.reset()
        gt = GodTracker(env.manager, color)
        env.manager.getRuler()().manager.addEventHandler('Hit', gt.on_hit)
        from ASRCAISim1.core import Fighter
        init_msls = {'Red': 0, 'Blue': 0}
        for wf in env.manager.getAssets(lambda a: isinstance(a, Fighter)):
            f = wf(); init_msls[f.getTeam()] += f.numMsls
        teams = [color] + ([('Blue' if color == 'Red' else 'Red')] if opp_pol is not None and opp_pol.record else [])
        tp = {color: me_pol}
        if opp_pol is not None and opp_pol.record:
            tp[teams[1]] = opp_pol
        # per-team reward accumulators between decision steps
        acc = {tm: 0.0 for tm in teams}
        rewards = {tm: [] for tm in teams}
        nsteps = {tm: 0 for tm in teams}
        snap = gt.snapshot()
        phi = {tm: potential(snap, tm, cfg) for tm in teams}
        prev_alive = dict(snap['alive'])
        prev_hits = 0; prev_missed = dict(gt.missed)
        term = {'__all__': False}; trunc = {'__all__': False}
        rw = {k: 0.0 for k in obs}; dn = {k: False for k in obs}; inf = {k: None for k in obs}
        while not (term['__all__'] or trunc['__all__']):
            obs_sp = env.get_observation_space(); act_sp = env.get_action_space()
            actions = {}
            for k, o in obs.items():
                pol = policies[k.split(':')[2]]
                actions[k] = pol.step(o, rw[k], dn[k], inf[k], k, obs_sp[k], act_sp[k])
            obs, rw, term, trunc, inf = env.step(actions)
            # 1) close transitions whose successor decision was taken on this tick (state = last snapshot)
            for tm in teams:
                pol = tp[tm]
                while len(pol.steps) > nsteps[tm]:
                    if nsteps[tm] > 0:
                        ph = potential(snap, tm, cfg)
                        rewards[tm].append(acc[tm] + cfg['gamma'] * ph - phi[tm])
                        phi[tm] = ph; acc[tm] = 0.0
                    else:
                        phi[tm] = potential(snap, tm, cfg)
                    nsteps[tm] += 1
            # 2) rewards produced by this tick belong to the currently active decision
            snap = gt.snapshot()
            new_hits = gt.hits[prev_hits:]; prev_hits = len(gt.hits)
            for tm in teams:
                opp = 'Blue' if tm == 'Red' else 'Red'
                r = 0.0
                for (_, wteam, tname, _) in new_hits:
                    if wteam == tm and not tname.endswith(f'{opp}E'):
                        r += cfg['w_hit']
                    if wteam == tm and tname.endswith(f'{opp}E'):
                        r += cfg['w_escort_hit']
                r -= cfg['w_miss'] * (gt.missed[tm] - prev_missed[tm])
                for nm, al in snap['alive'].items():
                    if prev_alive.get(nm, False) and not al:
                        if nm.startswith(tm + '/') and not nm.endswith('E'):
                            r -= cfg['w_loss']
                        elif nm.startswith(opp + '/') and not nm.endswith('E'):
                            r += cfg['w_kill']
                acc[tm] += r
            prev_alive = dict(snap['alive']); prev_missed = dict(gt.missed)
        ruler = env.manager.getRuler()()
        winner = ruler.winner
        endReason = ruler.getEndReason()
        T = env.manager.getElapsedTime()
        rem = {'Red': 0, 'Blue': 0}
        for wf in env.manager.getAssets(lambda a: isinstance(a, Fighter)):
            f = wf(); rem[f.getTeam()] += f.remMsls
        launched = {k: init_msls[k] - rem[k] for k in rem}
        hits = {'Red': sum(1 for h in gt.hits if h[1] == 'Red'), 'Blue': sum(1 for h in gt.hits if h[1] == 'Blue')}
    out = {'seed': job['seed'], 'color': color, 'opp': job.get('opp_name', job.get('opp_module')), 'winner': winner,
           'endReason': str(endReason), 'time': T, 'wall': time.time() - t0, 'trajs': {}}
    for tm in teams:
        pol = tp[tm]
        n = len(pol.steps)
        rs = rewards[tm] + [acc[tm]]  # last open transition gets remaining accumulated reward
        sc = match_score(winner, endReason, tm, hits, launched, T)
        rs[-1] += cfg['w_term'] * 2.0 * (sc - 0.5)
        rs = rs[:n]
        out['trajs'][tm] = {'steps': pol.steps, 'rewards': np.array(rs, np.float32), 'score': sc,
                            'hits': hits[tm], 'launched': launched[tm]}
    out['score'] = out['trajs'][color]['score']
    out['hits'] = hits[color]; out['launched'] = launched[color]
    TacPkg.POLICY_REGISTRY.clear()
    return out
