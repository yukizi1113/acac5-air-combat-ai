"""One training episode = one full match, run in a fresh process (the simulator allows one match per process).

Our side: agents/TacR in rl2 mode.
  mode 'policy' : NN2Policy samples (option, altitude band, fire mode) per fighter on decision ticks and records
                  (ego obs, action, log-prob, value).
  mode 'teacher': the rule commander (v17) acts; observations + its labels are recorded (behaviour cloning data).
Opponent: a rule agent from the league, a frozen policy snapshot, or the current policy (mirror; also recorded).

Rewards (training only, never seen by the agent at decision time): computed afterwards from the match's event log.
  per decision interval (team-shared): +w_kill per enemy fighter destroyed, -w_loss per own fighter lost,
                                       +w_esc / -w_esc when an escort is destroyed
  terminal: w_term * 2 * (competition score - 0.5)  (exact score of the match)  -  w_miss * missed missiles
A fighter that dies keeps the credit for what happens afterwards: its last transition receives the discounted
sum of the team rewards that follow (Monte-Carlo tail), then the trajectory terminates.
"""
import os
import sys
import time
import io
import contextlib
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
REWARD = dict(w_kill=0.1, w_loss=0.1, w_esc=0.0, w_term=1.0, w_miss=0.03, gamma=0.995)   # gamma = learner HP['gamma']
FKEYS = ('self', 'fr', 'frm', 'en', 'enm', 'ms', 'msm', 'mw', 'mwm')


class TeacherRec:
    def __init__(self):
        self.record = []

    def reset(self):
        pass

    def step(self, observation, reward, done, info, agentFullName, observation_space, action_space):
        if isinstance(observation, dict) and 'alive' in observation and observation['decide'][0] > 0.5:
            t = observation['teacher']
            a = np.concatenate([t[:4], t[4:8], t[8:12]]).astype(np.int64)
            self.record.append((observation, a, np.zeros(4), np.zeros(4)))
        return None


def _team_of(name):
    return name.split('/')[0] if '/' in name else ''


def _step_rewards(times, events, team, cfg, t_end):
    """Team reward for each decision interval (t_k, t_{k+1}] (last interval up to the end of the match)."""
    opp = 'Blue' if team == 'Red' else 'Red'
    r = np.zeros(len(times), np.float64)
    dead = set()
    for (t, kind, wpn, tgt) in events:
        if tgt in dead or not tgt:
            continue
        tt = _team_of(tgt)
        if tt not in ('Red', 'Blue'):
            continue
        dead.add(tgt)
        esc = tgt.endswith('E')
        if tt == opp:
            v = cfg['w_esc'] if esc else cfg['w_kill']
        else:
            v = -(cfg['w_esc'] if esc else cfg['w_loss'])
        k = int(np.searchsorted(times, t, side='left')) - 1
        r[max(0, min(len(times) - 1, k))] += v
    return r


def build_trajectories(rec, team, detail, cfg):
    """rec: list of (obs dict with fighter axis, action[12], logp[4], value[4]) on decision ticks."""
    if not rec:
        return []
    times = np.array([float(o['self'][o['alive'] > 0.5][0, 12] * 1200.0) if (o['alive'] > 0.5).any() else 0.0
                      for o, *_ in rec])
    rew = _step_rewards(times, detail.get('events', []), team, cfg, detail.get('spentTime', 0.0))
    score = float(detail['scores'][team])
    launched = detail['launched'].get(team, 0); hits = detail['hits'].get(team, 0)
    rew[-1] += cfg['w_term'] * 2.0 * (score - 0.5) - cfg['w_miss'] * max(0, launched - hits)
    g = cfg['gamma']
    # Monte-Carlo tails for dead fighters
    tail = np.zeros(len(rew) + 1)
    for k in range(len(rew) - 1, -1, -1):
        tail[k] = rew[k] + g * tail[k + 1]
    out = []
    for i in range(4):
        steps = [k for k, (o, *_r) in enumerate(rec) if o['alive'][i] > 0.5]
        if not steps:
            continue
        n = len(steps)
        tr = {k: np.stack([rec[s][0][k][i] for s in steps]).astype(np.float16) for k in FKEYS}
        acts = np.stack([[rec[s][1][i], rec[s][1][4 + i], rec[s][1][8 + i]] for s in steps]).astype(np.int8)
        teach = np.stack([[rec[s][0]['teacher'][i], 0, 1] for s in steps]).astype(np.int8)
        logp = np.array([rec[s][2][i] for s in steps], np.float32)
        val = np.array([rec[s][3][i] for s in steps], np.float32)
        r = np.array([rew[s] for s in steps], np.float64)
        last = steps[-1]
        if last < len(rec) - 1:                 # died: credit for the rest of the match
            r[-1] = tail[last]
        # consecutive decisions only (a fighter never revives), done at the end
        tr.update(act=acts, teacher=teach, logp=logp, val=val, rew=r.astype(np.float32), fighter=i, n=n)
        out.append(tr)
    return out


def play(job):
    """job: dict(seed, color, opp: {'module','args'} or {'snapshot': path} or {'mirror': True},
                 mode 'policy'|'teacher', weights, me_overrides, cfg, version, tag)"""
    os.chdir(ROOT)
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from tools.utils import make_match_info
    from tools.fight_ext import Fight
    import agents.TacR as R
    from rl2.common import V17
    cfg = dict(REWARD); cfg.update(job.get('cfg', {}))
    seed, color = int(job['seed']), job['color']
    np.random.seed(seed)
    me_ov = dict(V17); me_ov.update(job.get('me_overrides', {}))
    if job.get('mode') == 'teacher':
        me_pol = TeacherRec()
    else:
        me_pol = R.NN2Policy(job['weights'], greedy=job.get('greedy', False), seed=seed * 2 + (color == 'Red'))
        me_pol.record = []
    R.POLICY_REGISTRY['me'] = me_pol
    mine = {'userModelID': '0', 'userModuleID': 'agents.TacR',
            'args': {'overrides': me_ov, 'rl2': True, 'policy_slot': 'me'}}
    opp = job['opp']
    opp_pol = None
    if opp.get('mirror'):
        opp_pol = R.NN2Policy(job['weights'], greedy=False, seed=seed * 2 + (color != 'Red')); opp_pol.record = []
        R.POLICY_REGISTRY['opp'] = opp_pol
        theirs = {'userModelID': '1', 'userModuleID': 'agents.TacR',
                  'args': {'overrides': dict(V17), 'rl2': True, 'policy_slot': 'opp'}}
    elif opp.get('snapshot'):
        theirs = {'userModelID': '1', 'userModuleID': 'agents.TacR',
                  'args': {'overrides': dict(V17), 'policy': 'nn2', 'weights': opp['snapshot'], 'greedy': False}}
    else:
        theirs = {'userModelID': '1', 'userModuleID': opp['module'].replace('/', '.'), 'args': opp.get('args', {})}
    red, blue = (mine, theirs) if color == 'Red' else (theirs, mine)
    mi = make_match_info(red=red, blue=blue, replay=False, movie_dir='', time_out=1e9, common_dir='common',
                         control_init='random', youth=False, make_log=False, visualize=False)
    mi['seed'] = seed
    t0 = time.time()
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            f = Fight(mi)
            f.make_matchInfo(); f.set_match()
            winner, d = f.run(False)
    except Exception:
        import traceback
        return {'error': traceback.format_exc()[-2000:], 'job': {k: v for k, v in job.items() if k != 'weights'}}
    finally:
        R.POLICY_REGISTRY.clear()
    if d.get('error'):
        return {'error': (d.get('red_error_message', '') + d.get('blue_error_message', ''))[-2000:],
                'job': {k: v for k, v in job.items() if k != 'weights'}}
    ocolor = 'Blue' if color == 'Red' else 'Red'
    res = {'seed': seed, 'color': color, 'tag': job.get('tag', ''), 'opp_name': opp.get('name', ''),
           'version': job.get('version', 0), 'mode': job.get('mode', 'policy'),
           'score': float(d['scores'][color]), 'winner': d.get('winner'), 'time': d['spentTime'],
           'hits': d['hits'].get(color, 0), 'launched': d['launched'].get(color, 0), 'wall': time.time() - t0,
           'trajs': build_trajectories(me_pol.record, color, d, cfg)}
    if opp_pol is not None:
        res['trajs'] += build_trajectories(opp_pol.record, ocolor, d, cfg)
    return res
