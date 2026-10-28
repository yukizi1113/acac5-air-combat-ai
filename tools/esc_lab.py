"""Escort-strike lab: one Red shooter vs the (scripted, MWS-evading) Blue escort.

  python tools/esc_lab.py --episodes 200 --workers 2 --out results/esclab1.jsonl
Randomises shooter altitude (10-16.5 km), speed (fraction of the envelope's vMax), launch range (45-140 km),
bearing relative to the escort's track, and post-launch support behaviour.  All other aircraft are passive and
far away.  Output: one line per episode with launch geometry (from the god-view log), hit, time of flight.
"""
import os, sys, json, time, math, argparse, multiprocessing as mp, traceback
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

ALT = np.array([0., 1520., 3040., 4560., 6079., 7599., 9119., 10639., 12159., 13679., 14819., 15958., 17098.])
VMAX = np.array([411., 432., 456., 478., 499., 525., 547., 565., 582., 584., 564., 520., 268.])


def scenario(rng, a):
    esc_N = float(rng.uniform(-95000, 95000))
    esc_h = float(rng.choice([0.0, 180.0]))
    # the Blue escort flies its capsule clockwise: southbound on the y=115 km leg, northbound on y=125 km
    esc_y = 115000.0 if esc_h == 180.0 else 125000.0
    esc = {'pos': [esc_N, esc_y, -10000.0], 'vel': 250.0, 'heading': esc_h}
    alt = float(rng.uniform(a.alt_lo, a.alt_hi))
    vmax = float(np.interp(alt, ALT, VMAX))
    V = vmax * float(rng.uniform(a.vfrac_lo, 1.0))
    D = float(rng.uniform(a.d_lo, a.d_hi))
    # bearing from the escort to the shooter, measured in the world frame; Red side is y < 120 km
    brg = math.radians(float(rng.uniform(-180, 0)) if a.west_only else float(rng.uniform(-180, 180)))
    p = np.array([esc_N + D * math.cos(brg), esc_y + D * math.sin(brg), -alt])
    hdg = math.degrees(math.atan2(esc_y - p[1], esc_N - p[0])) % 360
    S = str(rng.choice(a.supports.split(',')))
    sh0 = {'alt': alt, 'S': S, 'E_tgt': 'escort', 't_fire': 0.0, 'tgt_hint': esc['pos'],
           'crank_deg': float(rng.uniform(30, 55))}
    passive_r = [{'passive': True, 'alt': 10000.0, 'heading_abs': 90.0} for _ in range(3)]
    passive_b = [{'passive': True, 'alt': 10000.0, 'heading_abs': 270.0} for _ in range(4)]
    init_r = [{'pos': p.tolist(), 'vel': V, 'heading': hdg}] + \
             [{'pos': [-30000.0 + 30000 * k, -130000.0, -10000.0], 'vel': 200.0, 'heading': 90.0} for k in range(3)]
    init_b = [{'pos': [-45000.0 + 30000 * k, 140000.0, -10000.0], 'vel': 200.0, 'heading': 270.0} for k in range(4)]
    if a.helper:
        # a passive team-mate that keeps the escort inside its radar coverage (tests datalink guidance)
        hb = brg + math.radians(float(rng.uniform(-40, 40)))
        hD = float(rng.uniform(60000, 90000))
        hp = [esc_N + hD * math.cos(hb), esc_y + hD * math.sin(hb), -11000.0]
        hh = math.degrees(math.atan2(esc_y - hp[1], esc_N - hp[0])) % 360
        init_r[1] = {'pos': hp, 'vel': 250.0, 'heading': hh}
        sh0_pass = {'passive': True, 'alt': 11000.0, 'heading_abs': hh}
        passive_r[0] = sh0_pass
    meta = {'helper': int(a.helper), 'alt': alt, 'V': V, 'D0': D, 'brg': math.degrees(brg), 'S': S, 'esc_h': esc_h, 'esc_N': esc_N,
            'crank_deg': sh0['crank_deg']}
    return [sh0] + passive_r, passive_b, init_r, init_b, esc, meta


def run(job):
    os.chdir(ROOT); sys.path.insert(0, ROOT)
    idx, seed, a = job
    os.environ['MATCH_ID'] = f"esclab_{seed}"
    import io, contextlib
    from tools.utils import make_match_info
    from tools.fight_ext import Fight, _god_step, _god_finish
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    sh, tg, init_r, init_b, esc, meta = scenario(rng, a)
    red = {'userModelID': '0', 'userModuleID': 'agents.Lab', 'args': {'overrides': {'lab_role': 'shooter', 'lab_params': sh}}}
    blue = {'userModelID': '1', 'userModuleID': 'agents.Lab', 'args': {'overrides': {'lab_role': 'target', 'lab_params': tg}}}
    mi = make_match_info(red=red, blue=blue, replay=False, movie_dir='', time_out=1e9, common_dir='common',
                         control_init='random', youth=False, make_log=False, visualize=False)
    mi['seed'] = seed
    t0 = time.time()
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            f = Fight(mi); f.make_matchInfo()
            for k in range(4):
                f.matchInfo['teams']['Red']['InitialState'][k] = init_r[k]
                f.matchInfo['teams']['Blue']['InitialState'][k] = init_b[k]
            f.matchInfo['teams']['Blue']['InitialState'][4] = esc
            f.matchInfo['teams']['Red']['InitialState'][4] = {'pos': [0.0, -120000.0, -10000.0], 'vel': 250.0, 'heading': 0.0}
            f.set_match()
            env = f.env
            pols = {'Policy_' + info['userModelID']: f.importedUserModules[info['userModuleID']].getUserPolicy(info['args'])
                    for team, info in f.match_config['teams'].items()}
            obs, info = env.reset()
            events = []
            from ASRCAISim1.core import Entity
            def onHit(ev):
                w = Entity.from_json(ev['wpn']); t = Entity.from_json(ev['tgt'])
                events.append((env.manager.getElapsedTime(), 'hit', w.getFullName(), t.getFullName()))
            env.manager.getRuler()().manager.addEventHandler('Hit', onHit)
            G = {'missiles': {}, 'uid2name': {}}
            term = {'__all__': False}; trunc = {'__all__': False}
            rw = {k: 0.0 for k in obs}; dn = {k: False for k in obs}; inf = {k: None for k in obs}
            while not (term['__all__'] or trunc['__all__']):
                osp = env.get_observation_space(); asp = env.get_action_space()
                acts = {k: pols[k.split(':')[2]].step(o, rw[k], dn[k], inf[k], k, osp[k], asp[k]) for k, o in obs.items()}
                obs, rw, term, trunc, inf = env.step(acts)
                _god_step(env.manager, G)
                t = env.manager.getElapsedTime()
                ms = G['missiles']
                if t > 20 and ms and all(r.get('t_last', 0) < t - 2 for r in ms.values()):
                    break
                if t > 230 or (t > 45 and not ms):
                    break
            gl = _god_finish(G, events)
            done_t = env.manager.getElapsedTime()
            end_reason = env.manager.getRuler()().endReason.name if (term['__all__'] or trunc['__all__']) else 'lab_stop'
    except Exception:
        return {'idx': idx, 'error': traceback.format_exc()[-2000:]}
    rec = {'idx': idx, 'seed': seed, 'meta': meta, 'match_id': os.environ['MATCH_ID'],
           'godlog': [{k: v for k, v in m.items() if k != 'traj'} for m in gl], 'wall': time.time() - t0, 'n_shots': len(gl), 'end_t': done_t, 'end_reason': end_reason}
    if gl:
        m = gl[0]
        s = np.array(m['s0']); g = np.array(m['g0'])
        rel = s[:3] - g[:3]; D = float(np.linalg.norm(rel))
        tv = g[3:] / max(1e-6, np.linalg.norm(g[3:]))
        asp = math.degrees(math.acos(float(np.clip(tv @ (rel / D), -1, 1))))   # 0 = escort flying at the shooter
        sv = s[3:]; ata = math.degrees(math.acos(float(np.clip((sv / np.linalg.norm(sv)) @ (-rel / D), -1, 1))))
        tr = m.get('traj') or []
        vend = float(np.linalg.norm(np.array(tr[-1][4:7]))) if tr else None
        dmin = float(min(np.linalg.norm(np.array(x[1:4]) - np.array(x[7:10])) for x in tr)) if tr else None
        mode = tr[-1][13] if tr else None
        rec.update({'t0': m['t0'], 'D': D, 'alt_s': -s[2], 'V_s': float(np.linalg.norm(sv)), 'asp': asp, 'ata': ata,
                    'hit': bool(m['hit']), 'tof': m.get('t_last', m['t0']) - m['t0'], 'vend': vend, 'dmin': dmin,
                    'mode_end': mode, 'esc_alt_end': (-tr[-1][9]) if tr else None, 's0': m['s0'], 'g0': m['g0']})
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=100)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--seed', type=int, default=70000)
    ap.add_argument('--out', default='results/esclab.jsonl')
    ap.add_argument('--alt-lo', dest='alt_lo', type=float, default=10000.0)
    ap.add_argument('--alt-hi', dest='alt_hi', type=float, default=16500.0)
    ap.add_argument('--vfrac-lo', dest='vfrac_lo', type=float, default=0.85)
    ap.add_argument('--d-lo', dest='d_lo', type=float, default=45000.0)
    ap.add_argument('--d-hi', dest='d_hi', type=float, default=140000.0)
    ap.add_argument('--supports', default='crank,pursue,drag')
    ap.add_argument('--west-only', dest='west_only', type=int, default=1)
    ap.add_argument('--helper', type=int, default=0)
    a = ap.parse_args()
    os.environ.setdefault('SHOT_LOG', os.path.join(ROOT, 'data', 'shots_' + os.path.basename(a.out).replace('.jsonl', '') + '.jsonl'))
    jobs = [(i, a.seed + i, a) for i in range(a.episodes)]
    ctx = mp.get_context('spawn')
    with ctx.Pool(a.workers, maxtasksperchild=1) as pool, open(os.path.join(ROOT, a.out), 'a') as fo:
        for r in pool.imap_unordered(run, jobs):
            if 'error' in r:
                print('ERR', r['error'][-600:], flush=True); continue
            fo.write(json.dumps(r) + '\n'); fo.flush()
            print(f"#{r['idx']} D={r.get('D',0)/1000:.0f} alt={r.get('alt_s',0)/1000:.1f} V={r.get('V_s',0):.0f} "
                  f"asp={r.get('asp',0):.0f} S={r['meta']['S']} hit={r.get('hit')} tof={r.get('tof',0):.0f} wall={r['wall']:.0f}", flush=True)


if __name__ == '__main__':
    main()
