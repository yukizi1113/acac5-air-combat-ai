"""Missile engagement lab: randomized 1v1 shot scenarios (4 isolated pairs per episode).

  python tools/lab.py --episodes 40 --workers 2 --out results/col_lab1.jsonl
Shot features are logged agent-side (SHOT_LOG) and outcomes via the god-view missile log, in the
same format as normal matches, so tools/train_pk.py can use both.
"""
import os, sys, json, time, argparse, multiprocessing as mp, traceback
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

EVASIONS = ['none', 'drag_level', 'beam_level', 'spiral', 'beam_osc', 'drag_osc', 'dive_x']
SUPPORTS = ['support', 'pursue', 'drag']


def scenario(rng, evasions, supports):
    sh, tg = [], []
    init_r, init_b = [], []
    for i, N in enumerate([-105000.0, -35000.0, 35000.0, 105000.0]):
        D = float(rng.choice([rng.uniform(8000, 35000), rng.uniform(25000, 80000)]))
        asp = rng.choice(['head', 'tail', 'beam', 'rand'])
        h_t = {'head': 270.0, 'tail': 90.0, 'beam': float(rng.choice([0.0, 180.0])), 'rand': rng.uniform(0, 360)}[asp]
        h_t += rng.uniform(-15, 15)
        alt_s, alt_t = rng.uniform(5000, 13000), rng.uniform(5000, 13000)
        dN = rng.uniform(-4000, 4000)
        init_r.append({'pos': [N, -D / 2, -alt_s], 'vel': rng.uniform(260, 450), 'heading': 90.0 + rng.uniform(-15, 15)})
        init_b.append({'pos': [N + dN, D / 2, -alt_t], 'vel': rng.uniform(260, 450), 'heading': h_t % 360})
        E = str(rng.choice(evasions)); S = str(rng.choice(supports))
        if os.environ.get('LAB_LONG'):
            # head-on long-range shot: target holds course (no knowledge of the shot) or evades on MWS only
            h_t = 270.0 + rng.uniform(-10, 10)
            init_b[-1]['heading'] = h_t % 360
            D = float(rng.uniform(40000, 85000)); init_b[-1]['pos'][1] = D / 2; init_r[-1]['pos'][1] = -D / 2
            alt_s = float(rng.uniform(10000, 14000)); alt_t = float(rng.uniform(8000, 14000))
            init_r[-1]['pos'][2] = -alt_s; init_b[-1]['pos'][2] = -alt_t
            init_r[-1]['vel'] = float(rng.uniform(450, 560)); init_b[-1]['vel'] = float(rng.uniform(420, 560))
            E = str(rng.choice(['none', 'drag_level']))
            sh.append({'alt': alt_s, 'S': 'support', 'E_tgt': E, 't_fire': 2.0})
            tg.append({'alt': alt_t, 'E': E, 'heading_abs': h_t % 360, 'thr': 1.0, 'hold': 999.0})
            continue
        if os.environ.get('LAB_COLD'):
            # head-on duel geometry, target turns cold at a random time after the (fixed) earliest launch time
            h_t = 270.0 + rng.uniform(-10, 10)
            init_b[-1]['heading'] = h_t % 360
            D = float(rng.uniform(25000, 65000)); init_b[-1]['pos'][1] = D / 2; init_r[-1]['pos'][1] = -D / 2
            alt_s = alt_t = float(rng.uniform(10000, 14000)); init_r[-1]['pos'][2] = -alt_s; init_b[-1]['pos'][2] = -alt_t
            init_r[-1]['vel'] = init_b[-1]['vel'] = float(rng.uniform(450, 560))
            sh.append({'alt': alt_s, 'S': 'support', 'E_tgt': 'cold_at', 't_fire': 3.0})
            tg.append({'alt': alt_t, 'E': 'cold_at', 'heading_abs': h_t % 360, 'thr': 1.0, 't_cold': 3.0 + float(rng.uniform(0, 40))})
            continue
        sh.append({'alt': alt_s, 'S': S, 'E_tgt': E, 't_fire': 0.0})
        tg.append({'alt': alt_t, 'E': E, 'heading_abs': h_t % 360, 'thr': float(rng.uniform(0.6, 1.0)), 'hold': float(rng.uniform(10, 40))})
    return sh, tg, init_r, init_b


def escort_scenario(rng, esc_state):
    """Shooter 0 placed around the Blue escort at random range/bearing, pointing at it."""
    import math
    ep = np.array(esc_state['pos'], dtype=float)
    for _ in range(50):
        D = rng.uniform(15000, 90000)
        brg = rng.uniform(0, 2 * math.pi)
        p = ep + np.array([D * math.cos(brg), D * math.sin(brg), 0.0])
        if abs(p[0]) < 140000 and abs(p[1]) < 140000:
            break
    alt = rng.uniform(6000, 13000)
    p[2] = -alt
    hdg = math.degrees(math.atan2(ep[1] - p[1], ep[0] - p[0])) % 360
    return {'pos': p.tolist(), 'vel': float(rng.uniform(300, 500)), 'heading': hdg}, \
        {'alt': alt, 'S': str(rng.choice(['support', 'pursue', 'drag'])), 'E_tgt': 'escort', 't_fire': 0.0,
         'tgt_hint': ep.tolist()}


def run(job):
    os.chdir(ROOT); sys.path.insert(0, ROOT)
    idx, seed, ev, su = job
    os.environ['MATCH_ID'] = f"{os.environ.get('RUN_TAG','lab')}_{seed}_Red_{idx}"
    import io, contextlib
    from tools.utils import make_match_info
    from tools.fight_ext import Fight, _god_step, _god_finish
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    sh, tg, init_r, init_b = scenario(rng, ev, su)
    esc = None
    if os.environ.get('LAB_ESCORT'):
        esc = {'pos': [float(rng.uniform(-95000, 95000)), 120000.0, -10000.0], 'vel': 250.0, 'heading': float(rng.choice([0.0, 180.0]))}
        esc_sh_state, sh[0] = escort_scenario(rng, esc)
        tg[0] = {'alt': 10000.0, 'E': 'none', 'heading_abs': 180.0, 'thr': 0.5, 'hold': 10.0}
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
            if esc is not None:
                f.matchInfo['teams']['Blue']['InitialState'][4] = esc
                f.matchInfo['teams']['Red']['InitialState'][0] = esc_sh_state
                f.matchInfo['teams']['Blue']['InitialState'][0] = {'pos': [-140000.0, 60000.0, -10000.0], 'vel': 260.0, 'heading': 0.0}
            f.set_match()
            env = f.env
            pols = {'Policy_' + info['userModelID']: f.importedUserModules[info['userModuleID']].getUserPolicy(info['args'])
                    for team, info in f.match_config['teams'].items()}
            obs, info = env.reset()
            events = []
            from ASRCAISim1.core import Entity
            def onHit(a):
                w = Entity.from_json(a['wpn']); t = Entity.from_json(a['tgt'])
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
                if t > 40 and ms and all(r.get('t_last', 0) < t - 2 for r in ms.values()):
                    break
                if t > 240 or (t > 60 and not ms):
                    break
            gl = _god_finish(G, events)
    except Exception:
        return {'idx': idx, 'error': traceback.format_exc()[-2000:]}
    return {'idx': idx, 'seed': seed, 'match_id': os.environ['MATCH_ID'], 'godlog': gl, 'wall': time.time() - t0,
            'n_shots': len(gl), 'n_hits': sum(m['hit'] for m in gl)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=20)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--seed', type=int, default=50000)
    ap.add_argument('--out', default='results/col_lab.jsonl')
    ap.add_argument('--evasions', default=','.join(EVASIONS))
    ap.add_argument('--supports', default=','.join(SUPPORTS))
    a = ap.parse_args()
    os.environ['RUN_TAG'] = os.path.basename(a.out).replace('.jsonl', '')
    os.environ.setdefault('SHOT_LOG', os.path.join(ROOT, 'data', f"shots_{os.environ['RUN_TAG']}.jsonl"))
    jobs = [(i, a.seed + i, a.evasions.split(','), a.supports.split(',')) for i in range(a.episodes)]
    ctx = mp.get_context('spawn')
    with ctx.Pool(a.workers, maxtasksperchild=1) as pool, open(os.path.join(ROOT, a.out), 'w') as fo:
        for r in pool.imap_unordered(run, jobs):
            if 'error' in r:
                print('ERR', r['error']); continue
            fo.write(json.dumps(r) + '\n'); fo.flush()
            print(f"lab #{r['idx']} shots={r['n_shots']} hits={r['n_hits']} wall={r['wall']:.0f}s", flush=True)


if __name__ == '__main__':
    main()
