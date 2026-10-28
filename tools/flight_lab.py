"""Flight-profile lab: how far toward the enemy do our fighters get in T seconds for a given climb schedule?

  python tools/flight_lab.py --me agents/TacB --overrides '{"climb_v0":420}' --seeds 6 --T 300
Red = our agent with all weapons disabled, Blue = passive lab aircraft far away.  Prints mean x (toward the enemy),
altitude and speed of our fighters at a few checkpoints.
"""
import os, sys, json, argparse, io, contextlib
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def run(me, ov, seed, T):
    os.chdir(ROOT); sys.path.insert(0, ROOT)
    from tools.utils import make_match_info
    from tools.fight_ext import Fight
    np.random.seed(seed)
    ov = dict(ov); ov.update({'escort_max_dist': 0.0, 'striker_max_dist': 0.0, 'fighter_max_dist': 0.0,
                              'n_interceptors': 0, 'engage_radius': 0.0, 'n_defenders_max': 0, 'use_vthreat': False})
    mod = '.'.join(me.strip('/').split('/'))
    red = {'userModelID': '0', 'userModuleID': mod, 'args': {'overrides': ov}}
    tg = [{'passive': True, 'alt': 10000.0, 'heading_abs': 270.0} for _ in range(4)]
    blue = {'userModelID': '1', 'userModuleID': 'agents.Lab', 'args': {'overrides': {'lab_role': 'target', 'lab_params': tg}}}
    mi = make_match_info(red=red, blue=blue, replay=False, movie_dir='', time_out=1e9, common_dir='common',
                         control_init='random', youth=False, make_log=False, visualize=False)
    mi['seed'] = seed
    out = {}
    with contextlib.redirect_stdout(io.StringIO()):
        f = Fight(mi); f.make_matchInfo()
        for k in range(4):
            f.matchInfo['teams']['Blue']['InitialState'][k] = {'pos': [-60000.0 + 40000 * k, 145000.0, -10000.0], 'vel': 250.0, 'heading': 90.0}
        f.set_match()
        env = f.env
        pols = {'Policy_' + info['userModelID']: f.importedUserModules[info['userModuleID']].getUserPolicy(info['args'])
                for team, info in f.match_config['teams'].items()}
        obs, info = env.reset()
        rw = {k: 0.0 for k in obs}; dn = {k: False for k in obs}; inf = {k: None for k in obs}
        from ASRCAISim1.core import Fighter, MotionState
        while True:
            osp = env.get_observation_space(); asp = env.get_action_space()
            acts = {k: pols[k.split(':')[2]].step(o, rw[k], dn[k], inf[k], k, osp[k], asp[k]) for k, o in obs.items()}
            obs, rw, term, trunc, inf = env.step(acts)
            t = env.manager.getElapsedTime()
            if int(t) % 50 == 0:
                fs = [a() for a in env.manager.getAssets(lambda a: isinstance(a, Fighter) and a.getTeam() == 'Red' and 'F' in a.getName().replace('Red', ''))]
                rows = []
                for a in fs:
                    p = np.array(a.pos()); v = np.array(a.vel())
                    rows.append((p[1] + 100000.0, -p[2], float(np.linalg.norm(v))))   # Red flies toward +y (world)
                out[int(t)] = np.mean(rows, axis=0).tolist()
            if t >= T or term['__all__'] or trunc['__all__']:
                out[-1] = [t, 0, 0]
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--me', default='agents/TacB')
    ap.add_argument('--overrides', default='{}')
    ap.add_argument('--seeds', type=int, default=4)
    ap.add_argument('--seed0', type=int, default=5000)
    ap.add_argument('--T', type=float, default=300)
    a = ap.parse_args()
    ov = json.loads(a.overrides)
    import multiprocessing as mp
    ctx = mp.get_context('spawn')
    with ctx.Pool(2, maxtasksperchild=1) as pool:
        res = pool.starmap(run, [(a.me, ov, a.seed0 + i, a.T) for i in range(a.seeds)])
    ks = sorted(set().union(*[r.keys() for r in res]))
    for k in ks:
        v = np.array([r[k] for r in res if k in r])
        print(f't={k:4d}  dist={v[:,0].mean()/1000:6.1f} km  alt={v[:,1].mean()/1000:5.1f} km  V={v[:,2].mean():4.0f}')


if __name__ == '__main__':
    main()
