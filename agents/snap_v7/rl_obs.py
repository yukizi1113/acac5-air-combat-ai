"""Entity-token observation for the high-level (option-selecting) policy.

Tokens
  global  (G_DIM)            : time, counts, escort status, ammunition
  friends (4 x F_DIM)        : own fighters (kinematics, sensors spec, ammo, current behaviour, threats)
  enemies (MAX_E x E_DIM)    : radar / IRST / memory tracks incl. the enemy escort estimate
  escorts (2 x S_DIM)        : own escort (truth via datalink) and enemy escort estimate
All positions in team frame normalised by 100 km (x,y) / 10 km (alt); velocities by 500 m/s.
"""
import math
import numpy as np

MAX_F = 4
MAX_E = 8
G_DIM = 10
F_DIM = 33
E_DIM = 20
S_DIM = 8
N_BEH = 11
N_OPT = 6
N_FIRE = 3


def _n(v):
    return float(np.linalg.norm(v))


def build_obs(ag):
    now = ag.now
    G = np.zeros(G_DIM, np.float32)
    Fm = np.zeros((MAX_F, F_DIM), np.float32)
    Em = np.zeros((MAX_E, E_DIM), np.float32)
    Sm = np.zeros((2, S_DIM), np.float32)
    fmask = np.zeros(MAX_F, np.float32)
    emask = np.zeros(MAX_E, np.float32)
    efs = ag.fighters_e(30.0)
    G[0] = now / 1800.0
    G[1] = len(ag.alive) / 4.0
    G[2] = len(efs) / 4.0
    G[3] = 1.0 if ag.myE is not None else 0.0
    G[4] = 1.0 if (ag.EE is not None and ag.EE.track is not None) else 0.0
    G[5] = len(ag.our_msls) / 8.0
    G[6] = sum(f['rem'] for f in ag.alive) / 24.0
    G[7] = ag.n_launched / 24.0
    G[8] = max(0.0, (now - 900.0) / 900.0)
    G[9] = 1.0
    ee = ag.ee_pos
    for i, f in enumerate(ag.F[:MAX_F]):
        if not f['alive']:
            continue
        fmask[i] = 1.0
        p, v = f['pos'], f['vel']
        x = Fm[i]
        x[0] = 1.0
        x[1:4] = [p[0] / 1e5, p[1] / 1e5, f['alt'] / 1e4]
        x[4] = f['V'] / 500.0
        x[5] = math.cos(f['az']); x[6] = math.sin(f['az'])
        x[7] = f['rem'] / 8.0
        x[8] = 1.0 if f['launchable'] else 0.0
        b = ag.beh.get(f['name'], 0)
        x[9 + min(b, N_BEH - 1)] = 1.0              # 9..19
        x[20] = 1.0 if f['mws'] else 0.0
        x[32] = 1.0 if [m for m in ag.our_msls if m['owner'] == f['name'] and m['mode'] == 'GUIDED'] else 0.0
        kmin, dmin = 3.0, 3.0
        for e in efs:
            hot, kE, d = ag.threat_level(f, e)
            if hot:
                kmin = min(kmin, kE)
            dmin = min(dmin, d / 1e5)
        x[21] = max(-1.0, min(3.0, kmin))
        x[22] = dmin
        x[23] = f['radarL'] / 1e5
        x[24] = f['radarFOR']
        x[25] = f['thrust'] / 30018.0
        x[26] = f['vmaxs']
        x[27] = f['rcs']
        rel = ee - p
        x[28] = _n(rel[:2]) / 1e5
        brg = math.atan2(rel[1], rel[0]) - f['az']
        x[29] = math.cos(brg); x[30] = math.sin(brg)
        x[31] = 1.0 if ag.opt.get(f['name']) is not None else 0.0
    # enemies: escort first, then fighters sorted by distance to our formation
    ents = []
    if ag.EE is not None:
        ents.append(ag.EE)
    cen = np.mean([f['pos'] for f in ag.alive], axis=0) if ag.alive else np.zeros(3)
    ents += sorted(efs, key=lambda e: _n(e.pos - cen))
    for j, e in enumerate(ents[:MAX_E]):
        emask[j] = 1.0
        x = Em[j]
        x[0] = 1.0
        x[1] = 1.0 if e.is_escort else 0.0
        x[2] = 1.0 if e.src == 'radar' else 0.0
        x[3] = 1.0 if e.src == 'irst' else 0.0
        x[4] = 1.0 if e.src == 'memory' else 0.0
        x[5] = min(1.0, (now - e.t) / 30.0)
        x[6:9] = [e.pos[0] / 1e5, e.pos[1] / 1e5, -e.pos[2] / 1e4]
        V = _n(e.vel)
        x[9] = V / 500.0
        az = math.atan2(e.vel[1], e.vel[0]) if V > 1 else 0.0
        x[10] = math.cos(az); x[11] = math.sin(az)
        if ag.myE is not None:
            r = ag.myE['pos'] - e.pos
            x[12] = _n(r[:2]) / 1e5
            x[13] = float(np.dot(r[:2], e.vel[:2]) / max(1.0, _n(r[:2]) * max(V, 1.0)))   # pointing at our escort
        x[14] = len(ag.msls_on(e.uid)) / 2.0
        x[15] = e.q / 2.0
        x[16] = len(e.irst) / 4.0
        x[17] = 1.0 if e.track is not None else 0.0
        x[18] = cen[0] / 1e5 - e.pos[0] / 1e5
        x[19] = 1.0
    if ag.myE is not None:
        p, v = ag.myE['pos'], ag.myE['vel']
        Sm[0] = [1.0, p[0] / 1e5, p[1] / 1e5, -p[2] / 1e4, v[0] / 500, v[1] / 500, 0.0, 1.0]
    p, v = ag.ee_pos, ag.ee_vel
    Sm[1] = [1.0, p[0] / 1e5, p[1] / 1e5, -p[2] / 1e4, v[0] / 500, v[1] / 500,
             1.0 if (ag.EE is not None and ag.EE.track is not None) else 0.0, 1.0]
    return {'g': G, 'f': Fm, 'e': Em, 's': Sm, 'fmask': fmask, 'emask': emask}
