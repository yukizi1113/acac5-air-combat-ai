"""Ego-centric entity observation for the decentralised high-level policy (one network, shared by the 4 fighters).

Every fighter sees the world from its own horizontal heading (forward / right / down; gravity stays "down"):
  self     (1 x S_DIM) : own state, absolute altitude / speed (turn and climb performance depend on them), battlefield
                         position, threat summary, and the rule commander's suggestion (advisor input)
  friends  (4 x F_DIM) : the 3 other fighters + own escort (datalink)
  enemies  (9 x E_DIM) : fused enemy tracks (radar / IRST / memory with age = "what the radar saw a moment ago")
                         + estimated enemy escort
  missiles (8 x M_DIM) : our missiles in flight
  mws      (4 x W_DIM) : missile-warning bearings with line-of-sight rate (near-zero LOS rate = collision course)
Relative geometry of an entity: bearing (sin, cos) from own nose, elevation, range (linear and log), altitude
difference, its heading relative to ours, its aspect (angle off its nose towards us), speed and closing speed.
Only own / datalink / sensor information is used (competition rule 1); IRST tracks are never identified by order.
"""
import math
import numpy as np

S_DIM, F_DIM, E_DIM, M_DIM, W_DIM = 40, 18, 22, 14, 6
N_FR, N_EN, N_MS, N_MW = 4, 9, 8, 4
N_OPT, N_ALT, N_FIRE = 6, 3, 3
ALT_BANDS = (None, 8000.0, 13500.0)     # 0: rule altitude (v17: 11 km), 1: low, 2: high
REL_DIM = 13


def _rot(az):
    c, s = math.cos(az), math.sin(az)
    return c, s


def rel_feats(f, p, v=None):
    """Geometry of point p (velocity v) as seen from fighter f (team frame, z down)."""
    c, s = _rot(f['az'])
    d = p - f['pos']
    fwd = d[0] * c + d[1] * s
    rgt = -d[0] * s + d[1] * c
    dn = d[2]
    hor = math.hypot(fwd, rgt)
    rng = math.sqrt(hor * hor + dn * dn)
    b = math.atan2(rgt, fwd)
    elev = math.atan2(-dn, max(1.0, hor))
    out = np.zeros(REL_DIM, np.float32)
    out[0] = math.sin(b); out[1] = math.cos(b); out[2] = elev
    out[3] = min(rng / 100000.0, 3.0); out[4] = math.log1p(rng / 1000.0) / 5.0
    out[5] = (-dn) / 10000.0                                   # + : the other is higher
    if v is not None:
        V = float(np.linalg.norm(v[:2]))
        if V > 1.0:
            h = math.atan2(v[1], v[0]) - f['az']
            out[6] = math.sin(h); out[7] = math.cos(h)
            # aspect: angle between its velocity and the line from it to us (0 = pointing at us)
            to_us = -d[:2]
            n = max(1.0, float(np.linalg.norm(to_us)))
            ca = float(v[0] * to_us[0] + v[1] * to_us[1]) / (V * n)
            out[8] = ca; out[9] = math.sqrt(max(0.0, 1.0 - ca * ca))
        out[10] = float(np.linalg.norm(v)) / 500.0
        if rng > 1.0:
            u = d / rng
            out[11] = float(-((v - f['vel']) @ u)) / 500.0         # closing speed (+ = closing)
    out[12] = 1.0
    return out


def _mws_los(ag, f):
    """Per MWS bearing: ego bearing/elevation and LOS rate (deg/s) w.r.t. the previous tick's nearest bearing."""
    hist = ag.__dict__.setdefault('_mws_hist', {})
    prev = hist.get(f['name'])
    now = ag.now
    out = []
    for d in f['mws'][:N_MW]:
        u = d / max(1e-9, np.linalg.norm(d))
        rate = 0.0; age = 0.0
        if prev is not None and prev[1]:
            dt = max(1e-3, now - prev[0])
            angs = [math.acos(max(-1.0, min(1.0, float(u @ q[0])))) for q in prev[1]]
            k = int(np.argmin(angs))
            if angs[k] < math.radians(20.0):
                rate = math.degrees(angs[k]) / dt
                age = prev[1][k][1] + dt
        out.append((u, rate, age))
    hist[f['name']] = (now, [(u, age) for u, rate, age in out])
    return out


def build_obs(ag, teacher=None, ext=None):
    """Returns dict of padded arrays with a leading fighter axis (4). teacher=(opts, tgts) of the rule commander."""
    from .agent import OPT_ATTACK, B_NAMES
    F = ag.F[:4]
    nF = 4
    S = np.zeros((nF, S_DIM), np.float32)
    FR = np.zeros((nF, N_FR, F_DIM), np.float32); FRm = np.zeros((nF, N_FR), np.float32)
    EN = np.zeros((nF, N_EN, E_DIM), np.float32); ENm = np.zeros((nF, N_EN), np.float32)
    MS = np.zeros((nF, N_MS, M_DIM), np.float32); MSm = np.zeros((nF, N_MS), np.float32)
    MW = np.zeros((nF, N_MW, W_DIM), np.float32); MWm = np.zeros((nF, N_MW), np.float32)
    alive = np.zeros(nF, np.float32)
    now = ag.now
    efs = ag.fighters_e(90.0)
    efs = sorted(efs, key=lambda e: now - e.t)[:N_EN - 1]
    t_opts = teacher[0] if teacher else {}
    gk = getattr(ag, 'gk_names', set())
    n_alive = len(ag.alive)
    n_en = len(ag.fighters_e(30.0))
    kills = float(ag._n_kills()) if hasattr(ag, '_n_kills') else 0.0
    for i, f in enumerate(F):
        if not f['alive']:
            continue
        alive[i] = 1.0
        s = S[i]
        s[0] = f['alt'] / 10000.0; s[1] = f['V'] / 500.0; s[2] = f['rem'] / max(1, f['nmsl0']); s[3] = f['nmsl0'] / 10.0
        s[4] = f['vmaxs']; s[5] = f['thrust'] / 30018.0; s[6] = f['rcs']; s[7] = f['radarL'] / 100000.0
        s[8] = f['pos'][0] / 100000.0; s[9] = f['pos'][1] / 100000.0
        s[10] = math.sin(f['az']); s[11] = math.cos(f['az'])
        s[12] = now / 1200.0; s[13] = n_alive / 4.0; s[14] = n_en / 4.0; s[15] = kills / 4.0
        vt = ag._vthreat_for(f) if hasattr(ag, '_vthreat_for') else None
        if vt is not None:
            s[16] = 1.0; s[17] = min(1.0, vt['dist'] / 50000.0); s[18] = min(1.0, vt['dist'] / max(1.0, vt['closing']) / 60.0)
        s[19] = len(f['mws']) / 2.0
        guided = [m for m in ag.our_msls if m['owner'] == f['name'] and m['mode'] == 'GUIDED']
        s[20] = len(guided) / 2.0; s[21] = 1.0 if f['launchable'] else 0.0
        s[22] = 1.0 if f['name'] in gk else 0.0
        to = t_opts.get(f['name'], OPT_ATTACK)
        s[23 + int(to)] = 1.0                                       # advisor: rule commander's option (6)
        cur = ag.opt.get(f['name'], OPT_ATTACK)
        s[29 + int(cur)] = 1.0                                      # option in force (6)
        s[35] = (ag.P['fire_delta'][int(ag.fire_mode.get(f['name'], 1))] if ag.fire_mode else 0.0)
        s[36] = ag.alt_band.get(f['name'], 0) / 2.0 if hasattr(ag, 'alt_band') else 0.0
        s[37] = 1.0 if ag.myE is not None else 0.0
        s[38] = 1.0 if getattr(ag, 'race_mode', False) else 0.0
        s[39] = 1.0
        # friends: other fighters then own escort
        k = 0
        for g in F:
            if g is f or not g['alive'] or k >= N_FR - 1:
                continue
            x = FR[i, k]; x[:REL_DIM] = rel_feats(f, g['pos'], g['vel'])
            x[13] = g['rem'] / max(1, g['nmsl0']); x[14] = g['alt'] / 10000.0
            x[15] = 1.0 if g['name'] in gk else 0.0; x[16] = 1.0 if g['mws'] else 0.0; x[17] = 0.0
            FRm[i, k] = 1.0; k += 1
        if ag.myE is not None:
            x = FR[i, N_FR - 1]; x[:REL_DIM] = rel_feats(f, ag.myE['pos'], ag.myE['vel'])
            x[17] = 1.0; FRm[i, N_FR - 1] = 1.0
        # enemies: fused fighter tracks + enemy-escort estimate
        k = 0
        for e in efs:
            x = EN[i, k]; x[:REL_DIM] = rel_feats(f, e.pos, e.vel)
            x[13] = {'radar': 1.0, 'irst': 0.5, 'memory': 0.0}.get(e.src, 0.0)
            x[14] = min(1.0, (now - e.t) / 60.0)                    # time since last seen (radar memory)
            x[15] = e.q / 2.0; x[16] = 1.0 if e.dead else 0.0
            try:
                sf = ag.shot_features(f, e, 'fighter')
                if sf is not None:
                    x[17] = min(3.0, sf['rn']); x[18] = float(ag.pk(sf)); x[19] = min(3.0, sf['rh'] / 100000.0)
            except Exception:
                pass
            x[20] = len(ag.msls_on(e.uid)) / 2.0 if hasattr(ag, 'msls_on') else 0.0
            x[21] = 0.0
            ENm[i, k] = 1.0; k += 1
        if getattr(ag, 'ee_pos', None) is not None:
            x = EN[i, N_EN - 1]; x[:REL_DIM] = rel_feats(f, ag.ee_pos, ag.ee_vel)
            x[13] = {'radar': 1.0, 'irst': 0.5, 'memory': 0.0}.get(ag.EE.src, 0.0) if ag.EE is not None else 0.0
            x[14] = min(1.0, (now - ag.EE.t) / 60.0) if ag.EE is not None else 1.0
            x[21] = 1.0; ENm[i, N_EN - 1] = 1.0
        # our missiles
        for k, m in enumerate(ag.our_msls[:N_MS]):
            x = MS[i, k]; x[:REL_DIM] = rel_feats(f, m['pos'], m['vel'])
            x[13] = min(3.0, float(np.linalg.norm(m['tpos'] - m['pos'])) / 50000.0)
            MSm[i, k] = 1.0
        # MWS with LOS rate
        for k, (u, rate, age) in enumerate(_mws_los(ag, f)):
            c, sn = _rot(f['az'])
            fwd = u[0] * c + u[1] * sn; rgt = -u[0] * sn + u[1] * c
            b = math.atan2(rgt, fwd)
            x = MW[i, k]
            x[0] = math.sin(b); x[1] = math.cos(b); x[2] = math.asin(max(-1.0, min(1.0, -u[2])))
            x[3] = min(3.0, rate / 2.0); x[4] = min(1.0, age / 30.0); x[5] = 1.0
            MWm[i, k] = 1.0
    return {'self': S, 'fr': FR, 'frm': FRm, 'en': EN, 'enm': ENm, 'ms': MS, 'msm': MSm, 'mw': MW, 'mwm': MWm,
            'alive': alive}


OBS_SHAPES = {'self': (4, S_DIM), 'fr': (4, N_FR, F_DIM), 'frm': (4, N_FR), 'en': (4, N_EN, E_DIM), 'enm': (4, N_EN),
              'ms': (4, N_MS, M_DIM), 'msm': (4, N_MS), 'mw': (4, N_MW, W_DIM), 'mwm': (4, N_MW), 'alive': (4,),
              'decide': (1,), 'teacher': (12,)}
