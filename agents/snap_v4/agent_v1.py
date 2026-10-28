"""Tactical hierarchical agent (M-mode: one Agent controls all 4 fighters).

Layers
  1. World model  : fuses radar Track3D (team-shared), IRST Track2D bearings (per fighter),
                    MWS warnings, own missiles, own escort, and prior knowledge of the
                    enemy escort racetrack into a single team-frame picture.
  2. Skills       : parameterised manoeuvres (attack/intercept, crank-support, evade, defend, ...).
  3. Fire control : candidate shots scored by an estimated hit probability (heuristic or
                    learned model), with de-confliction so that each target has at most
                    the desired number of missiles in the air.
  4. Commander    : chooses an option (macro-action) for each fighter. The rule-based commander
                    is the default; an RL policy can override via the action interface.
Team frame: x = towards the enemy, y = right, z = down, origin = battlefield centre.
"""
import math
import numpy as np
from math import atan2, cos, sin, sqrt, degrees, radians
from gymnasium import spaces

from ASRCAISim1.core import (  # type: ignore
    Agent, MotionState, Track3D, Track2D, AltitudeKeeper, Time, TimeSystem, getValueFromJsonKRD, nljson,
)
from BasicAgentUtility.util import TeamOrigin, calcRHead, calcRTail, calcRNorm, calcRHeadE, calcRTailE  # type: ignore

# ---------------------------------------------------------------- option ids
OPT_ATTACK = 0      # go for enemy escort, shoot it
OPT_SUPPORT = 1     # crank: keep missile target in radar FOR
OPT_DEFEND = 2      # intercept enemy fighter threatening our escort
OPT_ENGAGE = 3      # engage nearest enemy fighter
OPT_EVADE = 4       # missile evasion (forced by MWS)
OPT_RETREAT = 5     # disengage towards own side
OPT_NAMES = ['ATTACK', 'SUPPORT', 'DEFEND', 'ENGAGE', 'EVADE', 'RETREAT']
NUM_OPTIONS = 6

ESCORT_X = 120000.0       # enemy escort racetrack x (team frame)
ESCORT_YMAX = 100000.0
ESCORT_ALT = 10000.0


def norm(v):
    return float(np.linalg.norm(v))


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v * 0.0


def az_of(v):
    return atan2(v[1], v[0])


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def dir_from_az(az, pitch=0.0):
    return np.array([cos(az) * cos(pitch), sin(az) * cos(pitch), -sin(pitch)])


class EnemyTrack:
    __slots__ = ('uid', 'pos', 'vel', 't', 'track', 'src', 'escort_score', 'is_escort', 'first_t',
                 'irst_dirs', 'alive_guess')

    def __init__(self, uid):
        self.uid = uid
        self.pos = np.zeros(3)
        self.vel = np.zeros(3)
        self.t = -1e9
        self.track = None      # latest radar Track3D (local CRS) usable for launching
        self.src = 'none'      # 'radar' | 'irst' | 'memory'
        self.escort_score = 0.0
        self.is_escort = False
        self.first_t = None
        self.irst_dirs = []
        self.alive_guess = True


class TacAgent(Agent):
    # ======================================================== setup
    def initialize(self):
        super().initialize()
        mc = self.modelConfig
        g = lambda k, d: getValueFromJsonKRD(mc, k, self.randomGen, d)
        self.own = self.getTeam()
        # tunable tactical parameters
        self.P = {
            'attack_alt': g('attack_alt', 12000.0),
            'cruise_alt': g('cruise_alt', 11000.0),
            'evade_alt': g('evade_alt', 4000.0),
            'k_escort': g('k_escort', 0.45),          # normalised range threshold vs escort
            'escort_max_dist': g('escort_max_dist', 60000.0),
            'k_fighter': g('k_fighter', 0.15),        # vs fighters (self defence)
            'k_defend': g('k_defend', 0.30),
            'fighter_max_dist': g('fighter_max_dist', 40000.0),
            'crank_margin_deg': g('crank_margin_deg', 15.0),
            'crank_max_deg': g('crank_max_deg', 55.0),
            'mws_hold': g('mws_hold', 4.0),
            'evade_pitch_deg': g('evade_pitch_deg', 30.0),
            'defend_radius': g('defend_radius', 90000.0),
            'n_defenders_max': g('n_defenders_max', 2),
            'boundary_margin': g('boundary_margin', 12000.0),
            'max_msl_escort': g('max_msl_escort', 1),
            'max_msl_fighter': g('max_msl_fighter', 1),
            'threat_k': g('threat_k', 0.35),
            'min_speed': g('min_speed', 300.0),
            'debug': g('debug', False),
        }
        self.altK = AltitudeKeeper(mc().get('altitudeKeeper', {'pGain': -0.3, 'dGain': -0.1, 'minPitch': -10.0, 'maxPitch': 10.0}))
        self.altKfast = AltitudeKeeper({'pGain': -0.3, 'dGain': -0.1, 'minPitch': -self.P['evade_pitch_deg'], 'maxPitch': 15.0})
        self.enemies = {}
        self.ee_est = None          # enemy escort estimate (pos, vel, t, quality)
        self.opt = {}               # current option per fighter name
        self.opt_target = {}        # target uid per fighter
        self.opt_since = {}
        self.last_mws_t = {}
        self.evade_dir = {}
        self.pending_shot = {}      # name -> (t, uid)
        self.shot_log = []          # (t, shooter, uid, features) for data collection
        self.launched_seen = set()
        self.msl_target = {}        # missile fullName -> uid
        self.ext_action = None      # high-level action from external policy (RL)
        self.decision_interval = int(g('decision_interval', 5))
        self.tick = 0
        self.cmd = {}

    def validate(self):
        r = self.manager.getRuler()().observables()
        self.dOut = r['dOut']
        self.dLine = r['dLine']
        self.teamOrigin = TeamOrigin(self.own == r['eastSider'], self.dLine)
        self.names = [p.getFullName() for p in self.parents.values()]
        self.ports = list(self.parents.keys())

    # ======================================================== frames
    def T(self, v):            # local -> team (rotation only; origin = centre)
        return np.array(self.teamOrigin.relPtoB(np.asarray(v, dtype=np.float64)))

    def L(self, v):            # team -> local
        return np.array(self.teamOrigin.relBtoP(np.asarray(v, dtype=np.float64)))

    # ======================================================== world model
    def _build_world(self):
        t = self.manager.getElapsedTime()
        self.now = t
        lc = self.getLocalCRS()
        first = None
        for p in self.parents.values():
            if p.isAlive():
                first = p
                break
        self.first = first
        F = []
        for port, p in self.parents.items():
            name = p.getFullName()
            f = {'name': name, 'port': port, 'alive': p.isAlive(), 'acc': p}
            if p.isAlive():
                o = p.observables
                m = MotionState(o['motion']).transformTo(lc)
                f['motion'] = m
                f['omotion'] = MotionState(o['motion'])
                f['pos'] = self.T(m.pos())
                f['vel'] = self.T(m.vel())
                f['V'] = norm(f['vel'])
                f['alt'] = -f['pos'][2]
                f['az'] = az_of(f['vel'])
                spec = o['spec']()
                f['radarL'] = spec['sensor']['radar']['Lref']
                f['radarFOR'] = spec['sensor']['radar']['thetaFOR']
                f['irstL'] = spec['sensor']['irst']['Lref']
                f['irstFOR'] = spec['sensor']['irst']['thetaFOR']
                f['thrust'] = spec['weapon']['missile']['thrust']
                f['vmaxScale'] = spec['dynamics']['scale']['vMax']
                w = o['weapon']()
                f['rem'] = int(w['remMsls'])
                f['launchable'] = bool(w['launchable'])
                f['msls'] = w.get('missiles', [])
                f['mws'] = []
                if o.contains_p('/sensor/mws/track'):
                    for mo in o.at_p('/sensor/mws/track'):
                        tr = Track2D(mo).transformTo(lc)
                        f['mws'].append(self.T(tr.dir()))
                f['irst'] = []
                if o.contains_p('/sensor/irst/track'):
                    for io in o.at_p('/sensor/irst/track'):
                        tr = Track2D(io).transformTo(lc)
                        f['irst'].append((str(tr.truth), self.T(tr.dir()), self.T(tr.origin())))
            else:
                f['rem'] = 0
                try:
                    f['msls'] = first.observables.at_p('/shared/fighter').at(name)()['weapon'].get('missiles', []) if first is not None else []
                except Exception:
                    f['msls'] = []
            F.append(f)
        self.F = F
        # own escort
        self.myE = None
        if first is not None:
            for nm, fo in first.observables.at_p('/shared/fighter').items():
                if nm not in self.names:
                    if fo['isAlive']():
                        m = MotionState(fo['motion']).transformTo(lc)
                        self.myE = {'pos': self.T(m.pos()), 'vel': self.T(m.vel())}
        # radar tracks (team shared)
        seen = set()
        if first is not None:
            for tj in first.observables.at_p('/sensor/track'):
                tr = Track3D(tj).transformTo(lc)
                uid = str(tr.truth)
                e = self.enemies.get(uid)
                if e is None:
                    e = EnemyTrack(uid)
                    e.first_t = t
                    self.enemies[uid] = e
                e.pos = self.T(tr.pos())
                e.vel = self.T(tr.vel())
                e.t = t
                e.track = tr
                e.src = 'radar'
                seen.add(uid)
        for uid, e in self.enemies.items():
            if uid not in seen:
                e.track = None
                if e.src == 'radar' or e.src == 'memory':
                    e.src = 'memory'
        # IRST bearings
        irst_by_uid = {}
        for f in F:
            if f['alive']:
                for uid, d, org in f['irst']:
                    irst_by_uid.setdefault(uid, []).append((org, d))
        self.irst_by_uid = irst_by_uid
        for uid, lst in irst_by_uid.items():
            e = self.enemies.get(uid)
            if e is None:
                e = EnemyTrack(uid)
                e.first_t = t
                self.enemies[uid] = e
            e.irst_dirs = lst
            if uid not in seen:
                est = self._irst_fix(lst, e)
                if est is not None:
                    newpos, q = est
                    if e.t > -1e8 and t - e.t < 30:
                        e.vel = 0.7 * e.vel + 0.3 * (newpos - e.pos) / max(1.0, t - e.t) if e.src == 'irst' else e.vel
                    e.pos = newpos
                    e.t = t
                    e.src = 'irst'
        # escort classification
        self._classify()
        # own missiles
        self.our_msls = []
        for f in F:
            for mo in f['msls']:
                if mo.get('hasLaunched', False):
                    nm = mo['fullName']
                    alive = bool(mo['isAlive'])
                    tgt = Track3D(mo['target']).transformTo(lc)
                    uid = str(tgt.truth)
                    if nm not in self.launched_seen:
                        self.launched_seen.add(nm)
                        self.msl_target[nm] = uid
                    if alive:
                        mm = MotionState(mo['motion']).transformTo(lc)
                        self.our_msls.append({'name': nm, 'owner': f['name'], 'uid': self.msl_target.get(nm, uid),
                                              'pos': self.T(mm.pos()), 'vel': self.T(mm.vel()),
                                              'mode': mo['mode'], 'tpos': self.T(tgt.pos()),
                                              'launchedT': Time(mo['launchedT'])})

    def _irst_fix(self, lst, e):
        """Position fix from IRST bearings.
        >=2 observers: least squares intersection of bearing lines (3D).
        1 observer: if escort-like, intersect bearing with the known racetrack plane x=ESCORT_X;
                    otherwise use previous range estimate along the new bearing."""
        if len(lst) >= 2:
            A = np.zeros((3, 3)); b = np.zeros(3)
            for org, d in lst:
                d = unit(d)
                M = np.eye(3) - np.outer(d, d)
                A += M; b += M @ org
            # baseline check
            dmat = np.array([unit(d) for _, d in lst])
            spread = np.linalg.norm(dmat - dmat.mean(0), axis=1).max()
            if spread > 0.02:
                try:
                    p = np.linalg.solve(A + 1e-9 * np.eye(3), b)
                    rng = norm(p - lst[0][0])
                    if 1000 < rng < 400000:
                        return p, 2
                except np.linalg.LinAlgError:
                    pass
        org, d = lst[0]
        d = unit(d)
        if e.is_escort or e.escort_score > 0.5 or (e.track is None and e.first_t is not None and e.first_t < 5):
            if d[0] > 0.05:
                s = (ESCORT_X - org[0]) / d[0]
                p = org + s * d
                p[2] = -ESCORT_ALT
                return p, 1
        if e.t > -1e8:
            r = norm(e.pos - org)
            return org + d * r, 0
        return None

    def _classify(self):
        for uid, e in self.enemies.items():
            if e.is_escort:
                continue
            # evidence 1: seen by IRST at t~0 (only the escort's IR signature reaches that far)
            if e.first_t is not None and e.first_t < 3 and e.src == 'irst':
                e.escort_score += 1.0
            if e.src == 'radar':
                # evidence 2: radar detection range beyond any fighter's possible detection range
                dmin = min([norm(f['pos'] - e.pos) for f in self.F if f['alive']] or [0])
                if dmin > 100000:
                    e.escort_score += 2.0
                # evidence 3: kinematics consistent with the racetrack
                V = norm(e.vel)
                if abs(e.pos[0] - ESCORT_X) < 12000 and V < 320 and abs(-e.pos[2] - ESCORT_ALT) < 1500 \
                        and abs(e.vel[0]) < 0.5 * V:
                    e.escort_score += 0.5
                elif V > 360:
                    e.escort_score -= 1.0
            if e.escort_score >= 2.0:
                e.is_escort = True
        # only one escort: keep the strongest
        esc = [e for e in self.enemies.values() if e.is_escort]
        if len(esc) > 1:
            esc.sort(key=lambda e: -e.escort_score)
            for e in esc[1:]:
                e.is_escort = False
                e.escort_score = 0.0
        self.EE = esc[0] if esc else None
        # estimate of enemy escort state even without tracks
        if self.EE is not None and self.now - self.EE.t < 60:
            p = self.EE.pos + self.EE.vel * (self.now - self.EE.t) if self.EE.src != 'radar' else self.EE.pos
            self.ee_pos = p
            self.ee_vel = self.EE.vel
        else:
            self.ee_pos = np.array([ESCORT_X, 0.0, -ESCORT_ALT]) if getattr(self, 'ee_pos', None) is None else self.ee_pos
            self.ee_vel = np.zeros(3) if getattr(self, 'ee_vel', None) is None else self.ee_vel

    # ======================================================== helpers
    def enemy_fighters(self, max_age=20.0):
        return [e for e in self.enemies.values() if not e.is_escort and e.alive_guess and self.now - e.t <= max_age
                and e.src in ('radar', 'memory', 'irst')]

    def msls_on(self, uid):
        return [m for m in self.our_msls if m['uid'] == uid]

    def rnorm(self, f, e):
        try:
            return float(calcRNorm(f['acc'], f['motion'], e.track, False))
        except Exception:
            return 9.9

    def rnormE(self, f, e):
        """Normalised range of enemy e's missile against f (RTail->0, RHead->1)."""
        try:
            tr = e.track
            if tr is None:
                return 9.9
            rh = float(calcRHeadE(f['acc'], f['motion'], tr, False))
            rt = float(calcRTailE(f['acc'], f['motion'], tr, False))
            d = norm(f['pos'] - e.pos)
            return (d - rt) / max(1.0, rh - rt)
        except Exception:
            return 9.9

    def intercept_point(self, f, tp, tv, vown=None):
        vown = vown or max(f['V'], 300.0)
        rel = tp - f['pos']
        a = tv @ tv - vown * vown
        b = 2 * rel @ tv
        c = rel @ rel
        tt = None
        if abs(a) < 1e-6:
            if abs(b) > 1e-6:
                tt = -c / b
        else:
            disc = b * b - 4 * a * c
            if disc >= 0:
                r1 = (-b - sqrt(disc)) / (2 * a); r2 = (-b + sqrt(disc)) / (2 * a)
                cands = [r for r in (r1, r2) if r > 0]
                tt = min(cands) if cands else None
        if tt is None or tt > 600:
            tt = norm(rel) / vown
        return tp + tv * min(tt, 600.0)

    def off_boresight(self, f, p):
        return abs(wrap(az_of(p - f['pos']) - f['az']))

    # ======================================================== commander
    def _assign_roles(self):
        P = self.P
        alive = [f for f in self.F if f['alive']]
        threats = []
        if self.myE is not None:
            for e in self.enemy_fighters():
                d = norm(e.pos[:2] - self.myE['pos'][:2])
                closing = (self.myE['pos'] - e.pos)[:2] @ e.vel[:2] > 0
                if d < P['defend_radius'] or (e.pos[0] < -20000 and closing):
                    threats.append((d, e))
        threats.sort(key=lambda x: x[0])
        roles = {}
        tgts = {}
        used = set()
        ndef = 0
        for d, e in threats:
            if ndef >= P['n_defenders_max']:
                break
            cands = [f for f in alive if f['name'] not in used and f['rem'] > 0]
            if not cands:
                break
            # defender = friend closest to the threat (time to reach threat)
            best = min(cands, key=lambda f: norm(f['pos'] - e.pos) + (0 if f['pos'][0] < e.pos[0] + 30000 else 40000))
            # only defend if the threat is between us and our escort or near it
            roles[best['name']] = OPT_DEFEND
            tgts[best['name']] = e.uid
            used.add(best['name'])
            ndef += 1
        for f in alive:
            if f['name'] not in roles:
                roles[f['name']] = OPT_ATTACK if f['rem'] > 0 else OPT_RETREAT
        return roles, tgts

    # ======================================================== skills -> (az, alt, pitch_mode, throttle)
    def skill_attack(self, f):
        P = self.P
        tp = self.ee_pos
        aim = self.intercept_point(f, tp, self.ee_vel)
        # stand-off: don't fly past the escort towards the enemy line
        aim = aim.copy()
        aim[0] = min(aim[0], self.dLine - 20000)
        az = az_of(aim - f['pos'])
        return az, P['attack_alt'], 'normal', 1.0

    def skill_support(self, f, uid):
        """Crank: keep the target near the edge of the radar FOR while missile is guided."""
        P = self.P
        e = self.enemies.get(uid)
        if e is None:
            return self.skill_attack(f)
        brg = az_of(e.pos - f['pos'])
        crank = min(math.radians(P['crank_max_deg']), f['radarFOR'] - math.radians(P['crank_margin_deg']))
        crank = max(0.0, crank)
        # choose the side that turns us less / away from boundary
        c1, c2 = wrap(brg + crank), wrap(brg - crank)
        side_pref = c1 if abs(wrap(c1 - f['az'])) < abs(wrap(c2 - f['az'])) else c2
        return side_pref, P['attack_alt'], 'normal', 1.0

    def skill_engage(self, f, uid):
        e = self.enemies.get(uid)
        if e is None:
            return self.skill_attack(f)
        aim = self.intercept_point(f, e.pos, e.vel)
        return az_of(aim - f['pos']), max(self.P['cruise_alt'], -e.pos[2]), 'normal', 1.0

    def skill_evade(self, f):
        name = f['name']
        d = self.evade_dir.get(name)
        if d is None:
            return self.skill_attack(f)
        az = az_of(-d)
        return az, self.P['evade_alt'], 'fast', 1.0

    def skill_retreat(self, f):
        tgt = self.myE['pos'] if self.myE is not None else np.array([-100000.0, 0, -10000])
        return az_of(tgt - f['pos']), self.P['cruise_alt'], 'normal', 1.0

    # ======================================================== fire control
    def fire_control(self, f, role, role_tgt):
        P = self.P
        if not f['launchable'] or f['rem'] <= 0:
            return None
        pend = self.pending_shot.get(f['name'])
        if pend is not None and self.now - pend[0] < 8:
            return None
        cands = []
        # 1) enemy escort
        E = self.EE
        if E is not None and E.track is not None:
            d = norm(E.pos - f['pos'])
            if len(self.msls_on(E.uid)) < P['max_msl_escort'] and not self._pending_on(E.uid):
                rn = self.rnorm(f, E)
                if (rn <= P['k_escort'] or d < 25000) and d < P['escort_max_dist'] and self.off_boresight(f, E.pos) < f['radarFOR'] - 0.1:
                    cands.append((rn - 1.0, E, 'escort', rn, d))
        # 2) fighters
        for e in self.enemy_fighters(max_age=0.5):
            if e.track is None:
                continue
            if len(self.msls_on(e.uid)) >= P['max_msl_fighter'] or self._pending_on(e.uid):
                continue
            d = norm(e.pos - f['pos'])
            if d > P['fighter_max_dist']:
                continue
            rn = self.rnorm(f, e)
            k = P['k_defend'] if (role == OPT_DEFEND and role_tgt == e.uid) else P['k_fighter']
            if rn <= k and self.off_boresight(f, e.pos) < f['radarFOR'] - 0.1:
                cands.append((rn, e, 'fighter', rn, d))
        if not cands:
            return None
        cands.sort(key=lambda c: c[0])
        _, e, kind, rn, d = cands[0]
        return e, kind, rn, d

    def _pending_on(self, uid):
        for nm, (t0, u) in self.pending_shot.items():
            if u == uid and self.now - t0 < 8:
                return True
        return False

    # ======================================================== gym interface
    def observation_space(self):
        return spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)

    def action_space(self):
        return spaces.MultiDiscrete([NUM_OPTIONS] * len(self.parents))

    def makeObs(self):
        self.tick += 1
        self._build_world()
        return np.zeros(1, dtype=np.float32)

    def deploy(self, action):
        self.ext_action = action
        P = self.P
        self.cmd = {}
        # --- missile warnings
        for f in self.F:
            if f['alive'] and f['mws']:
                self.last_mws_t[f['name']] = self.now
                self.evade_dir[f['name']] = unit(np.mean([unit(d) for d in f['mws']], axis=0))
        # --- roles
        if self.tick % self.decision_interval == 1 or not self.opt:
            roles, tgts = self._assign_roles()
            for nm, r in roles.items():
                if self.opt.get(nm) != r:
                    self.opt_since[nm] = self.now
                self.opt[nm] = r
                self.opt_target[nm] = tgts.get(nm)
        for f in self.F:
            if not f['alive']:
                continue
            nm = f['name']
            role = self.opt.get(nm, OPT_ATTACK)
            rtgt = self.opt_target.get(nm)
            eff = role
            # forced evasion
            if self.now - self.last_mws_t.get(nm, -1e9) <= P['mws_hold']:
                eff = OPT_EVADE
            else:
                # support own guided missile
                mine = [m for m in self.our_msls if m['owner'] == nm and m['mode'] != 'SELF']
                if mine:
                    eff = OPT_SUPPORT
                    rtgt = mine[0]['uid']
            # --- skill
            if eff == OPT_EVADE:
                az, alt, mode, thr = self.skill_evade(f)
            elif eff == OPT_SUPPORT:
                az, alt, mode, thr = self.skill_support(f, rtgt)
            elif eff == OPT_DEFEND or eff == OPT_ENGAGE:
                az, alt, mode, thr = self.skill_engage(f, rtgt)
            elif eff == OPT_RETREAT:
                az, alt, mode, thr = self.skill_retreat(f)
            else:
                az, alt, mode, thr = self.skill_attack(f)
            az = self._boundary(f, az)
            # --- fire
            shot = None
            if eff != OPT_EVADE:
                shot = self.fire_control(f, role, rtgt)
            launch = False
            target = Track3D()
            if shot is not None:
                e, kind, rn, d = shot
                launch = True
                target = e.track
                self.pending_shot[nm] = (self.now, e.uid)
                self.shot_log.append({'t': self.now, 'shooter': nm, 'uid': e.uid, 'kind': kind, 'rn': rn, 'd': d})
                if P['debug']:
                    print(f"[DBG] t={self.now:.0f} {nm} SHOOT {kind} rn={rn:.2f} d={d/1000:.1f}km")
            self._command(f, az, alt, mode, thr, launch, target)
            f['eff'] = eff
        if P['debug'] and self.tick % 30 == 1:
            s = ' '.join(f"{f['name'][-2:]}:{OPT_NAMES[f.get('eff',0)]}({f['pos'][0]/1000:.0f},{f['pos'][1]/1000:.0f},{f['alt']/1000:.1f},{f['V']:.0f})"
                         for f in self.F if f['alive'])
            ee = self.EE
            print(f"[DBG] t={self.now:.0f} {s} EE={'none' if ee is None else ee.src}"
                  f" eepos=({self.ee_pos[0]/1000:.0f},{self.ee_pos[1]/1000:.0f}) nEnemy={len(self.enemy_fighters())} msl={len(self.our_msls)}")

    def _boundary(self, f, az):
        """Keep inside the battlefield: |y| < dOut - margin, -dLine+margin < x < dLine - margin."""
        m = self.P['boundary_margin']
        p = f['pos']
        d = dir_from_az(az)
        lim = [(1, self.dOut - m, +1), (1, self.dOut - m, -1), (0, self.dLine - m, +1), (0, self.dLine - m, -1)]
        for ax, L, sgn in lim:
            over = sgn * p[ax] - (L - 15000.0)
            if over > 0 and sgn * d[ax] > 0:
                # blend towards parallel/inward
                w = min(1.0, over / 15000.0)
                d[ax] = d[ax] * (1 - w) - sgn * 0.3 * w
                d = unit(d)
        return az_of(d)

    def _command(self, f, az, alt, mode, thr, launch, target):
        m = f['motion']
        dteam = dir_from_az(az)
        dloc = self.L(dteam)
        keeper = self.altKfast if mode == 'fast' else self.altK
        dloc = np.array(keeper(m, dloc, alt))
        # energy management: climb angle limited by current speed (avoid bleeding speed in climbs)
        n = sqrt(dloc[0] ** 2 + dloc[1] ** 2)
        pitch = atan2(-dloc[2], n)
        maxp = radians(float(np.clip((f['V'] - self.P['min_speed']) / 150.0 * 8.0, 0.0, 8.0)))
        if pitch > maxp:
            pitch = maxp
            dloc = np.array([dloc[0] / n * cos(pitch), dloc[1] / n * cos(pitch), -sin(pitch)])
        cmd_dir = f['omotion'].dirAtoP(dloc, m.pos(), self.getLocalCRS())
        self.commands[f['name']] = {
            'motion': {'dstDir': cmd_dir, 'dstThrottle': float(thr)},
            'weapon': {'launch': bool(launch), 'target': target.to_json()},
        }

    def control(self):
        # commands are re-issued every tick in deploy (step interval = 1 tick); clear launch flag
        for nm, c in list(self.commands.items()):
            pass
