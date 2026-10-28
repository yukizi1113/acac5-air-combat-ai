"""Hierarchical tactical agent for ACAC5 (M-mode: one Agent instance controls all 4 fighters).

Layers
  1. World model  : radar Track3D (team-shared) + IRST Track2D bearings (triangulated, or projected
                    onto the known enemy-escort racetrack) + MWS warnings + own missiles + own escort.
  2. Options      : per-fighter macro-actions (ATTACK escort / ENGAGE fighter / DEFEND own escort /
                    DRAG / FLANK-L / FLANK-R). Chosen every `decision_interval` ticks by the
                    rule-based commander or by an external (RL) policy through the action interface.
  3. Reflex layer : always-on behaviours that override options: missile evasion on MWS,
                    missile support (crank) while own missile is still mother-guided,
                    threat-driven drag, battlefield boundary keeping, energy management.
  4. Fire control : every tick, candidate shots are scored by estimated hit probability
                    (heuristic or learned model `pk_model`), de-conflicted per target.
Team frame: x = towards the enemy, y = right, z = down, origin = battlefield centre.
"""
import math
import os
import numpy as np
from math import atan2, cos, sin, sqrt, radians
from gymnasium import spaces

from ASRCAISim1.core import (  # type: ignore
    Agent, MotionState, Track3D, Track2D, AltitudeKeeper, getValueFromJsonKRD,
)
from BasicAgentUtility.util import TeamOrigin, calcRHead, calcRTail, calcRHeadE, calcRTailE  # type: ignore

# ------------------------------------------------------------------ options
OPT_ATTACK, OPT_ENGAGE, OPT_DEFEND, OPT_DRAG, OPT_FLANK_L, OPT_FLANK_R = range(6)
NUM_OPTIONS = 6
OPT_NAMES = ['ATTACK', 'ENGAGE', 'DEFEND', 'DRAG', 'FLANK_L', 'FLANK_R']
# effective behaviours (option or reflex)
B_NAMES = OPT_NAMES + ['EVADE', 'SUPPORT', 'TDRAG', 'NOAMMO', 'VDRAG']
B_EVADE, B_SUPPORT, B_TDRAG, B_NOAMMO, B_VDRAG = 6, 7, 8, 9, 10

ESCORT_X = 120000.0
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


def dir_from_az(az):
    return np.array([cos(az), sin(az), 0.0])


class EnemyTrack:
    __slots__ = ('uid', 'pos', 'vel', 't', 'track', 'src', 'escort_score', 'is_escort', 'first_t',
                 'irst', 'q', 'dead')

    def __init__(self, uid, t):
        self.uid = uid
        self.pos = np.zeros(3)
        self.vel = np.zeros(3)
        self.t = -1e9          # last update time
        self.track = None      # current radar Track3D (local CRS) -> needed for launching
        self.src = 'none'      # radar | irst | memory
        self.escort_score = 0.0
        self.is_escort = False
        self.first_t = t
        self.irst = []
        self.q = 0             # quality of last fix (2 radar/tri, 1 projected, 0 bearing+old range)
        self.dead = False


class TacAgent(Agent):
    # ============================================================ setup
    def initialize(self):
        super().initialize()
        mc = self.modelConfig
        g = lambda k, d: getValueFromJsonKRD(mc, k, self.randomGen, d)
        self.own = self.getTeam()
        self.P = {
            # flight
            'attack_alt': g('attack_alt', 12000.0),
            'cruise_alt': g('cruise_alt', 11000.0),
            'evade_alt': g('evade_alt', 3000.0),
            'drag_alt': g('drag_alt', 9000.0),
            'evade_pitch_deg': g('evade_pitch_deg', 35.0),
            'evade_mode': g('evade_mode', 'drag_level'),
            'min_speed': g('min_speed', 300.0),
            'climb_v0': g('climb_v0', 300.0),      # no climbing below this speed (accelerate first)
            'climb_dv': g('climb_dv', 150.0),
            'climb_pmax': g('climb_pmax', 8.0),
            'boundary_margin': g('boundary_margin', 12000.0),
            # fire control (normalised range RTail->0, RHead->1)
            'k_escort': g('k_escort', 0.9),
            'escort_max_dist': g('escort_max_dist', 70000.0),
            'k_fighter': g('k_fighter', 0.45),
            'fighter_max_dist': g('fighter_max_dist', 45000.0),
            'pk_thr_escort': g('pk_thr_escort', 0.6),
            'race_tte': g('race_tte', 300.0),           # raider time-to-our-escort below which attackers ignore virtual threats
            'inbound_deg': g('inbound_deg', 35.0),      # enemy velocity within this angle of our escort -> inbound raider
            'n_interceptors': g('n_interceptors', 2),
            'pk_thr_fighter': g('pk_thr_fighter', 0.65),
            'max_msl_escort': g('max_msl_escort', 1),
            'max_msl_fighter': g('max_msl_fighter', 1),
            'use_pk_model': g('use_pk_model', True),
            # support / threat reaction
            'crank_margin_deg': g('crank_margin_deg', 12.0),
            'crank_max_deg': g('crank_max_deg', 55.0),
            'mws_hold': g('mws_hold', 5.0),
            'threat_k': g('threat_k', -9.0),         # (legacy proximity drag; disabled by default)
            'use_vthreat': g('use_vthreat', True),   # virtual-missile threat estimation
            'vt_k': g('vt_k', 0.40),                 # assumed enemy launch threshold (normalised range)
            'vt_interval': g('vt_interval', 30.0),   # assumed enemy firing cadence per (enemy, us) pair [s]
            'vt_max': g('vt_max', 3),
            'vt_react_dist': g('vt_react_dist', 22000.0),
            'vt_speed_margin': g('vt_speed_margin', 60.0),
            'vt_react_tgo': g('vt_react_tgo', 0.0),     # turn cold when a virtual missile is < this many seconds out
            'alt_floor': g('alt_floor', 2500.0),
            'threat_hot_deg': g('threat_hot_deg', 45.0),
            'drag_min_time': g('drag_min_time', 20.0),
            'drag_release_k': g('drag_release_k', 1.1),
            'support_abort_k': g('support_abort_k', 0.25),
            'scoot': g('scoot', False),
            'flank_attack': g('flank_attack', False),
            'race_pk_escort': g('race_pk_escort', 0.6),   # = pk_thr_escort -> disabled
            'race_salvo': g('race_salvo', 1),
            'urgent_dist': g('urgent_dist', 90000.0),
            'racetrack_intercept': g('racetrack_intercept', True),
            'shot_range_plan': g('shot_range_plan', 45000.0),
            'scoot_k': g('scoot_k', 0.7),
            # strikers: the fastest fighters (spec-randomised vMax / missile thrust) race for the enemy escort
            'n_strikers': g('n_strikers', 0),
            'striker_alt': g('striker_alt', 13500.0),
            'striker_ignore_threats': g('striker_ignore_threats', True),
            'striker_mws_evade': g('striker_mws_evade', True),
            'striker_fighter_shots': g('striker_fighter_shots', True),
            'striker_rn': g('striker_rn', 0.1),
            # datalink guidance: a missile stays guided by any team-mate's radar track, so the shooter may turn cold
            # right after launch while a team-mate further back illuminates (lab: 15/16 escort hits with drag+helper)
            'dl_drag': g('dl_drag', False),
            'dl_drag_escort': g('dl_drag_escort', False),
            'esc_rn_dl': g('esc_rn_dl', -9.0),         # escort shot whenever rn <= this and a team-mate can keep it tracked
            'esc_dl_max_dist': g('esc_dl_max_dist', 100000.0),
            'dl_range_frac': g('dl_range_frac', 0.75),
            'dl_for_margin_deg': g('dl_for_margin_deg', 8.0),          # escort shot inside this normalised range (0 = RTail) regardless of Pk model
            'striker_max_dist': g('striker_max_dist', 110000.0),
            # commander
            'defend_radius': g('defend_radius', 80000.0),
            'n_defenders_max': g('n_defenders_max', 1),
            'n_goalkeepers': g('n_goalkeepers', 0),
            'gk_x': g('gk_x', -70000.0),
            'cap_legs': g('cap_legs', False),
            'cap_leg': g('cap_leg', 12000.0),
            # turtle phase: everyone defends (intercepts incoming fighters head-on) until turtle_t or until
            # turtle_kills enemy fighters are believed dead, then the normal commander takes over
            'turtle_t': g('turtle_t', 0.0),
            'turtle_kills': g('turtle_kills', 2),
            'turtle_cap_x': g('turtle_cap_x', -60000.0),
            'turtle_radius': g('turtle_radius', 110000.0),   # enemy fighters within this of our escort are engaged
            'turtle_per_target': g('turtle_per_target', 2),
            'engage_radius': g('engage_radius', 25000.0),
            'flank_y': g('flank_y', 120000.0),
            'decision_interval': g('decision_interval', 5),
            'rl_mode': g('rl_mode', False),
            'fire_delta': g('fire_delta', [0.25, 0.0, -0.15]),   # pk-threshold shift per fire mode (hold/normal/aggressive)
            'debug': g('debug', False),
            'debug_raise_after': g('debug_raise_after', -1),
            'log_shots': g('log_shots', ''),
        }
        self.altK = AltitudeKeeper({'pGain': -0.3, 'dGain': -0.1, 'minPitch': -10.0, 'maxPitch': 10.0})
        self.altKfast = AltitudeKeeper({'pGain': -0.3, 'dGain': -0.1, 'minPitch': -self.P['evade_pitch_deg'], 'maxPitch': 15.0})
        self.enemies = {}
        self.ee_pos = np.array([ESCORT_X, 0.0, -ESCORT_ALT])
        self.ee_vel = np.zeros(3)
        self.EE = None
        self.opt = {}
        self.opt_tgt = {}
        self.beh = {}
        self.last_mws_t = {}
        self.evade_dir = {}
        self.drag_from = {}       # name -> (t_start, uid)
        self.pending = {}         # name -> (t, uid)
        self.msl_target = {}
        self.msl_owner = {}
        self.shot_feats = {}      # missile name -> feature dict (for data collection)
        self.pre_shots = {}       # shooter name -> features of commanded shot awaiting launch
        self.ext_action = None
        self.fire_mode = {}
        self.teacher = None
        self.tick = 0
        self.prev_msl_alive = {}
        self.strikers = set()
        self.illum = {}           # illuminator name -> missile name it keeps guided
        self.pk_model = None
        if self.P['use_pk_model']:
            try:
                from .pk_model import PkModel
                pth = os.path.join(os.path.dirname(__file__), 'pk_model.json')
                if os.path.exists(pth):
                    self.pk_model = PkModel(pth)
            except Exception:
                self.pk_model = None

    def validate(self):
        r = self.manager.getRuler()().observables()
        self.dOut = r['dOut']
        self.dLine = r['dLine']
        self.teamOrigin = TeamOrigin(self.own == r['eastSider'], self.dLine)
        self.names = [p.getFullName() for p in self.parents.values()]
        self.ports0 = list(self.parents.keys())[0]
        from .vthreat import VirtualThreat
        self.vthreat = VirtualThreat(self.P['vt_k'], self.P['vt_interval'], int(self.P['vt_max']), bool(self.P['use_vthreat']))
        self.vthreat.create(self)

    # ============================================================ frames
    def T(self, v):
        return np.array(self.teamOrigin.relPtoB(np.asarray(v, dtype=np.float64)))

    def L(self, v):
        return np.array(self.teamOrigin.relBtoP(np.asarray(v, dtype=np.float64)))

    # ============================================================ world model
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
                f['thrust'] = spec['weapon']['missile']['thrust']
                f['nmsl0'] = spec['weapon']['numMsls']
                f['rcs'] = spec['stealth']['rcsScale']
                f['vmaxs'] = spec['dynamics']['scale']['vMax']
                w = o['weapon']()
                f['rem'] = int(w['remMsls'])
                f['launchable'] = bool(w['launchable'])
                f['msls'] = w.get('missiles', [])
                f['mws'] = []
                if o.contains_p('/sensor/mws/track'):
                    for mo in o.at_p('/sensor/mws/track'):
                        f['mws'].append(self.T(Track2D(mo).transformTo(lc).dir()))
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
        self.alive = [f for f in F if f['alive']]
        # own escort
        self.myE = None
        if first is not None:
            for nm, fo in first.observables.at_p('/shared/fighter').items():
                if nm not in self.names and fo['isAlive']():
                    m = MotionState(fo['motion']).transformTo(lc)
                    self.myE = {'pos': self.T(m.pos()), 'vel': self.T(m.vel())}
        # ---- sensor fusion (radar Track3D + anonymous IRST bearings), see fusion.py
        from .fusion import Fusion
        if not hasattr(self, 'fusion'):
            self.fusion = Fusion()
        radar, rtr = [], {}
        if first is not None:
            for tj in first.observables.at_p('/sensor/track'):
                tr = Track3D(tj).transformTo(lc)
                uid = str(tr.truth)
                rtr[uid] = tr
                radar.append((uid, self.T(tr.pos()), self.T(tr.vel())))
        bearings = []
        for f in self.alive:
            for anon, d, org in f['irst']:
                bearings.append((f['name'], anon, org, d))
        known = self.EE.uid if self.EE is not None else None
        self.fusion.irst_L = {f['name']: f['irstL'] for f in self.alive}
        seen = self.fusion.step(t, radar, bearings, [f['pos'] for f in self.alive], known)
        irst_by = {}
        for (fname, anon), tid in self.fusion.assoc.items():
            irst_by.setdefault(tid, []).append(fname)
        self.enemies = self.fusion.tracks
        for tid, e in self.enemies.items():
            e.track = rtr.get(tid) if tid in seen else None
            e.src = 'radar' if tid in seen else ('irst' if e.t_irst >= t else 'memory')
            e.irst = irst_by.get(tid, [])
        self._classify()
        # own missiles
        self.our_msls = []
        nlaunched = 0
        for f in F:
            for mo in f['msls']:
                if not mo.get('hasLaunched', False):
                    continue
                nlaunched += 1
                nm = mo['fullName']
                if nm not in self.msl_target:
                    tgt = Track3D(mo['target']).transformTo(lc)
                    self.msl_target[nm] = str(tgt.truth)
                    self.msl_owner[nm] = f['name']
                    feats = self.pre_shots.pop(f['name'], None)
                    if feats is not None:
                        self.shot_feats[nm] = feats
                        self._log_shot(nm, feats)
                if mo['isAlive']:
                    mm = MotionState(mo['motion']).transformTo(lc)
                    tgt = Track3D(mo['target']).transformTo(lc)
                    self.our_msls.append({'name': nm, 'owner': f['name'], 'uid': self.msl_target[nm],
                                          'pos': self.T(mm.pos()), 'vel': self.T(mm.vel()),
                                          'mode': mo['mode'], 'tpos': self.T(tgt.pos())})
        self.n_launched = nlaunched
        # kill inference: a missile in terminal (SELF) mode vanished within 3 km of its target and the
        # target's radar track disappeared at the same time -> assume destroyed
        alive_now = {m['name']: m for m in self.our_msls}
        for nm, pm in self.prev_msl_alive.items():
            if nm not in alive_now and pm['mode'] == 'SELF':
                e = self.enemies.get(pm['uid'])
                if e is not None and e.track is None and norm(pm['pos'] - pm['tpos']) < 3000:
                    e.dead = True
        self.prev_msl_alive = alive_now

    def _irst_fix(self, lst, e):
        if len(lst) >= 2:
            A = np.zeros((3, 3)); b = np.zeros(3)
            ds = []
            for org, d in lst:
                d = unit(d); ds.append(d)
                M = np.eye(3) - np.outer(d, d)
                A += M; b += M @ org
            ds = np.array(ds)
            spread = np.linalg.norm(ds - ds.mean(0), axis=1).max()
            if spread > 0.03:
                try:
                    p = np.linalg.solve(A + 1e-9 * np.eye(3), b)
                    r = norm(p - lst[0][0])
                    if 2000 < r < 300000 and all(unit(p - o) @ unit(d) > 0.99 for o, d in lst):
                        return p, 2
                except np.linalg.LinAlgError:
                    pass
        org, d = lst[0]
        d = unit(d)
        if e.is_escort or e.escort_score >= 1.0:
            if d[0] > 0.05:
                p = org + (ESCORT_X - org[0]) / d[0] * d
                p[2] = -ESCORT_ALT
                return p, 1
        if e.t > -1e8:
            return org + d * norm(e.pos - org), 0
        # unknown range: assume fighter at a typical IRST detection range
        return org + d * 100000.0, 0

    def _classify(self):
        for e in self.enemies.values():
            if e.is_escort:
                continue
            if getattr(e, 'escort_seed', False):
                e.escort_score = max(e.escort_score, 1.0)    # IR-visible from t=0 -> only the escort is that bright
            if e.src == 'radar':
                dmin = min([norm(f['pos'] - e.pos) for f in self.alive] or [0])
                if dmin > 100000:
                    e.escort_score += 2.0                     # radar detection beyond any fighter's range
                V = norm(e.vel)
                if abs(e.pos[0] - ESCORT_X) < 12000 and V < 320 and abs(-e.pos[2] - ESCORT_ALT) < 1500 and abs(e.vel[0]) < 0.5 * V:
                    e.escort_score += 0.5
                elif V > 360:
                    e.escort_score -= 1.0
            if e.escort_score >= 2.0:
                e.is_escort = True
        esc = sorted([e for e in self.enemies.values() if e.is_escort], key=lambda e: -e.escort_score)
        for e in esc[1:]:
            e.is_escort = False; e.escort_score = 0.0
        self.EE = esc[0] if esc else None
        if self.EE is None:
            cands = [e for e in self.enemies.values() if e.escort_score >= 1.0 and e.src == 'irst']
            self.ee_known = bool(cands)
            if cands:
                c = max(cands, key=lambda e: e.escort_score)
                self.ee_pos = c.pos.copy(); self.ee_vel = c.vel.copy()
        else:
            E = self.EE
            self.ee_known = self.now - E.t < 60
            self.ee_pos = E.pos + (E.vel * min(30.0, self.now - E.t) if E.src != 'radar' else 0.0)
            self.ee_vel = E.vel.copy()

    # ============================================================ helpers
    def fighters_e(self, max_age=30.0):
        return [e for e in self.enemies.values() if not e.is_escort and not e.dead and self.now - e.t <= max_age
                and e.escort_score < 1.0 and not (e.tentative and e.src != 'radar' and self.now - e.t_created < 3)]

    def msls_on(self, uid):
        return [m for m in self.our_msls if m['uid'] == uid]

    def ranges(self, f, e):
        """Our missile RHead/RTail vs e and e's vs us (needs a radar track)."""
        tr = e.track
        if tr is None:
            return None
        try:
            rh = float(calcRHead(f['acc'], f['motion'], tr, False))
            rt = float(calcRTail(f['acc'], f['motion'], tr, False))
            rhe = float(calcRHeadE(f['acc'], f['motion'], tr, False))
            rte = float(calcRTailE(f['acc'], f['motion'], tr, False))
            return rh, rt, rhe, rte
        except Exception:
            return None

    def threat_level(self, f, e):
        """(hot, kE, d): kE = normalised range of e's missile against f (0=RTail, 1=RHead)."""
        rel = f['pos'] - e.pos
        d = norm(rel)
        V = norm(e.vel)
        hot = V > 50 and (unit(e.vel) @ unit(rel)) > cos(radians(self.P['threat_hot_deg']))
        rg = self.ranges(f, e)
        if rg is not None:
            rh, rt, rhe, rte = rg
            kE = (d - rte) / max(1.0, rhe - rte)
        else:
            kE = (d - 40000.0) / 60000.0   # crude estimate when only IRST/memory
        return hot, kE, d

    def intercept_point(self, f, tp, tv):
        vown = max(f['V'], 350.0)
        rel = tp - f['pos']
        a = tv @ tv - vown * vown; b = 2 * rel @ tv; c = rel @ rel
        tt = None
        if abs(a) > 1e-6:
            disc = b * b - 4 * a * c
            if disc >= 0:
                cands = [r for r in ((-b - sqrt(disc)) / (2 * a), (-b + sqrt(disc)) / (2 * a)) if r > 0]
                tt = min(cands) if cands else None
        if tt is None:
            tt = norm(rel) / vown
        return tp + tv * min(tt, 400.0)

    def off_bore(self, f, p):
        return abs(wrap(az_of(p - f['pos']) - f['az']))

    # ============================================================ commander (rule based)
    def rule_commander(self):
        """Rule commander.
        1. Raiders: enemy fighters whose velocity points at our escort (within inbound_deg) and that are
           still on our escort's far side are intercepted *head-on* by the friend that can meet them
           earliest from in front (a tail chase has poor Pk, see engagement lab).
        2. Close enemy fighters (< engage_radius) are engaged by the nearest friend.
        3. Everyone else attacks the enemy escort."""
        P = self.P
        opts, tg = {}, {}
        alive = [f for f in self.alive if f['rem'] > 0]
        enemies = self.fighters_e(20.0)
        used = set()
        if self.now < P['turtle_t'] and self._n_kills() < P['turtle_kills'] and self.myE is not None:
            return self._turtle(alive, enemies)
        ns = int(P['n_strikers'])
        if ns > 0:
            names_alive = {f['name'] for f in alive}
            self.strikers = {n for n in self.strikers if n in names_alive}
            if len(self.strikers) < ns:
                rank = sorted([f for f in alive if f['name'] not in self.strikers],
                              key=lambda f: -(f['vmaxs'] * 1.0 + f['thrust'] / 30018.0 * 0.5 + 0.05 * f['rem']))
                for f in rank[:ns - len(self.strikers)]:
                    self.strikers.add(f['name'])
            for f in alive:
                if f['name'] in self.strikers:
                    opts[f['name']] = OPT_ATTACK; tg[f['name']] = None; used.add(f['name'])
        if self.myE is not None:
            E0 = self.myE['pos']
            raiders = []
            for e in enemies:
                V = norm(e.vel[:2])
                if V < 100:
                    continue
                to_e = E0[:2] - e.pos[:2]
                d = norm(to_e)
                cosang = float(to_e @ e.vel[:2]) / max(1.0, d * V)
                if cosang > cos(radians(P['inbound_deg'])) or d < P['defend_radius']:
                    raiders.append((d / V, e))       # time for the raider to reach our escort
            raiders.sort(key=lambda x: x[0])
            self.race_mode = bool(raiders) and raiders[0][0] < P['race_tte']
            for tte, e in raiders[:int(P['n_interceptors'])]:
                best, bscore = None, 1e9
                for f in alive:
                    if f['name'] in used:
                        continue
                    # in front of the raider (between it and our escort) -> head-on geometry
                    rel = f['pos'][:2] - e.pos[:2]
                    ahead = float(rel @ e.vel[:2]) > 0
                    score = norm(rel) / max(300.0, f['V']) + (0.0 if ahead else 120.0)
                    if score < bscore:
                        best, bscore = f, score
                if best is not None and bscore < tte + 60.0:
                    opts[best['name']] = OPT_DEFEND; tg[best['name']] = e.uid; used.add(best['name'])
        ngk = int(P['n_goalkeepers']) - sum(1 for v in opts.values() if v == OPT_DEFEND)
        if ngk > 0 and self.myE is not None:
            cands = sorted([f for f in alive if f['name'] not in used], key=lambda f: f['pos'][0])
            for f in cands[:ngk]:
                opts[f['name']] = OPT_DEFEND; tg[f['name']] = None; used.add(f['name'])
        for f in alive:
            if f['name'] in used:
                continue
            near = [(norm(e.pos - f['pos']), e) for e in enemies]
            near = [x for x in near if x[0] < P['engage_radius']]
            if near:
                d, e = min(near, key=lambda x: x[0])
                opts[f['name']] = OPT_ENGAGE; tg[f['name']] = e.uid
            else:
                if P['flank_attack'] and f['pos'][0] < 20000:
                    opts[f['name']] = OPT_FLANK_R if f['pos'][1] >= 0 else OPT_FLANK_L
                else:
                    opts[f['name']] = OPT_ATTACK
                tg[f['name']] = None
        return opts, tg

    def _n_kills(self):
        return sum(1 for e in self.enemies.values() if e.dead and not e.is_escort)

    def _turtle(self, alive, enemies):
        P = self.P
        opts, tg = {}, {}
        E0 = self.myE['pos']
        thr = [e for e in enemies if norm(e.pos[:2] - E0[:2]) < P['turtle_radius']]
        thr.sort(key=lambda e: norm(e.pos[:2] - E0[:2]))
        load = {e.uid: 0 for e in thr}
        for f in sorted(alive, key=lambda f: min([norm(f['pos'] - e.pos) for e in thr] or [0.0])):
            best = None
            for e in thr:
                if load[e.uid] >= P['turtle_per_target']:
                    continue
                d = norm(f['pos'] - e.pos)
                if best is None or d < best[0]:
                    best = (d, e)
            if best is not None:
                opts[f['name']] = OPT_DEFEND; tg[f['name']] = best[1].uid; load[best[1].uid] += 1
            else:
                opts[f['name']] = OPT_DEFEND; tg[f['name']] = None
        return opts, tg

    def apply_external(self, action):
        """action: option id per parent (from an RL policy). Targets chosen by heuristics."""
        opts, tg = {}, {}
        enemies = self.fighters_e(20.0)
        for i, f in enumerate(self.F):
            if not f['alive']:
                continue
            o = int(action[i]) if action is not None and i < len(action) else OPT_ATTACK
            opts[f['name']] = o
            if o in (OPT_ENGAGE, OPT_DRAG) and enemies:
                e = min(enemies, key=lambda e: norm(e.pos - f['pos']))
                tg[f['name']] = e.uid
            elif o == OPT_DEFEND and enemies and self.myE is not None:
                e = min(enemies, key=lambda e: norm(e.pos - self.myE['pos']))
                tg[f['name']] = e.uid
            else:
                tg[f['name']] = None
        return opts, tg

    # ============================================================ skills -> (az, alt, mode, throttle)
    def escort_future(self, tau):
        """Predicted enemy-escort position after tau seconds, propagated along its known capsule racetrack
        (straight legs along y at x = 120 km +- 5 km, turn-arounds at y = +-100 km), not by straight-line extrapolation."""
        p, v = self.ee_pos, self.ee_vel
        V = max(200.0, norm(v[:2])) if norm(v[:2]) > 50 else 250.0
        if norm(v[:2]) < 50:
            return p.copy()
        x, y = float(p[0]), float(p[1])
        sy = 1.0 if v[1] >= 0 else -1.0
        if abs(v[0]) > abs(v[1]):          # in a turn: treat as at the nearer end
            sy = -1.0 if y > 0 else 1.0
        s_left = V * tau
        yend = 100000.0
        turn = math.pi * 5000.0
        for _ in range(6):
            dist_end = max(0.0, (sy * yend - y) * sy)
            if s_left <= dist_end:
                y += sy * s_left
                s_left = 0.0
                break
            s_left -= dist_end
            y = sy * yend
            if s_left <= turn:
                frac = s_left / turn
                x = ESCORT_X + (x - ESCORT_X) * math.cos(math.pi * frac)
                s_left = 0.0
                break
            s_left -= turn
            x = 2 * ESCORT_X - x
            sy = -sy
        return np.array([x, y, p[2]])

    def sk_attack(self, f):
        if self.P['racetrack_intercept'] and getattr(self, 'ee_known', False):
            R = self.P['shot_range_plan']
            vown = max(f['V'], 450.0)
            aim = None
            for tau in range(0, 601, 5):
                q = self.escort_future(float(tau))
                if norm((q - f['pos'])[:2]) - R <= vown * tau:
                    aim = q
                    break
            if aim is None:
                aim = self.escort_future(600.0)
        else:
            aim = self.intercept_point(f, self.ee_pos, self.ee_vel).copy()
        aim = aim.copy()
        aim[0] = min(aim[0], self.dLine - 25000)
        alt = self.P['striker_alt'] if f['name'] in self.strikers else self.P['attack_alt']
        return az_of(aim - f['pos']), alt, 'n', 1.0

    def sk_flank(self, f, side):
        P = self.P
        y = side * P['flank_y']
        tgt = np.array([min(self.ee_pos[0] - 40000, f['pos'][0] + 60000), y, 0])
        if abs(f['pos'][1] - y) < 20000 or f['pos'][0] > 30000:
            return self.sk_attack(f)
        return az_of(tgt - f['pos']), P['attack_alt'], 'n', 1.0

    def sk_engage(self, f, uid):
        e = self.enemies.get(uid) if uid else None
        if e is None:
            return self.sk_attack(f)
        aim = self.intercept_point(f, e.pos, e.vel)
        return az_of(aim - f['pos']), max(self.P['cruise_alt'], min(13000.0, -e.pos[2] + 1000)), 'n', 1.0

    def sk_defend(self, f, uid):
        e = self.enemies.get(uid) if uid else None
        if e is not None and self.now - e.t < 20:
            return self.sk_engage(f, uid)
        base = self.myE['pos'] if self.myE is not None else np.array([-120000.0, 0, -10000])
        cx = self.P['turtle_cap_x'] if self.now < self.P['turtle_t'] else self.P['gk_x']
        cap = np.array([cx, base[1] * 0.7, 0.0])
        if self.P['cap_legs']:
            # east-west legs through the CAP point: the radar looks toward the enemy on the eastbound leg
            dy = cap[1] - f['pos'][1]
            if f['pos'][0] < cap[0] - self.P['cap_leg']:
                hd = 0.0
            elif f['pos'][0] > cap[0] + self.P['cap_leg']:
                hd = math.pi
            else:
                hd = 0.0 if abs(wrap(f['az'])) < math.pi / 2 else math.pi
            v = dir_from_az(hd) * 20000.0 + np.array([0.0, float(np.clip(dy, -15000, 15000)), 0.0])
            return az_of(v), self.P['cruise_alt'], 'n', 0.85
        if norm(cap[:2] - f['pos'][:2]) < 8000:
            return wrap(f['az'] + 0.6), self.P['cruise_alt'], 'n', 0.8
        return az_of(cap - f['pos']), self.P['cruise_alt'], 'n', 1.0

    def sk_drag(self, f, uid):
        e = self.enemies.get(uid) if uid else None
        if e is None:
            es = self.fighters_e(20.0)
            if not es:
                return self.sk_attack(f)
            e = min(es, key=lambda e: norm(e.pos - f['pos']))
        return az_of(f['pos'] - e.pos), self.P['drag_alt'], 'n', 1.0

    def sk_support(self, f, m):
        e = self.enemies.get(m['uid'])
        tp = e.pos if e is not None else m['tpos']
        brg = az_of(tp - f['pos'])
        crank = max(0.0, min(radians(self.P['crank_max_deg']), f['radarFOR'] - radians(self.P['crank_margin_deg'])))
        c1, c2 = wrap(brg + crank), wrap(brg - crank)
        s1 = abs(wrap(c1 - f['az'])) + (0.3 if (f['pos'][1] > 0) == (sin(c1) > 0) else 0.0)
        s2 = abs(wrap(c2 - f['az'])) + (0.3 if (f['pos'][1] > 0) == (sin(c2) > 0) else 0.0)
        alt = self.P['striker_alt'] if f['name'] in self.strikers else self.P['attack_alt']
        return (c1 if s1 < s2 else c2), alt, 'n', 1.0

    def sk_evade(self, f):
        d = self.evade_dir.get(f['name'])
        if d is None:
            return self.sk_attack(f)
        # engagement-lab result: a level drag defeats more terminal-phase missiles than a dive
        if self.P['evade_mode'] == 'drag_level':
            return az_of(-d), max(4000.0, min(f['alt'], 13000.0)), 'n', 1.0
        return az_of(-d), self.P['evade_alt'], 'f', 1.0

    def sk_noammo(self, f):
        base = self.myE['pos'] if self.myE is not None else np.array([-120000.0, 0, -10000])
        tgt = np.array([base[0] + 30000, base[1], 0])
        if norm(tgt[:2] - f['pos'][:2]) < 8000:
            return wrap(f['az'] + 0.6), self.P['cruise_alt'], 'n', 0.7
        return az_of(tgt - f['pos']), self.P['cruise_alt'], 'n', 1.0

    # ============================================================ fire control
    def shot_features(self, f, e, kind):
        rg = self.ranges(f, e)
        if rg is None:
            return None
        rh, rt, rhe, rte = rg
        rel = e.pos - f['pos']
        d = norm(rel)
        u = unit(rel)
        ve = e.vel; vf = f['vel']
        Ve = norm(ve)
        aspect = math.acos(float(np.clip((unit(ve) @ (-u)) if Ve > 1 else 1.0, -1, 1)))   # 0 = target head-on
        ata = math.acos(float(np.clip(unit(vf) @ u, -1, 1)))
        return {
            'd': d, 'rh': rh, 'rt': rt, 'rn': (d - rt) / max(1.0, rh - rt),
            'rhe': rhe, 'rte': rte, 'aspect': aspect, 'ata': ata,
            'alt_s': f['alt'], 'alt_t': -e.pos[2], 'V_s': f['V'], 'V_t': Ve,
            'closing': float(-(ve - vf) @ u), 'thrust': f['thrust'], 'kind': 1.0 if kind == 'escort' else 0.0,
            'dz': (-e.pos[2]) - f['alt'],
        }

    def pk(self, x):
        if self.pk_model is not None:
            try:
                return float(self.pk_model.predict(x))
            except Exception:
                pass
        rn = x['rn']
        if x['kind'] > 0.5:
            p = 0.95 if rn <= 0.45 else max(0.0, 0.95 - (rn - 0.45) * 2.0)
            if x['d'] > 70000:
                p *= 0.6
            return p
        p = 0.85 - 1.2 * max(0.0, rn - 0.1)
        p *= 1.0 - 0.4 * min(1.0, x['aspect'] / math.pi)
        return max(0.0, p)

    def fire_control(self, f, beh):
        P = self.P
        if not f['launchable'] or f['rem'] <= 0 or beh == B_EVADE:
            return None
        if f['name'] in self.pending and self.now - self.pending[f['name']][0] < 7:
            return None
        cands = []
        E = self.EE
        urgent = self._escort_urgent()
        max_esc = int(P['race_salvo']) if urgent else int(P['max_msl_escort'])
        thr_esc = P['race_pk_escort'] if urgent else P['pk_thr_escort']
        if E is not None and E.track is not None and len(self.msls_on(E.uid)) < max_esc and not self._pend_on(E.uid):
            if self.off_bore(f, E.pos) < f['radarFOR'] - 0.1:
                x = self.shot_features(f, E, 'escort')
                dmax = P['escort_max_dist'] + (15000.0 if urgent else 0.0)
                if x is not None and x['rn'] <= P['esc_rn_dl'] and x['d'] < P['esc_dl_max_dist'] and \
                        (self._illuminator(f, E) is not None or x['rn'] <= 0.0):
                    p = max(0.5, self.pk(x))
                    cands.append((p + 1.0, E, 'escort', x, p))
                elif x is not None and f['name'] in self.strikers and x['d'] < P['striker_max_dist'] and x['rn'] <= P['striker_rn']:
                    p = max(0.5, self.pk(x))
                    cands.append((p + 1.0, E, 'escort', x, p))
                elif x is not None and x['d'] < dmax and (x['rn'] <= P['k_escort'] or x['d'] < 25000):
                    p = self.pk(x)
                    if p >= thr_esc + self._fd(f):
                        cands.append((p + 1.0, E, 'escort', x, p))
        no_ftr = f['name'] in self.strikers and not P['striker_fighter_shots']
        for e in ([] if no_ftr else self.fighters_e(0.5)):
            if e.track is None or len(self.msls_on(e.uid)) >= P['max_msl_fighter'] or self._pend_on(e.uid):
                continue
            if self.off_bore(f, e.pos) > f['radarFOR'] - 0.1:
                continue
            x = self.shot_features(f, e, 'fighter')
            if x is None or x['d'] > P['fighter_max_dist'] or x['rn'] > P['k_fighter']:
                continue
            p = self.pk(x)
            if p >= P['pk_thr_fighter'] + self._fd(f):
                cands.append((p, e, 'fighter', x, p))
        if not cands:
            return None
        cands.sort(key=lambda c: -c[0])
        _, e, kind, x, p = cands[0]
        return e, kind, x, p

    def _escort_urgent(self):
        """Race-aware fire control: an enemy fighter is close enough to our escort that it may shoot it soon."""
        if self.myE is None or self.P['race_pk_escort'] >= self.P['pk_thr_escort']:
            return False
        for e in self.fighters_e(15.0):
            if norm(e.pos[:2] - self.myE['pos'][:2]) < self.P['urgent_dist']:
                return True
        return False

    def _fd(self, f):
        return float(self.P['fire_delta'][int(self.fire_mode.get(f['name'], 1))])

    def _pend_on(self, uid):
        return any(u == uid and self.now - t0 < 7 for t0, u in self.pending.values())

    def _log_shot(self, msl_name, feats):
        path = self.P['log_shots'] or os.environ.get('SHOT_LOG', '')
        if path:
            try:
                import json
                rec = {'msl': msl_name, 'match': os.environ.get('MATCH_ID', '')}
                for k, v in feats.items():
                    rec[k] = float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v
                with open(path, 'a') as fo:
                    fo.write(json.dumps(rec) + '\n')
            except Exception:
                pass

    # ============================================================ gym interface
    LOG_DIM = 160

    def observation_space(self):
        if not self.P['rl_mode']:
            return spaces.Box(low=-np.inf, high=np.inf, shape=(self.LOG_DIM,), dtype=np.float32)
        from .rl_obs import G_DIM, F_DIM, E_DIM, S_DIM, MAX_F, MAX_E
        B = lambda *sh: spaces.Box(low=-np.inf, high=np.inf, shape=sh, dtype=np.float32)
        return spaces.Dict({'g': B(G_DIM), 'f': B(MAX_F, F_DIM), 'e': B(MAX_E, E_DIM), 's': B(2, S_DIM),
                            'fmask': B(MAX_F), 'emask': B(MAX_E), 'decide': B(1), 'teacher': B(2 * MAX_F)})

    def action_space(self):
        return spaces.MultiDiscrete([NUM_OPTIONS] * len(self.parents) + [3] * len(self.parents))

    def _log_vector(self):
        """Compact situation snapshot (team frame, km / deg) written into the competition's match logs:
        [t, friends 4x10, own escort 3, enemy tracks 8x8, own missiles 8x6, n_launched, enemy-escort estimate x,y]."""
        v = np.zeros(self.LOG_DIM, dtype=np.float32)
        v[0] = self.now
        o = 1
        for f in self.F[:4]:
            if f['alive']:
                v[o:o + 10] = [1, f['pos'][0] / 1e3, f['pos'][1] / 1e3, f['alt'] / 1e3, f['V'], math.degrees(f['az']),
                               f['rem'], self.beh.get(f['name'], -1), 1 if f['mws'] else 0, 1 if f['launchable'] else 0]
            o += 10
        o = 41
        if self.myE is not None:
            v[o:o + 3] = [1, self.myE['pos'][0] / 1e3, self.myE['pos'][1] / 1e3]
        o = 44
        ents = sorted(self.enemies.values(), key=lambda e: (not e.is_escort, self.now - e.t))[:8]
        for e in ents:
            V = norm(e.vel)
            src = {'radar': 1, 'irst': 2, 'memory': 3}.get(e.src, 0)
            v[o:o + 8] = [1 if not e.dead else -1, e.pos[0] / 1e3, e.pos[1] / 1e3, -e.pos[2] / 1e3, V,
                          math.degrees(az_of(e.vel)) if V > 1 else 0, src, 1 if e.is_escort else 0]
            o += 8
        o = 108
        owners = {f['name']: i for i, f in enumerate(self.F)}
        for m in self.our_msls[:8]:
            v[o:o + 6] = [1, m['pos'][0] / 1e3, m['pos'][1] / 1e3, -m['pos'][2] / 1e3, owners.get(m['owner'], -1),
                          {'GUIDED': 1, 'SELF': 2, 'MEMORY': 3}.get(m['mode'], 0)]
            o += 6
        v[156] = self.n_launched
        v[157] = self.ee_pos[0] / 1e3
        v[158] = self.ee_pos[1] / 1e3
        return np.round(v, 2)

    def is_decision_tick(self):
        return (self.tick - 1) % int(self.P['decision_interval']) == 0

    def makeObs(self):
        self.tick += 1
        try:
            self._build_world()
            self._world_ok = True
        except Exception as ex:
            import traceback
            self._world_ok = False
            self._last_error = traceback.format_exc()[-600:]
            if self.P['rl_mode']:
                from .rl_obs import G_DIM, F_DIM, E_DIM, S_DIM, MAX_F, MAX_E
                z = lambda *sh: np.zeros(sh, np.float32)
                return {'g': z(G_DIM), 'f': z(MAX_F, F_DIM), 'e': z(MAX_E, E_DIM), 's': z(2, S_DIM), 'fmask': z(MAX_F),
                        'emask': z(MAX_E), 'decide': np.zeros(1, np.float32), 'teacher': z(2 * MAX_F)}
            return np.zeros(self.LOG_DIM, dtype=np.float32)
        if not self.P['rl_mode']:
            try:
                return self._log_vector()
            except Exception:
                return np.zeros(self.LOG_DIM, dtype=np.float32)
        from .rl_obs import build_obs
        o = build_obs(self)
        dec = self.is_decision_tick()
        o['decide'] = np.array([1.0 if dec else 0.0], np.float32)
        tl = np.zeros(2 * len(self.F), np.float32)
        if dec:
            self.teacher = self.rule_commander()
            opts, _ = self.teacher
            for i, f in enumerate(self.F):
                tl[i] = opts.get(f['name'], OPT_ATTACK)
                tl[len(self.F) + i] = 1
        o['teacher'] = tl
        return o

    def reflexes(self, f, opt, otgt):
        """Effective behaviour (+payload) after safety / support overrides."""
        P = self.P
        nm = f['name']
        striker = nm in self.strikers and opt == OPT_ATTACK
        if self.now - self.last_mws_t.get(nm, -1e9) <= P['mws_hold'] and (not striker or P['striker_mws_evade']):
            return B_EVADE, None
        guided = [m for m in self.our_msls if m['owner'] == nm and m['mode'] == 'GUIDED']
        duty = self.illum.get(nm)
        if duty is not None:
            dm = next((m for m in self.our_msls if m['name'] == duty and m['mode'] == 'GUIDED'), None)
            if dm is None:
                del self.illum[nm]
            elif not guided:
                vt0 = self._vthreat_for(f)
                if vt0 is None or vt0['dist'] > 12000:
                    return B_SUPPORT, dm
        if striker and P['striker_ignore_threats']:
            if f['rem'] <= 0 and not guided:
                return B_NOAMMO, None
            if guided:
                return B_SUPPORT, guided[0]
            return opt, otgt
        vt = self._vthreat_for(f)
        if vt is not None and getattr(self, 'race_mode', False) and opt == OPT_ATTACK:
            vt = None          # escort race: time matters more than a hypothetical missile
        if vt is not None:
            # keep supporting our own missile if it is about to go autonomous
            g_ok = [m for m in guided if norm(m['tpos'] - m['pos']) < 14000]
            if not g_ok:
                return B_VDRAG, vt
        if f['rem'] <= 0 and not guided:
            return B_NOAMMO, None
        worst = None
        for e in self.fighters_e(15.0):
            hot, kE, d = self.threat_level(f, e)
            if hot and kE < P['threat_k']:
                if worst is None or kE < worst[0]:
                    worst = (kE, e, d)
        if guided and P['scoot']:
            # shoot-and-scoot: after a shot at a fighter that is hot on us, turn cold at once. The missile flies on
            # (memory / teammate guidance); a head-on missile the target may have fired is defeated by our tail aspect.
            m = guided[0]
            e = self.enemies.get(m['uid'])
            if e is not None and not e.is_escort and e.escort_score < 1.0:
                hot, kE, d = self.threat_level(f, e)
                if hot and kE < P['scoot_k'] and norm(m['tpos'] - m['pos']) > 12000:
                    return B_TDRAG, e.uid
        if guided and P['dl_drag']:
            m = guided[0]
            e = self.enemies.get(m['uid'])
            is_esc = e is not None and (e.is_escort or e.escort_score >= 1.0)
            if e is not None and (not is_esc or P['dl_drag_escort']):
                il = self._illuminator(f, e)
                if il is not None:
                    self.illum[il['name']] = m['name']
                    if not is_esc:
                        self.drag_from[nm] = (self.now, e.uid)
                        return B_TDRAG, e.uid
                    near = [x for x in self.fighters_e(15.0) if norm(x.pos - f['pos']) < 60000]
                    if near:
                        x = min(near, key=lambda x: norm(x.pos - f['pos']))
                        self.drag_from[nm] = (self.now, x.uid)
                        return B_TDRAG, x.uid
        if guided:
            m = guided[0]
            tgo = norm(m['tpos'] - m['pos']) / max(300.0, norm(m['vel']))
            if worst is None or worst[0] > P['support_abort_k'] or tgo < 10:
                return B_SUPPORT, m
        if nm in self.drag_from:
            t0, uid = self.drag_from[nm]
            e = self.enemies.get(uid)
            release = True
            if e is not None and self.now - e.t < 15:
                hot, kE, d = self.threat_level(f, e)
                release = (not hot) or kE > P['drag_release_k']
            if self.now - t0 < P['drag_min_time'] or not release:
                return B_TDRAG, uid
            del self.drag_from[nm]
        if worst is not None:
            kE, e, d = worst
            if not self.msls_on(e.uid):
                x = self.shot_features(f, e, 'fighter') if e.track is not None else None
                can_shoot = (x is not None and x['rn'] <= P['k_fighter'] and self.pk(x) >= P['pk_thr_fighter']
                             and f['launchable'] and f['rem'] > 0 and self.off_bore(f, e.pos) < f['radarFOR'] - 0.1)
                if not can_shoot:
                    self.drag_from[nm] = (self.now, e.uid)
                    return B_TDRAG, e.uid
        return opt, otgt

    def _illuminator(self, f, e):
        """A team-mate (not f) that has e inside its radar range and field of regard and is at least as far from e
        as f is (so it is not the one being shot at)."""
        if e.track is None:
            return None
        P = self.P
        d_f = norm(e.pos - f['pos'])
        best = None
        for g2 in self.alive:
            if g2['name'] == f['name']:
                continue
            if g2['name'] in self.illum and self.illum[g2['name']] not in {m['name'] for m in self.our_msls}:
                pass
            d2 = norm(e.pos - g2['pos'])
            Lr = g2['radarL'] * (2.0 if (e.is_escort or e.escort_score >= 1.0) else 0.85)
            if d2 > P['dl_range_frac'] * Lr or d2 < 0.8 * d_f:
                continue
            # can it keep e inside its FOR while flying (turn needed is small)?
            if self.off_bore(g2, e.pos) > g2['radarFOR'] - radians(P['dl_for_margin_deg']) + radians(40.0):
                continue
            if any(mm['owner'] == g2['name'] and mm['mode'] == 'GUIDED' for mm in self.our_msls):
                continue
            if best is None or d2 < norm(e.pos - best['pos']):
                best = g2
        return best

    def _vthreat_for(self, f):
        if not self.P['use_vthreat'] or getattr(self, 'vthreat', None) is None:
            return None
        worst = None
        for v in self.vthreat.info.get(f['name'], []):
            if v['hit'] or v['closing'] <= 0:
                continue
            tgo = v['dist'] / max(1.0, v['closing'])
            if (v['dist'] < self.P['vt_react_dist'] or tgo < self.P['vt_react_tgo']) and v['speed'] > f['V'] + self.P['vt_speed_margin']:
                if worst is None or v['dist'] < worst['dist']:
                    worst = v
        return worst

    def sk_vdrag(self, f, v):
        rel = self.T(np.asarray(v['rel'], dtype=np.float64))   # from us to the virtual missile
        return az_of(-rel), self.P['drag_alt'], 'f', 1.0

    def deploy(self, action):
        # A program error loses the match, so any exception falls back to a safe manoeuvre.
        try:
            if not getattr(self, '_world_ok', True):
                raise RuntimeError('world model unavailable: ' + str(getattr(self, '_last_error', '')))
            self._deploy_impl(action)
        except Exception as ex:
            self._last_error = repr(ex)
            self._n_fallback = getattr(self, '_n_fallback', 0) + 1
            if self._n_fallback in (1, 10, 100, 1000):
                import traceback
                print(f"[DBG] FALLBACK#{self._n_fallback} t={getattr(self, 'now', -1)} {traceback.format_exc()[-600:]}")
            self._fallback()

    def _fallback(self):
        lc = self.getLocalCRS()
        for p in self.parents.values():
            try:
                if not p.isAlive():
                    continue
                m = MotionState(p.observables['motion']).transformTo(lc)
                om = MotionState(p.observables['motion'])
                pos = self.T(m.pos())
                tgt = np.array([0.0, 0.0, 0.0]) if abs(pos[0]) < 100000 else np.array([-60000.0, 0.0, 0.0])
                d = unit(np.array([tgt[0] - pos[0], tgt[1] - pos[1], 0.0]))
                alt = -pos[2]
                d[2] = -0.1 if alt < 6000 else (0.1 if alt > 14000 else 0.0)
                self.commands[p.getFullName()] = {
                    'motion': {'dstDir': om.dirAtoP(self.L(unit(d)), m.pos(), lc), 'dstThrottle': 1.0},
                    'weapon': {'launch': False, 'target': Track3D().to_json()},
                }
            except Exception:
                pass

    def _deploy_impl(self, action):
        P = self.P
        self.ext_action = action
        if P.get('debug_raise_after', -1) >= 0 and self.tick > P['debug_raise_after']:
            raise RuntimeError('injected test error')
        if getattr(self, 'vthreat', None) is not None and self.vthreat.vs is not None:
            crs_tracks = []
            for e in self.fighters_e(2.0):
                if e.track is not None:
                    crs_tracks.append((e.uid, e.track))
                elif e.q >= 1:
                    # IRST-only fused track with usable range estimate -> synthesise a Track3D for the virtual shooter
                    crs_tracks.append((e.uid, Track3D(self.parents[self.ports0].getUUIDForTrack(), self.getLocalCRS(),
                                                      self.manager.getTime(), self.L(e.pos), self.L(e.vel))))
            try:
                self.vthreat.step(self, crs_tracks)
            except Exception:
                self.vthreat.vs = None
        for f in self.alive:
            if f['mws']:
                self.last_mws_t[f['name']] = self.now
                self.evade_dir[f['name']] = unit(np.mean([unit(d) for d in f['mws']], axis=0))
        if self.is_decision_tick() or not self.opt:
            act = None
            if action is not None:
                try:
                    a = np.atleast_1d(np.asarray(action)).astype(int)
                    if len(a) >= len(self.F):
                        act = a
                except Exception:
                    act = None
            if act is not None:
                opts, tg = self.apply_external(act[:len(self.F)])
                if len(act) >= 2 * len(self.F):
                    for i, f in enumerate(self.F):
                        self.fire_mode[f['name']] = int(np.clip(act[len(self.F) + i], 0, 2))
            else:
                opts, tg = self.teacher if (self.teacher is not None and self.P['rl_mode']) else self.rule_commander()
                for f in self.F:
                    self.fire_mode[f['name']] = 1
            self.opt.update(opts); self.opt_tgt.update(tg)
        for f in self.alive:
            nm = f['name']
            opt = self.opt.get(nm, OPT_ATTACK)
            beh, pay = self.reflexes(f, opt, self.opt_tgt.get(nm))
            if beh == B_EVADE:
                az, alt, mode, thr = self.sk_evade(f)
            elif beh == B_SUPPORT:
                az, alt, mode, thr = self.sk_support(f, pay)
            elif beh == B_VDRAG:
                az, alt, mode, thr = self.sk_vdrag(f, pay)
            elif beh in (B_TDRAG, OPT_DRAG):
                az, alt, mode, thr = self.sk_drag(f, pay)
            elif beh == B_NOAMMO:
                az, alt, mode, thr = self.sk_noammo(f)
            elif beh == OPT_ENGAGE:
                az, alt, mode, thr = self.sk_engage(f, pay)
            elif beh == OPT_DEFEND:
                az, alt, mode, thr = self.sk_defend(f, pay)
            elif beh == OPT_FLANK_L:
                az, alt, mode, thr = self.sk_flank(f, -1)
            elif beh == OPT_FLANK_R:
                az, alt, mode, thr = self.sk_flank(f, +1)
            else:
                az, alt, mode, thr = self.sk_attack(f)
            az = self._boundary(f, az)
            shot = self.fire_control(f, beh)
            launch, target = False, Track3D()
            if shot is not None:
                e, kind, x, p = shot
                launch, target = True, e.track
                self.pending[nm] = (self.now, e.uid)
                x = dict(x); x['pk_pred'] = p; x['t'] = self.now; x['beh'] = beh
                self.pre_shots[nm] = x
                if P['debug']:
                    print(f"[DBG] t={self.now:.0f} {nm[-3:]} SHOOT {kind} rn={x['rn']:.2f} d={x['d']/1000:.1f}km pk={p:.2f} asp={math.degrees(x['aspect']):.0f}")
            self._command(f, az, alt, mode, thr, launch, target)
            self.beh[nm] = beh
        tf = os.environ.get('TRACE_FILE')
        if tf and self.tick % 2 == 0:
            try:
                import json as _j
                rec = {'t': self.now, 'team': self.own,
                       'F': [[f['name'][-2:], B_NAMES[self.beh.get(f['name'], 0)], round(f['pos'][0] / 1000, 1), round(f['pos'][1] / 1000, 1),
                              round(f['alt'] / 1000, 1), round(f['V']), f['rem'], round(math.degrees(f['az']))] for f in self.alive],
                       'E': [[e.uid[:4], e.src, round(e.pos[0] / 1000, 1), round(e.pos[1] / 1000, 1), round(-e.pos[2] / 1000, 1)]
                             for e in self.fighters_e(10.0)],
                       'M': [[m['owner'][-2:], m['uid'][:4], m['mode'], round(norm(m['tpos'] - m['pos']) / 1000, 1)] for m in self.our_msls],
                       'IL': dict(self.illum)}
                with open(tf, 'a') as fo:
                    fo.write(_j.dumps(rec) + '\n')
            except Exception:
                pass
        if P['debug'] and self.tick % 20 == 1:
            s = ' '.join(f"{f['name'][-2:]}:{B_NAMES[self.beh.get(f['name'],0)]}({f['pos'][0]/1000:.0f},{f['pos'][1]/1000:.0f},{f['alt']/1000:.1f},{f['V']:.0f},m{f['rem']})" for f in self.alive)
            print(f"[DBG] t={self.now:.0f} {s} EE={'none' if self.EE is None else self.EE.src} ee=({self.ee_pos[0]/1000:.0f},{self.ee_pos[1]/1000:.0f}) nE={len(self.fighters_e())} msl={len(self.our_msls)}")

    def _boundary(self, f, az):
        m = self.P['boundary_margin']
        p = f['pos']
        d = dir_from_az(az)
        for ax, L, sgn in ((1, self.dOut - m, 1), (1, self.dOut - m, -1), (0, self.dLine - m, 1), (0, self.dLine - m, -1)):
            over = sgn * p[ax] - (L - 15000.0)
            if over > 0 and sgn * d[ax] > 0:
                w = min(1.0, over / 15000.0)
                d[ax] = d[ax] * (1 - w) - sgn * 0.3 * w
                d = unit(d)
        return az_of(d)

    def _command(self, f, az, alt, mode, thr, launch, target):
        m = f['motion']
        dloc = self.L(dir_from_az(az))
        keeper = self.altKfast if mode == 'f' else self.altK
        dloc = np.array(keeper(m, dloc, alt))
        n = sqrt(dloc[0] ** 2 + dloc[1] ** 2)
        pitch = atan2(-dloc[2], n)
        maxp = radians(float(np.clip((f['V'] - self.P['climb_v0']) / self.P['climb_dv'] * self.P['climb_pmax'], 0.0, self.P['climb_pmax'])))
        if f['alt'] < self.P['alt_floor']:
            pitch = max(pitch, radians(5.0)); maxp = max(maxp, radians(5.0))
            dloc = np.array([dloc[0] / n * cos(pitch), dloc[1] / n * cos(pitch), -sin(pitch)])
        if pitch > maxp:
            pitch = maxp
            dloc = np.array([dloc[0] / n * cos(pitch), dloc[1] / n * cos(pitch), -sin(pitch)])
        self.commands[f['name']] = {
            'motion': {'dstDir': f['omotion'].dirAtoP(dloc, m.pos(), self.getLocalCRS()), 'dstThrottle': float(thr)},
            'weapon': {'launch': bool(launch), 'target': target.to_json()},
        }

    def control(self):
        pass
