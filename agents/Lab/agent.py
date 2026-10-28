"""Missile-engagement lab agent (training/analysis only; not part of the submission).

Shooter team: fighter i flies at its paired target, fires one missile as soon as a radar track
exists (and the scenario's minimum delay has passed), then executes post-launch behaviour S_i:
  'support' crank at FOR edge, 'pursue' keep pointing at target, 'drag' turn away immediately.
Target team: fighter i flies its scenario heading until its MWS fires, then executes evasion E_i:
  'none', 'drag_dive', 'drag_level', 'beam_dive', 'beam_level', 'climb_drag', 'turn_in'
"""
import math
import numpy as np
from math import radians
from agents.TacB.agent import TacAgent, norm, unit, az_of, wrap
from ASRCAISim1.core import Track3D  # type: ignore


class LabAgent(TacAgent):
    def initialize(self):
        super().initialize()
        mc = self.modelConfig()
        self.role = mc.get('lab_role', 'shooter')
        self.lab = mc.get('lab_params', [{} for _ in range(4)])
        self.assigned = {}
        self.fired = {}
        self.mws_t = {}

    def deploy(self, action):
        for i, f in enumerate(self.F):
            if not f['alive']:
                continue
            prm = self.lab[i] if i < len(self.lab) else {}
            nm = f['name']
            alt = prm.get('alt', 10000.0)
            if prm.get('passive'):
                h = radians(prm.get('heading_abs', 90.0))
                self._command(f, az_of(self.T(np.array([math.cos(h), math.sin(h), 0.0]))), alt, 'n', 0.6, False, Track3D())
                continue
            if self.role == 'shooter':
                if nm not in self.assigned:
                    best = None
                    hint = prm.get('tgt_hint')
                    if hint is not None:
                        hp = self.T(np.array(hint, dtype=np.float64))
                        for e in self.enemies.values():
                            if e.track is None:
                                continue
                            if norm(e.pos - hp) < 20000 and (best is None or norm(e.pos - hp) < norm(best.pos - hp)):
                                best = e
                    else:
                        for e in self.enemies.values():
                            if e.track is None or e.is_escort:
                                continue
                            dy = abs(e.pos[1] - f['pos'][1])
                            if dy < 25000 and (best is None or norm(e.pos - f['pos']) < norm(best.pos - f['pos'])):
                                best = e
                    if best is not None:
                        self.assigned[nm] = best.uid
                uid = self.assigned.get(nm)
                e = self.enemies.get(uid) if uid else None
                launch, target = False, Track3D()
                if e is None:
                    az = 0.0 if f['pos'][1] * 0 == 0 else 0.0
                    az = az_of(np.array([1.0, 0.0, 0.0]))
                else:
                    msl = [m for m in self.our_msls if m['owner'] == nm]
                    if nm in self.fired and not msl and self.now - self.fired[nm] > 8 and f['launchable']:
                        del self.fired[nm]          # launch request was not approved (track lost) -> retry
                    if nm not in self.fired:
                        az = az_of(e.pos - f['pos'])
                        if e.track is not None and f['launchable'] and self.now >= prm.get('t_fire', 0.0) and \
                                self.off_bore(f, e.pos) < f['radarFOR'] - 0.1:
                            x = self.shot_features(f, e, 'escort' if prm.get('tgt_hint') is not None else 'fighter')
                            if x is not None:
                                launch, target = True, e.track
                                x = dict(x); x['t'] = self.now; x['S'] = prm.get('S', 'support'); x['E'] = prm.get('E_tgt', '')
                                x['pk_pred'] = 0.0; x['beh'] = 0
                                self.pre_shots[nm] = x
                                self.fired[nm] = self.now
                    else:
                        S = prm.get('S', 'support')
                        if S == 'crank':
                            # keep the target near the radar gimbal limit (offset 50 deg), same side as now
                            los = az_of(e.pos - f['pos'])
                            side = 1.0 if wrap(az_of(f['vel']) - los) >= 0 else -1.0
                            az = wrap(los + side * radians(prm.get('crank_deg', 50.0)))
                        elif S == 'drag' and self.now - self.fired[nm] > 4:
                            az = az_of(f['pos'] - e.pos)
                        elif S == 'pursue':
                            az = az_of(e.pos - f['pos'])
                        else:
                            if msl:
                                az, _, _, _ = self.sk_support(f, msl[0])
                            else:
                                az = az_of(e.pos - f['pos'])
                self._command(f, az, alt, 'n', 1.0, launch, target)
            else:
                h = radians(prm.get('heading_abs', 90.0))
                hdg = az_of(self.T(np.array([math.cos(h), math.sin(h), 0.0])))
                E = prm.get('E', 'none')
                if f['mws']:
                    self.mws_t[nm] = (self.now, unit(np.mean([unit(d) for d in f['mws']], axis=0)))
                mode, az = 'n', hdg
                if E == 'cold_at':
                    # turn cold (away from the shooter's side) at an absolute time, independent of MWS
                    if self.now >= prm.get('t_cold', 1e9):
                        self._command(f, wrap(hdg + math.pi), prm.get('alt', 10000.0), 'n', 1.0, False, Track3D())
                    else:
                        self._command(f, hdg, prm.get('alt', 10000.0), 'n', prm.get('thr', 1.0), False, Track3D())
                    continue
                if E.startswith('weave0'):
                    # pre-emptive weave from the start (missile unseen): vertical oscillation at constant heading
                    per, amp = (6.0, 0.45) if E == 'weave0' else ((3.0, 0.8) if E == 'weave0f' else (10.0, 0.35))
                    ph = 1.0 if int(self.now / per) % 2 == 0 else -1.0
                    if f['alt'] > 15500: ph = -1.0
                    if f['alt'] < 5000: ph = 1.0
                    self._command(f, hdg, ph * amp, 'x', prm.get('thr', 1.0), False, Track3D())
                    continue
                if nm in self.mws_t and E != 'none':
                    t0, d = self.mws_t[nm]
                    if self.now - t0 < prm.get('hold', 30.0):
                        maz = az_of(d)            # direction to missile
                        if E == 'drag_dive':
                            az, alt, mode = wrap(maz + math.pi), 2000.0, 'f'
                        elif E == 'drag_level':
                            az = wrap(maz + math.pi)
                        elif E == 'beam_dive':
                            az, alt, mode = wrap(maz + math.pi / 2), 2000.0, 'f'
                        elif E == 'beam_level':
                            az = wrap(maz + math.pi / 2)
                        elif E == 'climb_drag':
                            az, alt = wrap(maz + math.pi), 16000.0
                        elif E == 'turn_in':
                            az, alt, mode = maz, 2000.0, 'f'
                        else:
                            per = prm.get('osc_period', 4.0)
                            ph = 1.0 if int((self.now - t0) / per) % 2 == 0 else -1.0
                            amp = prm.get('osc_pitch', 0.6)
                            if E == 'spiral':
                                az, alt, mode = wrap(f['az'] + 1.2), ph * amp, 'x'
                            elif E == 'beam_osc':
                                side = 1.0 if wrap(f['az'] - maz) >= 0 else -1.0
                                az, alt, mode = wrap(maz + side * math.pi / 2), ph * amp, 'x'
                            elif E == 'drag_osc':
                                az, alt, mode = wrap(maz + math.pi), ph * amp, 'x'
                            elif E == 'dive_x':
                                az, alt, mode = wrap(maz + math.pi / 2), -0.8, 'x'
                self._command(f, az, alt, mode, prm.get('thr', 1.0), False, Track3D())
