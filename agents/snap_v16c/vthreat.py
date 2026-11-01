"""Virtual-missile threat estimation (inference of unseen enemy launches).

Enemy missiles are invisible until their seeker locks on (MWS, ~10 km), which is too late to defeat a
head-on shot.  Data from our matches show that what decides survival is the target aspect during the
missile's terminal phase (cold >135 deg: 0 % hits, hot: ~35 %).  We therefore assume that a hot enemy
fighter inside its own launch envelope against us (normalised range <= kShoot) may have fired, and fly a
*virtual* missile from its track against our real trajectory inside the simulator's virtual simulation
manager (same missile model, proportional navigation, our actual manoeuvres as target motion).
A fighter only needs to turn cold when a virtual missile is close and still energetic enough to catch it;
when the virtual missile runs out of energy the fighter re-commits.
"""
import numpy as np
from ASRCAISim1.core import (  # type: ignore
    nljson, MotionState, Track3D, Quaternion, CommunicationBuffer, SimPhase,
)
from BasicAgentUtility.util import calcRNorm  # type: ignore


class VirtualThreat:
    def __init__(self, k_shoot=0.4, interval=30.0, max_per_parent=3, enabled=True):
        self.k_shoot = k_shoot
        self.interval = interval
        self.max_per_parent = max_per_parent
        self.enabled = enabled
        self.vs = None
        self.vm = {}          # parent name -> list of (missile, src uid, t_launch)
        self.last_spawn = {}  # (parent, uid) -> time
        self.info = {}        # parent name -> list of dicts(dist, closing, speed, dir_from_target, age)

    def create(self, agent):
        if not self.enabled:
            return
        try:
            option = nljson({"patch": {"Manager": {"TimeStep": {"defaultAgentStepInterval": agent.interval[SimPhase.AGENT_STEP]},
                                                   "skipNoAgentStep": False}}})
            self.vs = agent.manager.createVirtualSimulationManager(option)
            self.vs.reset()
        except Exception:
            self.vs = None

    def step(self, agent, enemy_tracks):
        """enemy_tracks: list of (uid, Track3D in root CRS-compatible form)."""
        if self.vs is None:
            return
        now_tick = agent.manager.getTickCount()
        t = agent.manager.getElapsedTime()
        crs = agent.manager.getRootCRS()
        time = agent.manager.getTime()
        dt = agent.interval[SimPhase.AGENT_STEP] * agent.manager.getBaseTimeStep()
        self.info = {}
        for port, parent in agent.parents.items():
            nm = parent.getFullName()
            lst = self.vm.setdefault(nm, [])
            if not parent.isAlive():
                for m, _, _ in lst:
                    m.manager.requestToKillAsset(m)
                self.vm[nm] = []
                continue
            motion = MotionState(parent.observables.at("motion"))
            selfTrack = Track3D(parent.getUUIDForTrack(), crs, time, motion.pos(crs), motion.vel(crs))
            keep, inf = [], []
            for (m, uid, t0) in lst:
                if not m.isAlive():
                    continue
                m.communicationBuffers["MissileComm:" + m.getFullName()]().send({"target": selfTrack}, CommunicationBuffer.UpdateType.MERGE)
                mp = np.array(m.pos()); mv = np.array(m.vel())
                tp = np.array(motion.pos(m.getParentCRS())); tv = np.array(motion.vel(m.getParentCRS()))
                if m.hitCheck(tp, tp - tv * dt):
                    m.manager.requestToKillAsset(m)
                    inf.append({'dist': 0.0, 'closing': 1000.0, 'speed': float(np.linalg.norm(mv)), 'rel': mp - tp,
                                'age': t - t0, 'uid': uid, 'hit': True})
                    continue
                rel = mp - tp
                d = float(np.linalg.norm(rel))
                closing = float(-(mv - tv) @ (rel / max(1.0, d)) * -1.0) if d > 1 else 0.0
                inf.append({'dist': d, 'closing': -closing, 'speed': float(np.linalg.norm(mv)), 'rel': rel,
                            'age': t - t0, 'uid': uid, 'hit': False})
                keep.append((m, uid, t0))
            self.vm[nm] = keep
            self.info[nm] = inf
        while self.vs.getTickCount() < now_tick:
            self.vs.step({})
        # spawn new virtual missiles
        for port, parent in agent.parents.items():
            nm = parent.getFullName()
            if not parent.isAlive():
                continue
            motion = MotionState(parent.observables.at("motion"))
            selfTrack = Track3D(parent.getUUIDForTrack(), crs, time, motion.pos(crs), motion.vel(crs))
            for uid, tr in enemy_tracks:
                if len(self.vm[nm]) >= self.max_per_parent:
                    break
                if t - self.last_spawn.get((nm, uid), -1e9) < self.interval:
                    continue
                sm = MotionState(crs, time, tr.pos(crs), tr.vel(crs), np.array([0.0, 0.0, 0.0]), Quaternion(1, 0, 0, 0), "FSD")
                try:
                    if calcRNorm(parent, sm, selfTrack, False) <= self.k_shoot:
                        cmd = nljson({"target": selfTrack, "motion": sm})
                        self.vm[nm].append((parent.launchInVirtual(self.vs, cmd), uid, t))
                        self.last_spawn[(nm, uid)] = t
                except Exception:
                    pass
