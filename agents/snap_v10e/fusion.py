"""Multi-sensor multi-target tracking: radar Track3D + anonymous IRST bearings -> fused enemy tracks.

Why: radar tracks carry the true target id and full 3-D state, but radar range/coverage is short.
IRST sees ~1.5x further (and the large escort ~2x further) but gives bearings only, and its track ids
are *anonymous per sensor* (re-randomised after each loss of contact), so bearings from different
fighters, or from IRST vs radar, cannot be matched by id.  We therefore solve data association:

  1. predict every track (EKF, nearly-constant-velocity model)
  2. radar: update the track bound to that true id, else adopt a gated IRST-only track (3-D
     Mahalanobis gate), else create a new track
  3. IRST: a bearing keeps its previous association while the sensor's anonymous id persists;
     otherwise global-nearest-neighbour on the normalised angular innovation
     (sigma^2 = sigma_IRST^2 + (sigma_perp / range)^2), gate = 3 sigma
  4. unassociated bearings from two different fighters are triangulated into a new track;
     a lone bearing seen at t~0 (only the escort's IR signature reaches that far) seeds the escort
     track using the racetrack prior; other lone bearings seed a low-quality tentative track
  5. tracks without updates for 30 s are dropped; at most 5 enemy tracks (4 fighters + escort)
"""
import math
import numpy as np
from .tracker import EKFTrack, init_from_bearing

ESCORT_X = 120000.0
ESCORT_ALT = 10000.0


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


class FTrack:
    _ctr = 0

    def __init__(self, kf, t, radar_uid=None):
        FTrack._ctr += 1
        self.id = radar_uid if radar_uid else f'I{FTrack._ctr}'
        self.kf = kf
        self.radar_uid = radar_uid
        self.t_radar = t if radar_uid else -1e9
        self.t_irst = -1e9
        self.t_created = t
        self.n_irst = 0
        self.n_tri = 0
        self.tentative = radar_uid is None
        self.escort_seed = False
        # attributes used by the agent (EnemyTrack interface)
        self.escort_score = 0.0
        self.is_escort = False
        self.dead = False
        self.track = None
        self.irst = []
        self.src = 'none'
        self.first_t = t

    @property
    def uid(self):
        return self.id

    @property
    def pos(self):
        return self.kf.pos

    @property
    def vel(self):
        return self.kf.vel

    @property
    def t(self):
        return max(self.t_radar, self.t_irst)

    @property
    def q(self):
        if self.radar_uid is not None and self.src == 'radar':
            return 2
        s = self.kf.pos_sigma()
        return 2 if s < 5000 else (1 if s < 20000 else 0)


class Fusion:
    def __init__(self, sigma_irst_deg=0.6):
        self.tracks = {}          # id -> FTrack
        self.assoc = {}           # (fighter name, irst anon uid) -> track id
        self.s_irst = math.radians(sigma_irst_deg)

    # ------------------------------------------------------------------ helpers
    def _bearing_err(self, tr, org, d):
        rel = tr.kf.pos - org
        r = np.linalg.norm(rel)
        if r < 1.0:
            return 1e9, r
        u = rel / r
        ang = math.acos(max(-1.0, min(1.0, float(u @ d))))
        P = tr.kf.P[0:3, 0:3]
        perp_var = max(0.0, float(np.trace(P) - u @ P @ u)) / 2.0
        s2 = self.s_irst ** 2 + perp_var / (r * r)
        return ang / math.sqrt(s2), r

    @staticmethod
    def _triangulate(o1, d1, o2, d2):
        w0 = o1 - o2
        a, b, c = d1 @ d1, d1 @ d2, d2 @ d2
        dd, e = d1 @ w0, d2 @ w0
        den = a * c - b * b
        if den < 1e-6:
            return None
        s = (b * e - c * dd) / den
        t = (a * e - b * dd) / den
        if s < 5000 or t < 5000 or s > 260000 or t > 260000:
            return None
        p1, p2 = o1 + s * d1, o2 + t * d2
        if np.linalg.norm(p1 - p2) > 4000:
            return None
        sin_ang = math.sqrt(max(1e-12, 1.0 - b * b))
        return 0.5 * (p1 + p2), s, t, sin_ang

    # ------------------------------------------------------------------ main step
    def step(self, t, radar, bearings, friends_pos, known_escort_id=None):
        """radar: list of (uid, pos, vel); bearings: list of (fighter, anon_uid, origin, dir).
        Returns set of track ids updated by radar this tick."""
        for tr in self.tracks.values():
            tr.kf.predict(t)
        radar_ids = set()
        # ---- radar
        for uid, p, v in radar:
            rng = min([np.linalg.norm(p - fp) for fp in friends_pos] or [50000.0])
            tr = self.tracks.get(uid)
            if tr is None:
                # adopt an IRST-only track?
                best, bd = None, 11.3
                for c in self.tracks.values():
                    if c.radar_uid is not None:
                        continue
                    S = c.kf.P[0:3, 0:3] + np.eye(3) * max(150.0, rng * 0.005) ** 2
                    y = p - c.kf.pos
                    try:
                        d2 = float(y @ np.linalg.solve(S, y))
                    except np.linalg.LinAlgError:
                        continue
                    if d2 < bd:
                        best, bd = c, d2
                if best is not None:
                    del self.tracks[best.id]
                    for k, v_ in list(self.assoc.items()):
                        if v_ == best.id:
                            self.assoc[k] = uid
                    best.id = uid; best.radar_uid = uid; best.tentative = False
                    tr = self.tracks[uid] = best
                    tr.kf.update_radar(p, v, rng)
                else:
                    P0 = np.diag([300.0 ** 2] * 3 + [30.0 ** 2] * 3)
                    tr = self.tracks[uid] = FTrack(EKFTrack(np.concatenate([p, v]), P0, t, q=15.0), t, uid)
                    tr.tentative = False
            else:
                tr.kf.update_radar(p, v, rng)
            tr.t_radar = t
            radar_ids.add(uid)
        # ---- IRST association
        unassigned = []
        used = {}
        for (fname, anon, org, d) in bearings:
            d = _unit(d)
            key = (fname, anon)
            tid = self.assoc.get(key)
            if tid is not None and tid in self.tracks:
                err, _ = self._bearing_err(self.tracks[tid], org, d)
                if err < 6.0:
                    used.setdefault(tid, []).append((org, d, key))
                    continue
            best, be = None, 3.0
            for c in self.tracks.values():
                err, r = self._bearing_err(c, org, d)
                if err < be:
                    best, be = c, err
            if best is not None:
                self.assoc[key] = best.id
                used.setdefault(best.id, []).append((org, d, key))
            else:
                unassigned.append((fname, anon, org, d))
        for tid, lst in used.items():
            tr = self.tracks[tid]
            tr.t_irst = t
            tr.n_irst += 1
            if tid in radar_ids:
                continue
            for org, d, _ in lst:
                tr.kf.update_bearing(org, d, math.degrees(self.s_irst))
            if len(set(k[0] for _, _, k in lst)) >= 2:
                tr.n_tri += 1
            if tr.escort_seed or tid == known_escort_id:
                tr.kf.update_escort_prior(ESCORT_X, ESCORT_ALT)
            if tr.tentative and (tr.n_tri >= 2 or tr.kf.pos_sigma() < 8000):
                tr.tentative = False
        # ---- new tracks from unassigned bearings
        taken = set()
        for i in range(len(unassigned)):
            if i in taken:
                continue
            fi, ai, oi, di = unassigned[i]
            best = None
            for j in range(i + 1, len(unassigned)):
                if j in taken or unassigned[j][0] == fi:
                    continue
                res = self._triangulate(oi, di, unassigned[j][2], unassigned[j][3])
                if res is not None and (best is None or res[3] > best[1][3]):
                    best = (j, res)
            if best is not None and best[1][3] > math.sin(math.radians(2.0)):
                j, (p, s, tt, sin_ang) = best
                taken.update((i, j))
                sig_r = s * self.s_irst / sin_ang
                kf = init_from_bearing(oi, di, t, s, max(1000.0, sig_r))
                kf.x[0:3] = p
                tr = FTrack(kf, t)
                tr.t_irst = t; tr.n_tri = 1
                self.tracks[tr.id] = tr
                self.assoc[(fi, ai)] = tr.id
                self.assoc[(unassigned[j][0], unassigned[j][1])] = tr.id
        for i, (fi, ai, oi, di) in enumerate(unassigned):
            if i in taken:
                continue
            di = _unit(di)
            if len(self.tracks) >= 7:
                continue
            # only the escort's IR signature is detectable at t~0 (fighters start ~200 km apart)
            escort_like = t < 3.0
            if not escort_like and known_escort_id is None and not any(tr.escort_seed for tr in self.tracks.values()):
                # first detection happens at the target's max IRST range: escort (IR x4) -> 2 x our range.
                # If that hypothesis puts the contact on the escort racetrack, it is the escort.
                L = getattr(self, 'irst_L', {}).get(fi)
                if L:
                    pe = oi + di * (2.0 * L)
                    if abs(pe[0] - ESCORT_X) < 15000 and abs(pe[1]) < 112000 and -pe[2] > 6000:
                        escort_like = True
            if escort_like and di[0] > 0.05 and not any(tr.escort_seed for tr in self.tracks.values()):
                rg = (ESCORT_X - oi[0]) / di[0]
                kf = init_from_bearing(oi, di, t, rg, 30000.0, v_sigma=250.0, q=3.0)
                tr = FTrack(kf, t); tr.escort_seed = True; tr.tentative = False
            else:
                kf = init_from_bearing(oi, di, t, 80000.0, 40000.0, v_sigma=350.0)
                tr = FTrack(kf, t)
            tr.t_irst = t
            self.tracks[tr.id] = tr
            self.assoc[(fi, ai)] = tr.id
        # ---- pruning
        for tid in [k for k, tr in self.tracks.items() if t - max(tr.t_radar, tr.t_irst) > 30.0]:
            del self.tracks[tid]
        for k in [k for k, v in self.assoc.items() if v not in self.tracks]:
            del self.assoc[k]
        # merge near-duplicate non-radar tracks
        ids = list(self.tracks.keys())
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                ta, tb = self.tracks.get(ids[a]), self.tracks.get(ids[b])
                if ta is None or tb is None:
                    continue
                if ta.radar_uid and tb.radar_uid:
                    continue
                if np.linalg.norm(ta.kf.pos - tb.kf.pos) < 3000 and np.linalg.norm(ta.kf.vel - tb.kf.vel) < 150:
                    keep, drop = (ta, tb) if (ta.radar_uid or ta.kf.pos_sigma() < tb.kf.pos_sigma()) else (tb, ta)
                    keep.escort_seed = keep.escort_seed or drop.escort_seed
                    del self.tracks[drop.id]
                    for k, v in list(self.assoc.items()):
                        if v == drop.id:
                            self.assoc[k] = keep.id
        return radar_ids

