"""Extended Kalman filter tracker fusing radar (3-D position/velocity) and IRST (bearing-only) data.

State  x = [px, py, pz, vx, vy, vz]  (team frame, metres, m/s)
Motion model: nearly-constant velocity, white acceleration noise with std `q` (m/s^2)
Measurements
  radar : z = [p, v]                     H linear, R = diag(sig_p^2 (range dependent), sig_v^2)
  IRST  : z = [azimuth, elevation] of (p - observer)   (EKF linearisation, sigma ~0.5 deg)
  prior : escort racetrack pseudo-measurements  px = 120 km (sigma 6 km), alt = 10 km (sigma 0.5 km)
Bearing-only observability comes from several fighters observing simultaneously (triangulation)
and from own-ship motion over time; the escort prior makes a single bearing sufficient.
"""
import math
import numpy as np


class EKFTrack:
    def __init__(self, x0, P0, t, q=15.0):
        self.x = np.asarray(x0, dtype=np.float64).copy()
        self.P = np.asarray(P0, dtype=np.float64).copy()
        self.t = t
        self.q = q

    def predict(self, t):
        dt = t - self.t
        if dt <= 0:
            return
        F = np.eye(6)
        F[0:3, 3:6] = np.eye(3) * dt
        q = self.q ** 2
        G = np.zeros((6, 3))
        G[0:3] = np.eye(3) * 0.5 * dt * dt
        G[3:6] = np.eye(3) * dt
        Q = G @ G.T * q
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.t = t

    def _update(self, z, h, H, R, angle_idx=()):
        y = z - h
        for i in angle_idx:
            y[i] = (y[i] + math.pi) % (2 * math.pi) - math.pi
        S = H @ self.P @ H.T + R
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return
        # Mahalanobis gating (reject outliers)
        d2 = float(y @ np.linalg.solve(S, y))
        if d2 > 60.0:
            return
        self.x = self.x + K @ y
        I_KH = np.eye(6) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T   # Joseph form

    def update_radar(self, p, v, rng):
        sp = max(150.0, rng * math.radians(0.3))
        R = np.diag([sp ** 2] * 3 + [20.0 ** 2] * 3)
        self._update(np.concatenate([p, v]), self.x.copy(), np.eye(6), R)

    def update_bearing(self, org, d, sigma_deg=0.6):
        rel = self.x[0:3] - org
        rh2 = rel[0] ** 2 + rel[1] ** 2
        rh = math.sqrt(max(rh2, 1.0))
        r2 = rh2 + rel[2] ** 2
        az = math.atan2(rel[1], rel[0])
        el = math.atan2(-rel[2], rh)
        zaz = math.atan2(d[1], d[0])
        zel = math.atan2(-d[2], math.sqrt(d[0] ** 2 + d[1] ** 2))
        H = np.zeros((2, 6))
        H[0, 0] = -rel[1] / rh2; H[0, 1] = rel[0] / rh2
        H[1, 0] = rel[0] * rel[2] / (r2 * rh); H[1, 1] = rel[1] * rel[2] / (r2 * rh); H[1, 2] = -rh / r2
        s = math.radians(sigma_deg)
        self._update(np.array([zaz, zel]), np.array([az, el]), H, np.diag([s * s, s * s]), angle_idx=(0, 1))

    def update_escort_prior(self, x_line, alt, sx=6000.0, salt=500.0):
        H = np.zeros((2, 6)); H[0, 0] = 1.0; H[1, 2] = 1.0
        self._update(np.array([x_line, -alt]), np.array([self.x[0], self.x[2]]), H, np.diag([sx * sx, salt * salt]))

    @property
    def pos(self):
        return self.x[0:3]

    @property
    def vel(self):
        return self.x[3:6]

    def pos_sigma(self):
        return math.sqrt(max(0.0, float(np.trace(self.P[0:3, 0:3]))))


def init_from_bearing(org, d, t, rng_guess, rng_sigma, v_sigma=300.0, q=15.0):
    d = np.asarray(d) / max(1e-9, np.linalg.norm(d))
    p = np.asarray(org) + d * rng_guess
    # covariance: large along line of sight, small across
    P = np.zeros((6, 6))
    across = (rng_guess * math.radians(0.6)) ** 2
    Pp = np.eye(3) * across + np.outer(d, d) * (rng_sigma ** 2 - across)
    P[0:3, 0:3] = Pp
    P[3:6, 3:6] = np.eye(3) * v_sigma ** 2
    return EKFTrack(np.concatenate([p, np.zeros(3)]), P, t, q)
