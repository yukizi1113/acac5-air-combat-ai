"""Opponent-model features at t=200 s computed from field logs (same definitions as agent.opp_features)."""
import pickle, math, numpy as np, pandas as pd
M = pickle.load(open('/home/claude/acac/lb/matches.pkl', 'rb'))
rows = []
for mi, m in enumerate(M):
    X = m['X']; t = 200
    if len(X) <= t: continue
    ex, ey = X[t, 42], X[t, 43]
    en = [(X[t, 44+8*k+1], X[t, 44+8*k+2], X[t, 44+8*k+3], X[t, 44+8*k+4], X[t, 44+8*k+5]) for k in range(8) if X[t, 44+8*k] == 1 and X[t, 44+8*k+7] == 0]
    n_in = 0
    for x, y, a, V, az in en:
        if V > 100:
            vx, vy = math.cos(math.radians(az)), math.sin(math.radians(az)); dx, dy = ex-x, ey-y; d = math.hypot(dx, dy)
            if d > 1 and (vx*dx+vy*dy)/d > math.cos(math.radians(35)): n_in += 1
    lost = sum(1 for i in range(4) if X[t, 1+10*i] == 0)
    kills = sum(1 for k in range(8) if X[t, 44+8*k] == -1 and X[t, 44+8*k+7] == 0)
    rows.append(dict(mi=mi, model=m['model'], res=m['result'], n_in=n_in, n_trk=len(en), x_mean=np.mean([e[0] for e in en]) if en else 100,
                     x_min=min([e[0] for e in en]) if en else 100, alt_med=np.median([e[2] for e in en]) if en else 10,
                     V_med=np.median([e[3] for e in en if e[3] > 50]) if any(e[3] > 50 for e in en) else 500,
                     d_esc_min=min([math.hypot(e[0]-ex, e[1]-ey) for e in en]) if en else 300, kills=kills, lost=lost))
F = pd.DataFrame(rows); F.to_pickle('/home/claude/acac/lb/fieldfeat200.pkl')
print(F.describe().round(2).T)
