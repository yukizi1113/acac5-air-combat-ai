"""Field-log Critic: win probability from the logged situation (own sensors only) at time t.
Trained on real leaderboard matches with match-grouped cross-validation."""
import pickle, math, numpy as np, pandas as pd, json, sys
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
M = pickle.load(open('/home/claude/acac/lb/matches.pkl', 'rb'))
def feats(X, t):
    ex, ey = X[t, 42], X[t, 43]; eex, eey = X[t, 157], X[t, 158]
    al = [i for i in range(4) if X[t, 1 + 10 * i] == 1]
    rem = sum(X[t, 7 + 10 * i] for i in al)
    en = [(X[t, 44 + 8 * k + 1], X[t, 44 + 8 * k + 2], X[t, 44 + 8 * k + 3], X[t, 44 + 8 * k + 4], X[t, 44 + 8 * k + 5], X[t, 44 + 8 * k + 6])
          for k in range(8) if X[t, 44 + 8 * k] == 1 and X[t, 44 + 8 * k + 7] == 0]
    dead = sum(1 for k in range(8) if X[t, 44 + 8 * k] == -1 and X[t, 44 + 8 * k + 7] == 0)
    d_en_esc = [math.hypot(x - ex, y - ey) for x, y, *_ in en]
    n_in = 0
    for x, y, a, V, az, src in en:
        if V > 100:
            vx, vy = math.cos(math.radians(az)), math.sin(math.radians(az)); dx, dy = ex - x, ey - y; d = math.hypot(dx, dy)
            if d > 1 and (vx * dx + vy * dy) / d > math.cos(math.radians(35)): n_in += 1
    d_my_ee = [math.hypot(X[t, 2 + 10 * i] - eex, X[t, 3 + 10 * i] - eey) for i in al]
    msl = sum(1 for k in range(8) if X[t, 108 + 6 * k] == 1)
    msl_ee = sum(1 for k in range(8) if X[t, 108 + 6 * k] == 1 and math.hypot(X[t, 109 + 6 * k] - eex, X[t, 110 + 6 * k] - eey) < 40)
    return dict(t=t, n_my=len(al), rem=rem, n_en=len(en), en_dead=dead, n_in=n_in,
                d_en_esc_min=min(d_en_esc) if d_en_esc else 300, d_my_ee_min=min(d_my_ee) if d_my_ee else 300,
                msl=msl, msl_ee=msl_ee, alt_my=np.mean([X[t, 4 + 10 * i] for i in al]) if al else 0,
                alt_en=np.median([e[2] for e in en]) if en else 10, x_en=np.mean([e[0] for e in en]) if en else 100,
                x_my=max([X[t, 2 + 10 * i] for i in al]) if al else -150)
rows = []
for mi, m in enumerate(M):
    if m['result'] == 'Draw': continue
    X = m['X']
    for t in range(0, len(X), 10):
        r = feats(X, t); r['mi'] = mi; r['y'] = int(m['result'] == 'Win'); rows.append(r)
D = pd.DataFrame(rows); F = [c for c in D.columns if c not in ('mi', 'y')]
oof = np.zeros(len(D))
for tr, te in GroupKFold(5).split(D, D.y, D.mi):
    mdl = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
    mdl.fit(D.iloc[tr][F], D.iloc[tr].y); oof[te] = mdl.predict_proba(D.iloc[te][F])[:, 1]
D['p'] = oof
print('rows', len(D), 'matches', D.mi.nunique(), 'overall AUC', round(roc_auc_score(D.y, D.p), 3))
for lo, hi in [(0, 100), (100, 200), (200, 300), (300, 400), (400, 600), (600, 2000)]:
    s = D[(D.t >= lo) & (D.t < hi)]
    if s.y.nunique() > 1: print(f't {lo}-{hi}: n={len(s)} AUC={roc_auc_score(s.y, s.p):.3f}')
mdl = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1).fit(D[F], D.y)
imp = sorted(zip(mdl.booster_.feature_importance('gain'), F), reverse=True)
print('gain importance', [(f, round(g)) for g, f in imp])
D.to_pickle('/home/claude/acac/lb/critic_oof.pkl'); mdl.booster_.save_model('/home/claude/acac/lb/critic_lgb.txt')
