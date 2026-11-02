import pickle, math, numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
M = pickle.load(open('/home/claude/acac/lb/matches.pkl', 'rb'))
rows = []
for mi, m in enumerate(M):
    X = m['X']; f = {}
    ok = True
    for t in [170, 210]:
        if len(X) <= t: ok = False; break
        en = [(X[t,44+8*k+1], X[t,44+8*k+2], X[t,44+8*k+3], X[t,44+8*k+4], X[t,44+8*k+5]) for k in range(8)
              if X[t,44+8*k] == 1 and X[t,44+8*k+7] == 0 and X[t,44+8*k+6] == 1]
        if len(en) < 2: ok = False; break
        a = np.array(en)
        f[f'alt{t}'] = np.median(a[:, 2]); f[f'altsd{t}'] = a[:, 2].std(); f[f'V{t}'] = np.median(a[:, 3])
        f[f'ysp{t}'] = a[:, 1].max() - a[:, 1].min(); f[f'x{t}'] = a[:, 0].mean(); f[f'n{t}'] = len(en)
        f[f'hdg{t}'] = np.median(np.abs(((a[:, 4] - 180) + 180) % 360 - 180))   # deviation from heading west (toward us)
    if not ok: continue
    f.update(mi=mi, model=m['model'], res=m['result'], stamp=m['stamp'])
    rows.append(f)
F = pd.DataFrame(rows)
cols = [c for c in F.columns if c not in ('mi', 'model', 'res', 'stamp')]
Z = StandardScaler().fit_transform(F[cols])
for k in [6, 8, 10]:
    km = KMeans(k, n_init=20, random_state=0).fit(Z)
    F[f'c{k}'] = km.labels_
g = F.groupby('c8').agg(n=('mi', 'size'), win=('res', lambda s: (s == 'Win').mean()), **{c: (c, 'median') for c in ['alt170', 'V170', 'ysp170', 'x170', 'hdg170', 'altsd170', 'x210']})
print(g.round(2))
F.to_pickle('/home/claude/acac/lb/fingerprint.pkl')
