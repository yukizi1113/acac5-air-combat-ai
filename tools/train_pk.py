"""Train the hit-probability (Pk) model from logged shots.

Inputs : agent-side shot features (data/shots_*.jsonl, keyed by match id + missile name)
         harness-side outcomes (results/*.jsonl with godlog: missile name -> hit)
Models : (a) logistic regression on engineered features, (b) LightGBM (monotone constraints),
         (c) small MLP. Compared by grouped (per-match) cross-validation: log-loss, AUC, Brier,
         and calibration. Best model is exported to agents/<pkg>/pk_model.json for numpy-only inference
         (logistic / MLP) or pk_model.txt (LightGBM).
"""
import json, glob, argparse, os, math
import numpy as np

FEATS = ['rn', 'd_km', 'rh_km', 'rt_km', 'aspect', 'ata', 'alt_s_km', 'alt_t_km', 'V_s', 'V_t',
         'closing', 'thrust_r', 'kind', 'dz_km', 'rhe_km', 'is_lab']


def load(shot_files, result_files):
    outcome = {}
    meta = {}
    for rf in result_files:
        for line in open(rf):
            r = json.loads(line)
            mid = r.get('match_id')
            if not mid or not r.get('godlog'):
                continue
            for m in r['godlog']:
                outcome[(mid, m['name'])] = 1 if m['hit'] else 0
                meta[(mid, m['name'])] = m
    X, y, groups, raw = [], [], [], []
    for sf in shot_files:
        for line in open(sf):
            s = json.loads(line)
            key = (s.get('match'), s['msl'])
            if key not in outcome:
                continue
            X.append(featurize(s)); y.append(outcome[key]); groups.append(s.get('match')); raw.append(s)
    return np.array(X, dtype=np.float64), np.array(y), np.array(groups), raw


def featurize(s):
    return [s['rn'], s['d'] / 1000, s['rh'] / 1000, s['rt'] / 1000, s['aspect'], s['ata'],
            s['alt_s'] / 1000, s['alt_t'] / 1000, s['V_s'], s['V_t'], s['closing'],
            s['thrust'] / 30017.9989, s['kind'], s['dz'] / 1000, s.get('rhe', 0.0) / 1000,
            1.0 if str(s.get('match', '')).startswith('col_lab') else 0.0]


def metrics(y, p):
    from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
    p = np.clip(p, 1e-4, 1 - 1e-4)
    out = {'logloss': log_loss(y, p), 'brier': brier_score_loss(y, p)}
    try:
        out['auc'] = roc_auc_score(y, p)
    except ValueError:
        out['auc'] = float('nan')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shots', default='data/shots_*.jsonl')
    ap.add_argument('--results', default='results/col_*.jsonl')
    ap.add_argument('--out', default='agents/Tac/pk_model.json')
    ap.add_argument('--model', default='auto')
    ap.add_argument('--search', action='store_true')
    a = ap.parse_args()
    X, y, g, raw = load(sorted(glob.glob(a.shots)), sorted(glob.glob(a.results)))
    print(f'samples={len(y)} hits={y.sum()} rate={y.mean():.3f}')
    for k in (0, 1):
        m = X[:, 12] == k
        if m.any():
            print(f'  kind={"escort" if k else "fighter"} n={m.sum()} hit={y[m].mean():.3f}')
    from sklearn.model_selection import GroupKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.neural_network import MLPClassifier
    import lightgbm as lgb
    models = {
        'logreg': lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)),
        'mlp': lambda: make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(32, 16), alpha=1e-2, max_iter=3000, random_state=0)),
        'lgbm': lambda: lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=8, min_child_samples=10,
                                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1,
                                           monotone_constraints=[-1, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]),
    }
    if a.search:
        grid = {}
        for C in (0.03, 0.1, 0.3, 1.0, 3.0):
            grid[f'logreg_C{C}'] = (lambda C=C: make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000)))
        for nl in (4, 8, 16):
            for ne, lr in ((150, 0.05), (400, 0.02)):
                for mcs in (10, 30):
                    grid[f'lgbm_l{nl}_n{ne}_lr{lr}_m{mcs}'] = (lambda nl=nl, ne=ne, lr=lr, mcs=mcs: lgb.LGBMClassifier(
                        n_estimators=ne, learning_rate=lr, num_leaves=nl, min_child_samples=mcs, subsample=0.8, subsample_freq=1,
                        colsample_bytree=0.8, reg_lambda=1.0, verbose=-1))
        for hs in ((16,), (32, 16), (64, 32)):
            for al in (1e-3, 1e-2, 1e-1):
                grid[f'mlp_{"x".join(map(str, hs))}_a{al}'] = (lambda hs=hs, al=al: make_pipeline(StandardScaler(), MLPClassifier(
                    hidden_layer_sizes=hs, alpha=al, max_iter=3000, early_stopping=True, random_state=0)))
        gkf = GroupKFold(n_splits=min(5, len(set(g))))
        rows = []
        for name, mk in grid.items():
            p = np.zeros(len(y))
            for tr, te in gkf.split(X, y, g):
                m = mk(); m.fit(X[tr], y[tr]); p[te] = m.predict_proba(X[te])[:, 1]
            r = metrics(y, p); rows.append((name, r))
            print(f"  {name:32s} logloss={r['logloss']:.4f} brier={r['brier']:.4f} auc={r['auc']:.4f}", flush=True)
        json.dump(rows, open(a.out.replace('.json', '_search.json'), 'w'), indent=1)
        best_name = min(rows, key=lambda r: r[1]['logloss'])[0]
        print('search best:', best_name)
        models[best_name.split('_')[0]] = grid[best_name]
    nfold = min(5, len(set(g)))
    gkf = GroupKFold(n_splits=nfold)
    res = {}
    oof = {}
    for name, mk in models.items():
        p = np.zeros(len(y))
        for tr, te in gkf.split(X, y, g):
            m = mk(); m.fit(X[tr], y[tr]); p[te] = m.predict_proba(X[te])[:, 1]
        res[name] = metrics(y, p); oof[name] = p
        print(name, {k: round(v, 4) for k, v in res[name].items()})
    base = metrics(y, np.full(len(y), y.mean()))
    print('constant', {k: round(v, 4) for k, v in base.items()})
    # calibration table for best model
    best = min(res, key=lambda k: res[k]['logloss']) if a.model == 'auto' else a.model
    p = oof[best]
    print('best =', best)
    bins = np.linspace(0, 1, 6)
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p >= lo) & (p < hi)
        if m.any():
            print(f'  pred[{lo:.1f},{hi:.1f}) n={m.sum()} obs={y[m].mean():.2f} mean_pred={p[m].mean():.2f}')
    # export
    mdl = models[best](); mdl.fit(X, y)
    export(mdl, best, a.out)
    print('saved', a.out)


def export(mdl, kind, out):
    if kind == 'lgbm':
        txt = mdl.booster_.model_to_string()
        json.dump({'type': 'lgbm', 'feats': FEATS, 'model': txt}, open(out, 'w'))
        return
    scaler = mdl.steps[0][1]
    est = mdl.steps[1][1]
    d = {'type': kind, 'feats': FEATS, 'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist()}
    if kind == 'logreg':
        d['coef'] = est.coef_[0].tolist(); d['intercept'] = float(est.intercept_[0])
    else:
        d['W'] = [w.tolist() for w in est.coefs_]; d['b'] = [b.tolist() for b in est.intercepts_]
    json.dump(d, open(out, 'w'))


if __name__ == '__main__':
    main()
