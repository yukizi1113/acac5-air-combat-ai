"""Job lists for the Snowflake runs (see SNOWFLAKE_PLAN.md).  python make_jobs.py -> jobs/*.jsonl"""
import json, os

P2 = {"attack_alt": 13500, "cruise_alt": 13000, "escort_max_dist": 90000, "k_escort": 1.2, "pk_thr_escort": 0.55,
      "race_tte": 99999, "engage_radius": 12000}
L1 = dict(P2, tc_hold=6, tc_fire_first=True, hot_pk=0.0, hot_dist=65000, hot_deg=35, hot_rn=0.3)
V15 = dict(L1, gk_fixed=True, cap_legs=True, cap_leg=15000, gk_commit=True, loft_alt=15500)
V16 = dict(V15, hot_low_alt=5000, hot_low_dist=28000)
V17 = dict(V16, attack_alt=11000, cruise_alt=11000, hot_up_dz=500, hot_up_pk=0.25, gk_release_t=900)
F10 = dict(V16, attack_alt=10000, cruise_alt=10000)
SN = {"n_interceptors": 0, "n_defenders_max": 0, "engage_radius": 0.0, "race_tte": 99999, "attack_alt": 13500,
      "cruise_alt": 13000, "pk_thr_escort": 0.45, "escort_max_dist": 80000}
SCR = dict(L1, turtle_t=700, turtle_kills=2, turtle_cap_x=-55000)
LOW = {"attack_alt": 3000, "cruise_alt": 3000, "n_interceptors": 0, "race_tte": 99999, "engage_radius": 12000,
       "escort_max_dist": 60000, "pk_thr_escort": 0.5}

ME = {'v15': ('agents/snap_v15w', V15), 'v17': ('agents/snap_v17e', V17)}
OPP = {'v15': ('agents/snap_v15w', V15), 'p2': ('agents/snap_v9d', P2), 'sn': ('agents/snap_v8f', SN),
       'scr': ('agents/snap_v13a', SCR), 'f10': ('agents/snap_v17a', F10), 'low': ('agents/snap_v9d', LOW),
       'v7': ('agents/snap_v7', {}), 'bm': ('Agents/BenchMark', None)}


def jobs_for(tag, me, opp, seed0, n):
    (mp, mov), (op, oov) = ME[me], OPP[opp]
    out = []
    for i in range(n):
        seed = seed0 + i // 2
        color = 'Red' if i % 2 == 0 else 'Blue'
        out.append({'tag': tag, 'me': mp, 'me_ov': mov, 'opp': op, 'opp_ov': oov if oov is not None else {},
                    'opp_args_raw': oov is None, 'seed': seed, 'color': color, 'idx': i})
    return out


def write(name, jobs):
    os.makedirs('jobs', exist_ok=True)
    with open(f'jobs/{name}.jsonl', 'w') as f:
        for j in jobs:
            f.write(json.dumps(j) + '\n')
    print(name, len(jobs))


if __name__ == '__main__':
    # phase 1: smoke + cross-machine determinism (first 6 jobs reproduce local results N_v17_v15 seeds 41000-41002)
    write('phase1', jobs_for('det_v17_v15', 'v17', 'v15', 41000, 6) + jobs_for('smoke_v15_p2', 'v15', 'p2', 9100, 6))
    # phase 2: large paired evaluation, in tranches so it can stop early (seed blocks never overlap)
    ph2 = []
    for tr in range(4):                      # 4 tranches x 96 = 384 head-to-head games
        ph2 += jobs_for(f'h2h_v17_v15', 'v17', 'v15', 50000 + 1000 * tr, 96)
    for opp in ['p2', 'sn', 'scr', 'f10', 'low', 'v7']:
        for me in ['v15', 'v17']:
            ph2 += jobs_for(f'{me}_{opp}', me, opp, 60000, 96)   # same seeds for v15 and v17 (paired)
    ph2.sort(key=lambda j: (j['idx'] // 16, j['tag']))   # all comparisons advance together (early stop is fair)
    write('phase2', ph2)
