"""Shared settings for the RL pipeline (agents/TacR)."""
V17 = {"attack_alt": 11000, "cruise_alt": 11000, "escort_max_dist": 90000, "k_escort": 1.2, "pk_thr_escort": 0.55,
       "race_tte": 99999, "engage_radius": 12000, "tc_hold": 6, "tc_fire_first": True, "hot_pk": 0.0, "hot_dist": 65000,
       "hot_deg": 35, "hot_rn": 0.3, "gk_fixed": True, "cap_legs": True, "cap_leg": 15000, "gk_commit": True,
       "loft_alt": 15500, "hot_low_alt": 5000, "hot_low_dist": 28000, "hot_up_dz": 500, "hot_up_pk": 0.25,
       "gk_release_t": 900}

# league opponents (rule agents; overrides as in snowpack/acac/make_jobs.py)
P2 = {"attack_alt": 13500, "cruise_alt": 13000, "escort_max_dist": 90000, "k_escort": 1.2, "pk_thr_escort": 0.55,
      "race_tte": 99999, "engage_radius": 12000}
L1 = dict(P2, tc_hold=6, tc_fire_first=True, hot_pk=0.0, hot_dist=65000, hot_deg=35, hot_rn=0.3)
V15 = dict(L1, gk_fixed=True, cap_legs=True, cap_leg=15000, gk_commit=True, loft_alt=15500)
V16 = dict(V15, hot_low_alt=5000, hot_low_dist=28000)
F10 = dict(V16, attack_alt=10000, cruise_alt=10000)
SN = {"n_interceptors": 0, "n_defenders_max": 0, "engage_radius": 0.0, "race_tte": 99999, "attack_alt": 13500,
      "cruise_alt": 13000, "pk_thr_escort": 0.45, "escort_max_dist": 80000}
SCR = dict(L1, turtle_t=700, turtle_kills=2, turtle_cap_x=-55000)
LOW = {"attack_alt": 3000, "cruise_alt": 3000, "n_interceptors": 0, "race_tte": 99999, "engage_radius": 12000,
       "escort_max_dist": 60000, "pk_thr_escort": 0.5}
LEAGUE = {
    'v17': {'module': 'agents/snap_v17e', 'args': {'overrides': V17}},
    'v15': {'module': 'agents/snap_v15w', 'args': {'overrides': V15}},
    'f10': {'module': 'agents/snap_v17a', 'args': {'overrides': F10}},
    'low': {'module': 'agents/snap_v9d', 'args': {'overrides': LOW}},
    'p2': {'module': 'agents/snap_v9d', 'args': {'overrides': P2}},
    'sn': {'module': 'agents/snap_v8f', 'args': {'overrides': SN}},
    'scr': {'module': 'agents/snap_v13a', 'args': {'overrides': SCR}},
    'v7': {'module': 'agents/snap_v7', 'args': {}},
    'bm': {'module': 'Agents/BenchMark', 'args': {}},
}
