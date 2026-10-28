"""Apply v3 doctrine changes to agents/Tac/agent.py (virtual-missile drag, rush defaults)."""
p = 'agents/Tac/agent.py'
s = open(p).read()

# ---- new parameters
s = s.replace("""            'threat_k': g('threat_k', 0.55),         # enemy normalised range at which we start to drag""",
"""            'threat_k': g('threat_k', -9.0),         # (legacy proximity drag; disabled by default)
            'use_vthreat': g('use_vthreat', True),   # virtual-missile threat estimation
            'vt_k': g('vt_k', 0.40),                 # assumed enemy launch threshold (normalised range)
            'vt_interval': g('vt_interval', 30.0),   # assumed enemy firing cadence per (enemy, us) pair [s]
            'vt_max': g('vt_max', 3),
            'vt_react_dist': g('vt_react_dist', 22000.0),
            'vt_speed_margin': g('vt_speed_margin', 60.0),""")
s = s.replace("""            'engage_radius': g('engage_radius', 70000.0),""", """            'engage_radius': g('engage_radius', 25000.0),""")
s = s.replace("""            'n_defenders_max': g('n_defenders_max', 2),""", """            'n_defenders_max': g('n_defenders_max', 1),""")
s = s.replace("""            'defend_radius': g('defend_radius', 100000.0),""", """            'defend_radius': g('defend_radius', 80000.0),""")
s = s.replace("""            'k_fighter': g('k_fighter', 0.35),""", """            'k_fighter': g('k_fighter', 0.45),""")
s = s.replace("""            'fighter_max_dist': g('fighter_max_dist', 60000.0),""", """            'fighter_max_dist': g('fighter_max_dist', 45000.0),""")
s = s.replace("""            'pk_thr_fighter': g('pk_thr_fighter', 0.35),""", """            'pk_thr_fighter': g('pk_thr_fighter', 0.45),""")
s = s.replace("""            'escort_max_dist': g('escort_max_dist', 65000.0),""", """            'escort_max_dist': g('escort_max_dist', 75000.0),""")

# ---- behaviour id
s = s.replace("""B_NAMES = OPT_NAMES + ['EVADE', 'SUPPORT', 'TDRAG', 'NOAMMO']
B_EVADE, B_SUPPORT, B_TDRAG, B_NOAMMO = 6, 7, 8, 9""", """B_NAMES = OPT_NAMES + ['EVADE', 'SUPPORT', 'TDRAG', 'NOAMMO', 'VDRAG']
B_EVADE, B_SUPPORT, B_TDRAG, B_NOAMMO, B_VDRAG = 6, 7, 8, 9, 10""")

# ---- create virtual simulator in validate
s = s.replace("""        self.names = [p.getFullName() for p in self.parents.values()]
""", """        self.names = [p.getFullName() for p in self.parents.values()]
        from .vthreat import VirtualThreat
        self.vthreat = VirtualThreat(self.P['vt_k'], self.P['vt_interval'], int(self.P['vt_max']), bool(self.P['use_vthreat']))
        self.vthreat.create(self)
""", 1)

# ---- reflex: virtual-missile drag (after MWS evasion, before support)
s = s.replace("""        guided = [m for m in self.our_msls if m['owner'] == nm and m['mode'] == 'GUIDED']
        if f['rem'] <= 0 and not guided:
            return B_NOAMMO, None""", """        guided = [m for m in self.our_msls if m['owner'] == nm and m['mode'] == 'GUIDED']
        vt = self._vthreat_for(f)
        if vt is not None:
            # keep supporting our own missile if it is about to go autonomous
            g_ok = [m for m in guided if norm(m['tpos'] - m['pos']) < 14000]
            if not g_ok:
                return B_VDRAG, vt
        if f['rem'] <= 0 and not guided:
            return B_NOAMMO, None""")
s = s.replace("""    def deploy(self, action):
        P = self.P
        self.ext_action = action""", """    def _vthreat_for(self, f):
        if not self.P['use_vthreat'] or getattr(self, 'vthreat', None) is None:
            return None
        worst = None
        for v in self.vthreat.info.get(f['name'], []):
            if v['hit'] or v['closing'] <= 0:
                continue
            if v['dist'] < self.P['vt_react_dist'] and v['speed'] > f['V'] + self.P['vt_speed_margin']:
                if worst is None or v['dist'] < worst['dist']:
                    worst = v
        return worst

    def sk_vdrag(self, f, v):
        rel = self.T(np.asarray(v['rel'], dtype=np.float64))   # from us to the virtual missile
        return az_of(-rel), self.P['drag_alt'], 'f', 1.0

    def deploy(self, action):
        P = self.P
        self.ext_action = action
        if getattr(self, 'vthreat', None) is not None and self.vthreat.vs is not None:
            crs_tracks = [(e.uid, e.track) for e in self.fighters_e(0.5) if e.track is not None]
            try:
                self.vthreat.step(self, crs_tracks)
            except Exception:
                self.vthreat.vs = None""")
s = s.replace("""            elif beh == B_SUPPORT:
                az, alt, mode, thr = self.sk_support(f, pay)""", """            elif beh == B_SUPPORT:
                az, alt, mode, thr = self.sk_support(f, pay)
            elif beh == B_VDRAG:
                az, alt, mode, thr = self.sk_vdrag(f, pay)""")
open(p, 'w').write(s)
print('patched')
