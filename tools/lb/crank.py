"""Is an enemy 'crank' (hot -> 35-100 deg aspect inside 80 km) a launch signature?
For every friendly fighter and second, track the nearest enemy aspect; detect crank events and measure the
probability that this friendly dies within the next 60 s, versus matched hot-no-crank baselines."""
import pandas as pd,numpy as np,pickle,math
M=pickle.load(open('/home/claude/acac/lb/matches.pkl','rb'))
ev=[]
for mi,m in enumerate(M):
    X=m['X'];T=len(X)
    for i in range(4):
        b=1+10*i
        alive=X[:,b]
        dz=np.where((alive[:-1]==1)&(alive[1:]==0))[0]; td=dz[0] if len(dz) else 10**9
        hist=[]
        for t in range(100,min(T,td+1)):
            fx,fy=X[t,b+1],X[t,b+2]; best=None
            for k in range(8):
                o=44+8*k
                if X[t,o]!=1 or X[t,o+7]!=0 or X[t,o+4]<150: continue
                ex,ey=X[t,o+1],X[t,o+2]; d=math.hypot(fx-ex,fy-ey)
                if best is None or d<best[0]:
                    los=math.atan2(fy-ey,fx-ex); a=abs((math.radians(X[t,o+5])-los+math.pi)%(2*math.pi)-math.pi)
                    best=(d,math.degrees(a))
            hist.append(best)
            if best is None or best[0]>80 or len(hist)<12: continue
            prev=[h for h in hist[-12:-2] if h is not None and h[0]<85]
            if len(prev)<6: continue
            was_hot=np.mean([h[1]<30 for h in prev])>0.7
            if not was_hot: continue
            now=best[1]
            kind='crank' if 35<=now<=100 else ('hot' if now<30 else 'other')
            if kind=='other': continue
            die60= td<=t+60
            ev.append(dict(mi=mi,t=t,d=best[0],kind=kind,die60=die60,res=m['result']))
E=pd.DataFrame(ev)
E=E.sort_values(['mi','t'])
print(E.groupby(['kind',pd.cut(E.d,[0,30,45,60,80])]).die60.agg(['size','mean']).round(3))
E.to_pickle('/home/claude/acac/lb/crank_events.pkl')
