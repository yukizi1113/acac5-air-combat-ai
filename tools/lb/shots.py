import pickle,numpy as np,math,pandas as pd
M=pickle.load(open('matches.pkl','rb'))
def ents(X,t):
    out=[]
    for k in range(8):
        o=44+8*k
        if X[t,o]!=0: out.append(dict(alive=X[t,o]==1,x=X[t,o+1],y=X[t,o+2],alt=X[t,o+3],V=X[t,o+4],az=X[t,o+5],src=X[t,o+6],esc=X[t,o+7]==1))
    return out
recs=[]
for mi,m in enumerate(M):
    X=m['X'];T=len(X)
    for i in range(4):
        b=1+10*i; r=X[:,b+6]; a=X[:,b]
        for t in np.where((r[1:]<r[:-1])&(a[1:]==1))[0]+1:
            px,py,alt,V,az=X[t,b+1],X[t,b+2],X[t,b+3],X[t,b+4],math.radians(X[t,b+5])
            E=[e for e in ents(X,t-1) if e['alive'] and not e['esc']]
            best=None
            for e in E:
                dx,dy=e['x']-px,e['y']-py;d=math.hypot(dx,dy)
                off=abs((math.atan2(dy,dx)-az+math.pi)%(2*math.pi)-math.pi)
                sc=math.degrees(off)+d
                if best is None or sc<best[0]: best=(sc,e,d,math.degrees(off))
            # escort candidate
            dx,dy=X[t,157]-px,X[t,158]-py; de=math.hypot(dx,dy); offe=math.degrees(abs((math.atan2(dy,dx)-az+math.pi)%(2*math.pi)-math.pi))
            if best is None or (offe+de<best[0] and offe<30): continue  # escort shot
            _,e,d,off=best
            # target aspect: angle between target velocity and LOS target->shooter (0 = hot)
            los=math.atan2(py-e['y'],px-e['x']); asp=abs((math.radians(e['az'])-los+math.pi)%(2*math.pi)-math.pi) if e['V']>50 else np.nan
            # follow the target track by continuity
            cx,cy=e['x'],e['y']; out='lost'; tend=None
            for tt in range(t,min(T,t+120)):
                Es=ents(X,tt)
                if not Es: continue
                c=min(Es,key=lambda q:math.hypot(q['x']-cx,q['y']-cy))
                dd=math.hypot(c['x']-cx,c['y']-cy)
                if dd>15: continue
                cx,cy=c['x'],c['y']
                if not c['alive']: out='kill'; tend=tt; break
            if out=='lost' and t+120>=T: out='end'
            # did shooter die within 60s
            dz=np.where(a[t:min(T,t+60)]==0)[0]
            recs.append(dict(mi=mi,res=m['result'],model=m['model'],t=t,d=d,off=off,asp=math.degrees(asp) if asp==asp else np.nan,
                alt=alt,talt=e['alt'],src=e['src'],tV=e['V'],out=out,tk=(tend-t) if tend else np.nan,shooter_dead60=len(dz)>0))
S=pd.DataFrame(recs); S.to_pickle('shots.pkl')
print(len(S), S.out.value_counts().to_dict())
S['kill']=(S.out=='kill').astype(int)
print(S.groupby(pd.cut(S.d,[0,20,30,40,50,60,80,200])).agg(n=('kill','size'),pk=('kill','mean'),asp=('asp','median'),tk=('tk','median')).round(2))
print(S.groupby(pd.cut(S.asp,[0,30,60,90,120,180])).agg(n=('kill','size'),pk=('kill','mean'),d=('d','median')).round(2))
print(S.groupby('src').agg(n=('kill','size'),pk=('kill','mean')).round(2))
print('shooter died within 60s',S.shooter_dead60.mean().round(2))
