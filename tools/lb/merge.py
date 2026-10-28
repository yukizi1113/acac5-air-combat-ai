import pickle,numpy as np,math,pandas as pd
M=pickle.load(open('matches.pkl','rb')); df=pd.read_pickle('df2.pkl'); S=pd.read_pickle('shots.pkl')
rows=[]
for mi,m in enumerate(M):
    X=m['X'];T=len(X)
    tc=None;dmin_t=[]
    for t in range(T):
        best=999
        for i in range(4):
            b=1+10*i
            if X[t,b]!=1: continue
            for k in range(8):
                o=44+8*k
                if X[t,o]==1 and X[t,o+7]==0 and X[t,o+6]==1:
                    best=min(best,math.hypot(X[t,o+1]-X[t,b+1],X[t,o+2]-X[t,b+2]))
        dmin_t.append(best)
    dmin_t=np.array(dmin_t)
    t60=int(np.argmax(dmin_t<60)) if (dmin_t<60).any() else None
    t50=int(np.argmax(dmin_t<50)) if (dmin_t<50).any() else None
    s=S[S.mi==mi]; t_shot=s.t.min() if len(s) else None
    r=df.iloc[mi]
    rows.append(dict(mi=mi,res=r.res,t60=t60,t50=t50,t_shot1=t_shot,fd=r.first_death,dt_shot=(t_shot-t60) if (t_shot is not None and t60 is not None) else None,
                     dt_death=(r.first_death-t60) if (r.first_death==r.first_death and r.first_death is not None and t60 is not None) else None))
G=pd.DataFrame(rows); G.to_pickle('merge.pkl')
print(G.groupby('res')[['t60','t50','t_shot1','fd','dt_shot','dt_death']].median())
G['we_first']=(G.t_shot1<G.fd-35)|(G.fd.isna()&G.t_shot1.notna())
print(pd.crosstab(G.res,G.we_first,normalize='columns').round(2), pd.crosstab(G.res,G.we_first))
