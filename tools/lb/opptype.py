import pickle,numpy as np,math,pandas as pd
M=pickle.load(open('matches.pkl','rb')); df=pd.read_pickle('df2.pkl')
rows=[]
for mi,m in enumerate(M):
    X=m['X'];T=len(X)
    feats={}
    for tt in [120,160,200,240]:
        if tt>=T: feats[f'r{tt}']=np.nan; feats[f'n{tt}']=np.nan; continue
        ex,ey=X[tt,42],X[tt,43]; nr=0; n=0; alts=[]
        for k in range(8):
            o=44+8*k
            if X[tt,o]==1 and X[tt,o+7]==0 and X[tt,o+4]>100:
                n+=1; vx=X[tt,o+4]*math.cos(math.radians(X[tt,o+5])); vy=X[tt,o+4]*math.sin(math.radians(X[tt,o+5]))
                dx,dy=ex-X[tt,o+1],ey-X[tt,o+2]; c=(dx*vx+dy*vy)/max(1e-6,math.hypot(dx,dy)*math.hypot(vx,vy))
                if c>math.cos(math.radians(30)): nr+=1
                alts.append(X[tt,o+3])
        feats[f'r{tt}']=nr; feats[f'n{tt}']=n; feats[f'alt{tt}']=np.median(alts) if alts else np.nan
    rows.append(feats)
F=pd.DataFrame(rows); D=pd.concat([df,F],axis=1); D.to_pickle('opp.pkl')
D['win']=(D.res=='Win').astype(int)
print(D.groupby('r200').agg(n=('win','size'),win=('win','mean'),my_lost=('my_lost','mean'),en_lost=('en_lost','mean')).round(2))
print(D.groupby(pd.cut(D.alt200,[-5,5,9,11,13,20])).agg(n=('win','size'),win=('win','mean')).round(2))
for mdl in ['acac5_v14_gk','acac5_v9_p2','acac5_v13_lr']:
    d=D[D.model==mdl]; print(mdl, d.groupby(d.r200.clip(0,3)).win.agg(['size','mean']).round(2).to_dict('index'))
