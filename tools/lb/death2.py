import pickle,numpy as np,math,pandas as pd
M=pickle.load(open('matches.pkl','rb'))
recs=[]
def tracks(X,t,radar_only=False):
    out=[]
    for k in range(8):
        o=44+8*k
        if X[t,o]==1 and X[t,o+7]==0 and (not radar_only or X[t,o+6]==1):
            out.append((X[t,o+1],X[t,o+2],X[t,o+3],X[t,o+4],X[t,o+5],X[t,o+6]))
    return out
for mi,m in enumerate(M):
    X=m['X']
    for i in range(4):
        b=1+10*i
        a=X[:,b]; dz=np.where((a[:-1]==1)&(a[1:]==0))[0]
        if not len(dz): continue
        t=int(dz[0])
        mws_t=np.where(X[:t+1,b+8]>0)[0]; t_mws=int(mws_t[0]) if len(mws_t) else None
        # first MWS of the final episode: walk back from t while mws flag on
        k=t
        while k>0 and X[k-1,b+8]>0: k-=1
        row={}
        for lag in [0,20,30,40,50]:
            tt=max(0,t-lag)
            tr=tracks(X,tt,True); trall=tracks(X,tt)
            px,py=X[tt,b+1],X[tt,b+2]
            d=[math.hypot(e[0]-px,e[1]-py) for e in tr]; da=[math.hypot(e[0]-px,e[1]-py) for e in trall]
            row[f'dR{lag}']=min(d) if d else np.nan; row[f'dA{lag}']=min(da) if da else np.nan
        row.update(mi=mi,res=m['result'],t=t,mws_on=t-k,rem=X[t,b+6],rem0=X[0,b+6],launchable=X[max(0,t-40):t+1,b+9].mean(),
                   fired=X[0,b+6]-X[t,b+6],alt=X[t,b+3])
        recs.append(row)
R=pd.DataFrame(recs); R.to_pickle('deaths2.pkl')
pd.set_option('display.width',200)
print(R.describe(percentiles=[.25,.5,.75]).round(1).T)
nf=R[R.fired==0]
print('never fired',len(nf),'/',len(R))
print('never fired: radar track within 60km at t-40:',(nf.dR40<60).mean().round(2),' within 50:',(nf.dR40<50).mean().round(2),'median dR40',nf.dR40.median(), 'dA40',nf.dA40.median())
print('MWS warning duration before death (s):',R.mws_on.describe().round(1).to_dict())
