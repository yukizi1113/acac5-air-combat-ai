import pickle,numpy as np,math,pandas as pd,collections
M=pickle.load(open('matches.pkl','rb')); df=pd.read_pickle('df2.pkl')
BEH={}
recs=[]
for mi,m in enumerate(M):
    X=m['X']
    for i in range(4):
        a=X[:,1+10*i]; dz=np.where((a[:-1]==1)&(a[1:]==0))[0]
        if not len(dz): continue
        t=int(dz[0])  # last alive row
        px,py,alt,V,az=X[t,2+10*i],X[t,3+10*i],X[t,4+10*i],X[t,5+10*i],X[t,6+10*i]
        # nearest enemy fighter track at t-20..t
        best=(999,None,None,None)
        for tt in range(max(0,t-30),t+1):
            for k in range(8):
                o=44+8*k
                if X[tt,o]!=0 and X[tt,o+7]==0:
                    d=math.hypot(X[tt,o+1]-X[tt,2+10*i],X[tt,o+2]-X[tt,3+10*i])
                    if d<best[0]: best=(d,X[tt,o+6],X[tt,o+3],tt)
        # launches by this fighter before death
        r=X[:t+1,7+10*i]; nl=int(r[0]-r[-1])
        mws=X[max(0,t-15):t+1,9+10*i-1].max()
        beh=X[max(0,t-10):t+1,8+10*i]
        recs.append(dict(mi=mi,res=m['result'],model=m['model'],t=t,alt=alt,V=V,mws15=mws,beh=collections.Counter(beh.astype(int).tolist()).most_common(1)[0][0],
             near=best[0],near_src=best[1],near_alt=best[2],nl=nl,x=px))
R=pd.DataFrame(recs); R.to_pickle('deaths.pkl')
pd.set_option('display.width',200)
print(len(R)); print(R.describe().round(1).T)
print('near<40km',(R.near<40).mean(),'src',R[R.near<60].near_src.value_counts().to_dict(),'no track within 60km',(R.near>=60).mean())
print('mws in last15s',(R.mws15>0).mean()); print('beh',R.beh.value_counts().to_dict())
print('fired before death',(R.nl>0).mean())
print(R.groupby(pd.cut(R.t,[0,200,250,300,400,2000])).size())
