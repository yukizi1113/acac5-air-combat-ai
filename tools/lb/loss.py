import pickle,numpy as np,math,pandas as pd
M=pickle.load(open('matches.pkl','rb')); df=pd.read_pickle('df.pkl')
out=[]
for m,r in zip(M,df.itertuples()):
    X=m['X']; T=len(X)
    # our escort position track
    ex,ey=X[:,42],X[:,43]
    # enemy fighter tracks near our escort over time (any src)
    near=[];
    for t in range(T):
        if X[t,41]!=1: continue
        best=999
        for k in range(8):
            o=44+8*k
            if X[t,o]==1 and X[t,o+7]==0:
                d=math.hypot(X[t,o+1]-ex[t],X[t,o+2]-ey[t]); best=min(best,d)
        near.append(best)
    near=np.array(near)
    t100=int(np.argmax(near<100)) if (near<100).any() else None
    t60=int(np.argmax(near<60)) if (near<60).any() else None
    # our fighters: alive count at end-30, distance of nearest own fighter to own escort at end
    alive_end=sum(X[T-2,1+10*i]==1 for i in range(4))
    dmin_own=min([math.hypot(X[T-2,2+10*i]-ex[T-2],X[T-2,3+10*i]-ey[T-2]) for i in range(4) if X[T-2,1+10*i]==1] or [999])
    # did we see an enemy track within 60km of our escort in the last 120s?
    seen_last=(near[-120:]<60).any() if len(near) else False
    # enemy escort distance from our nearest fighter at end
    dee=min([math.hypot(X[T-2,2+10*i]-X[T-2,157],X[T-2,3+10*i]-X[T-2,158]) for i in range(4) if X[T-2,1+10*i]==1] or [999])
    out.append(dict(t100=t100,t60=t60,alive_end=alive_end,dmin_own=dmin_own,seen_last=seen_last,dee=dee))
df=pd.concat([df,pd.DataFrame(out)],axis=1); df.to_pickle('df2.pkl')
pd.set_option('display.width',250); pd.set_option('display.max_columns',40); pd.set_option('display.max_rows',200)
L=df[df.res=='Lose']
print(L[['model','T','my_lost','en_lost','n_esc','t_esc1','deaths','alive_end','t100','t60','seen_last','dmin_own','dee','en_alt','en_V','en_minx']].round(0).to_string())
