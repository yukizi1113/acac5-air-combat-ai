import pickle,numpy as np,math,pandas as pd
M=pickle.load(open('matches.pkl','rb')); df=pd.read_pickle('df2.pkl')
rows=[]
for mi,m in enumerate(M):
    if m['result']!='Lose' or m['reason']!='ELIMINATION': continue
    X=m['X'];T=len(X)
    if T<m['T']-5: continue   # logs stop when our fighters are all dead
    ex,ey=X[:,42],X[:,43]
    # enemy tracks within 120 km of our escort: first time, and status in final 60 s
    first=None;fs=None
    for t in range(T):
        for k in range(8):
            o=44+8*k
            if X[t,o]==1 and X[t,o+7]==0:
                d=math.hypot(X[t,o+1]-ex[t],X[t,o+2]-ey[t])
                if d<120 and first is None: first=t;fs=(X[t,o+6],X[t,o+3],X[t,o+4],d)
    # last 60s: any track within 70km of escort? src
    last=[]
    for t in range(max(0,T-90),T):
        for k in range(8):
            o=44+8*k
            if X[t,o]==1 and X[t,o+7]==0:
                d=math.hypot(X[t,o+1]-ex[t],X[t,o+2]-ey[t])
                if d<70: last.append((t,X[t,o+6],round(d),X[t,o+3]))
    own=[(i,round(math.hypot(X[first or 0,2+10*i]-ex[first or 0],X[first or 0,3+10*i]-ey[first or 0]))) for i in range(4) if X[first or 0,1+10*i]==1]
    # our escort-direction y vs enemy escort: lateral offset
    rows.append(dict(mi=mi,model=m['model'],T=T,alive=int(sum(X[-1,1+10*i] for i in range(4))),t_first120=first,first_src=fs[0] if fs else None,first_alt=fs[1] if fs else None,
        first_V=fs[2] if fs else None,own_d_at_first=sorted(d for _,d in own),n_last=len(last),last_src=sorted(set(l[1] for l in last)),last_dmin=min([l[2] for l in last]) if last else None,
        dee_end=min([math.hypot(X[-1,2+10*i]-X[-1,157],X[-1,3+10*i]-X[-1,158]) for i in range(4) if X[-1,1+10*i]==1] or [999])))
D=pd.DataFrame(rows); pd.set_option('display.width',250); pd.set_option('display.max_rows',100)
print(D.round(0).to_string())
print('n',len(D),'median T',D['T'].median(),'enemy within 120km first at',D.t_first120.median(),'alive',D.alive.mean())
