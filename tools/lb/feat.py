import pickle,numpy as np,math,collections
M=pickle.load(open('matches.pkl','rb'))
rows=[]
for m in M:
    X=m['X']; own=m['own']; oth='Red' if own=='Blue' else 'Blue'
    res=m['result']; T=m['T']
    my_esc=0 if (res=='Lose' and m['reason']=='ELIMINATION') else 1
    en_esc=0 if (res=='Win' and m['reason']=='ELIMINATION') else 1
    my_f=int(m['alive'][own]-my_esc); en_f=int(m['alive'][oth]-en_esc)
    deaths=[]
    for i in range(4):
        a=X[:,1+10*i]
        dz=np.where((a[:-1]==1)&(a[1:]==0))[0]
        if len(dz): deaths.append(int(dz[0]+1))
    # launches
    L=[]
    for i in range(4):
        r=X[:,7+10*i]; a=X[:,1+10*i]
        for t in np.where((r[1:]<r[:-1])&(a[1:]==1))[0]:
            t=t+1; px,py=X[t,2+10*i],X[t,3+10*i]; az=math.radians(X[t,6+10*i])
            cands=[('esc',X[t,157],X[t,158])]
            for k in range(8):
                o=44+8*k
                if X[t,o]==1 and X[t,o+7]==0: cands.append(('ftr',X[t,o+1],X[t,o+2]))
            best=None
            for typ,ex,ey in cands:
                dx,dy=ex-px,ey-py; d=math.hypot(dx,dy)
                off=abs((math.atan2(dy,dx)-az+math.pi)%(2*math.pi)-math.pi)
                sc=off+ d/400
                if best is None or sc<best[0]: best=(sc,typ,d,math.degrees(off))
            L.append((t,i,best[1],best[2],X[t,4+10*i]))
    esc_shots=[l for l in L if l[2]=='esc']; ftr_shots=[l for l in L if l[2]=='ftr']
    # enemy fighter track stats (radar)
    alts=[];Vs=[];minx=999;first_contact=None
    for k in range(8):
        o=44+8*k
        sel=(X[:,o]==1)&(X[:,o+6]==1)&(X[:,o+7]==0)
        if sel.any():
            alts+=list(X[sel,o+3]); Vs+=list(X[sel,o+4]); minx=min(minx,X[sel,o+1].min())
            fc=np.where(sel)[0][0]; first_contact=fc if first_contact is None else min(first_contact,fc)
    myalt=[X[X[:,1+10*i]==1,4+10*i].mean() for i in range(4)]
    rows.append(dict(model=m['model'],own=own,res=res,T=T,reason=m['reason'],score=m['scores'][own],my_f=my_f,en_f=en_f,
        my_esc=my_esc,en_esc=en_esc,deaths=sorted(deaths),n_launch=len(L),n_esc=len(esc_shots),
        t_esc1=esc_shots[0][0] if esc_shots else None,d_esc1=esc_shots[0][3] if esc_shots else None,
        n_ftr=len(ftr_shots),t_ftr1=ftr_shots[0][0] if ftr_shots else None,
        en_alt=np.median(alts) if alts else None,en_alt_hi=np.percentile(alts,90) if alts else None,en_V=np.median(Vs) if Vs else None,en_minx=minx,fc=first_contact,
        my_alt=np.mean(myalt)))
pickle.dump(rows,open('rows.pkl','wb'))
import pandas as pd
df=pd.DataFrame(rows)
df['first_death']=df.deaths.apply(lambda d:d[0] if d else None)
df['my_lost']=4-df.my_f; df['en_lost']=4-df.en_f
pd.set_option('display.width',250)
print(df.groupby('res').agg(n=('T','size'),T=('T','median'),my_lost=('my_lost','mean'),en_lost=('en_lost','mean'),nl=('n_launch','mean'),
   n_esc=('n_esc','mean'),t_esc1=('t_esc1','median'),d_esc1=('d_esc1','median'),n_ftr=('n_ftr','mean'),t_ftr1=('t_ftr1','median'),fd=('first_death','median'),
   en_alt=('en_alt','median'),en_hi=('en_alt_hi','median'),en_V=('en_V','median'),minx=('en_minx','median'),fc=('fc','median')).round(1))
df.to_pickle('df.pkl')
