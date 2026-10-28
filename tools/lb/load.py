import json,glob,hashlib,os,pickle,numpy as np
D=os.environ.get("LB_LOG_DIR","/mnt/user-data/uploads/空戦AI/ログ")
out=[];seen=set()
for f in sorted(glob.glob(D+'/*.txt')):
    base=os.path.basename(f)[:-4]; model=base[:-13]; stamp=base[-12:]
    if model=='acac5_v7_rule': continue
    d=json.load(open(f))
    for k,m in enumerate(d[1:]):
        lg=m['detail'].get('logs') or {}
        o=lg.get('observations')
        if not o: continue
        X=np.array([e['data']['1'] for e in o],dtype=np.float32)
        h=hashlib.md5(X[:3].tobytes()).hexdigest()
        if h in seen: continue
        seen.add(h)
        out.append(dict(model=model,stamp=stamp,idx=k,own=m['own'],result=m['result'],T=m['detail']['step_count'],
            alive=m['detail']['numAlives'],reason=m['detail']['endReason'],scores=m['detail']['scores'],X=X))
pickle.dump(out,open('/home/claude/acac/lb/matches.pkl','wb'))
print(len(out))
