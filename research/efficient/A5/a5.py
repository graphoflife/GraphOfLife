import sys, numpy as np, json
sys.path.insert(0,'/home/stefan/Documents/CodingProjects/GraphOfLife/.claude/worktrees/notebook')
import gol_store as s
RUNS=["B1-10000-085a32-s4801","B1-10000-f511a5-s4801","B1-10000-1cfaa4-s4801"]
rng=np.random.default_rng(2)
def comps(n,ei,ej,b):
    m=b[ei]==b[ej]; ei=ei[m]; ej=ej[m]
    p=np.arange(n)
    def find(x):
        while p[x]!=x:
            p[x]=p[p[x]]; x=p[x]
        return x
    for a,c in zip(ei,ej):
        ra,rc=find(a),find(c)
        if ra!=rc: p[ra]=rc
    r=np.array([find(x) for x in range(n)])
    return np.bincount(np.unique(r,return_inverse=True)[1])
def stats(sz,n):
    big=sz[sz>=2]
    return dict(nterr=int((sz>=2).sum()),largest=int(sz.max()),share10=float(sz[sz>=10].sum()/n))
out=open('a5.csv','w'); out.write("run,iter,agents,nbrains,edges_used,terr,largest,share10,null_terr,null_largest,null_share10\n")
sizes={}
for r in RUNS:
    it=100
    while True:
        try: f=s.read_frame(r,2*it+1)
        except Exception: break
        if f is None: break
        ids=np.array(f['ids']); n=len(ids); idx={int(x):i for i,x in enumerate(ids)}
        E=[(idx[a],idx[b]) for a,b in f['edges'] if a in idx and b in idx]
        E=np.array(E).reshape(-1,2); ei,ej=E[:,0],E[:,1]
        b=np.unique(np.array(f['brain_ids']),return_inverse=True)[1]
        sz=comps(n,ei,ej,b); a=stats(sz,n)
        nulls=[];nsz=[]
        for _ in range(3):
            z=comps(n,ei,ej,rng.permutation(b)); nulls.append(stats(z,n)); nsz.append(z)
        mean=lambda k:np.mean([x[k] for x in nulls])
        out.write(f"{r},{it},{n},{b.max()+1},{len(E)},{a['nterr']},{a['largest']},{a['share10']:.4f},{mean('nterr'):.1f},{mean('largest'):.1f},{mean('share10'):.4f}\n"); out.flush()
        sizes.setdefault(r,[]).append((it,sz[sz>=2],np.concatenate(nsz)[np.concatenate(nsz)>=2]))
        it+=100
    print(r,it,flush=True)
import pickle; pickle.dump(sizes,open('sizes.pkl','wb'))
