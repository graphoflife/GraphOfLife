import sys, numpy as np
sys.path.insert(0,'/home/stefan/Documents/CodingProjects/GraphOfLife/.claude/worktrees/notebook')
import gol_store as s, gol_record as rec
RUNS=["B1-10000-085a32-s4801","B1-10000-f511a5-s4801","B1-10000-1cfaa4-s4801"]
rng=np.random.default_rng(1)
def H(c):
    c=c[c>0].astype(float); p=c/c.sum(); return float(-(p*np.log(p)).sum())
out=open('/home/stefan/Documents/CodingProjects/GraphOfLife/.claude/worktrees/notebook/research/efficient/A6/a6.csv','w')
out.write("run,iter,agents,tokens_held,tok_per_agent,H,H_random,dist,flow,sh1,sh2,sh3p,sh0\n")
for r in RUNS:
    last=s.load_meta(r) and None
    it=50
    while True:
        try: f=s.read_frame(r,2*it+1)
        except Exception as e: break
        if f is None: break
        t=np.array(f['tokens']); n=len(t); T=int(t.sum())
        h=H(t)
        hr=np.mean([H(np.bincount(rng.integers(0,n,T),minlength=n)) for _ in range(20)])
        fl=f['cleanup']['redistributed']/10000
        out.write(f"{r},{it},{n},{T},{T/n:.4f},{h:.5f},{hr:.5f},{hr-h:.5f},{fl:.5f},{(t==1).mean():.4f},{(t==2).mean():.4f},{(t>=3).mean():.4f},{(t==0).mean():.4f}\n")
        out.flush(); it+=50
    print(r,it,flush=True)
