import csv,pickle,numpy as np
from collections import defaultdict
d=defaultdict(list)
for r in csv.DictReader(open('a5.csv')): d[r['run']].append(r)
S=pickle.load(open('sizes.pkl','rb'))
bins=[2,3,4,6,10,20,50,100,1e9]
for run,rows in d.items():
    n=len(rows);print(run,n,'frames')
    for w in range(4):
        sl=slice(w*n//4,(w+1)*n//4); R=rows[sl]
        g=lambda c:np.mean([float(x[c]) for x in R])
        its=f"{R[0]['iter']}-{R[-1]['iter']}"
        ag=g('agents')
        sz=np.concatenate([x[1] for x in S[run][sl]]); nz=np.concatenate([x[2] for x in S[run][sl]])
        h=np.histogram(sz,bins)[0]/len(R); hn=np.histogram(nz,bins)[0]/(len(R)*3)
        print(f" W{w+1} it {its}: agents {ag:.0f} brains {g('nbrains'):.0f} | terr {g('terr'):.0f} (null {g('null_terr'):.0f}) largest mean {g('largest'):.1f} max {max(float(x['largest']) for x in R):.0f} (null mean {g('null_largest'):.1f}) largest/agents {np.mean([float(x['largest'])/float(x['agents']) for x in R]):.4f} share10 {g('share10'):.3f} (null {g('null_share10'):.3f}) p99 size {np.percentile(sz,99):.0f} (null {np.percentile(nz,99):.0f})")
        print('   per-frame counts by size bin',bins[:-1],'real',np.round(h,1).tolist(),'null',np.round(hn,1).tolist())
