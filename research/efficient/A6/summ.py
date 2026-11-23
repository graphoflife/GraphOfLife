import csv,numpy as np
from collections import defaultdict
d=defaultdict(list)
for r in csv.DictReader(open('a6.csv')): d[r['run']].append(r)
cols=['agents','tokens_held','tok_per_agent','H','H_random','dist','flow','sh1','sh2','sh3p']
for run,rows in d.items():
    n=len(rows);print(run,n,'samples')
    for name,sl in [('start(first 10)',slice(0,10)),('middle',slice(n//2-5,n//2+5)),('end(last 10)',slice(n-10,n))]:
        print(' ',name,' '.join(f"{c}={np.mean([float(x[c]) for x in rows[sl]]):.3f}" for c in cols))
    it=np.array([float(x['iter']) for x in rows])
    for c in ['dist','flow','tok_per_agent','agents']:
        y=np.array([float(x[c]) for x in rows]);m=it>=1000
        sl=np.polyfit(it[m],y[m],1)[0]*10000;print(f'  slope {c} (it>=1000) per 10k it: {sl:.4f}; sd {y[m].std():.4f}; corr(dist,flow)={np.corrcoef([float(x["dist"]) for x in rows],[float(x["flow"]) for x in rows])[0,1]:.2f}' if c=='dist' else f'  slope {c} per 10k it: {sl:.4f}; sd {y[m].std():.4f}')
