import json, glob, numpy as np, sys
EP = sorted(glob.glob('/home/stefan/Documents/CodingProjects/GraphOfLife/.claude/worktrees/notebook/research/efficient/A2/a2_rows_12*_12*.json'))
EP = [f for f in EP if int(f.split('_')[-1][:5]) - int(f.split('_')[-2]) == 25]
K = ['n_start', 'tok_before', 'own', 'kin_geno', 'kin_lineage', 'prey', 'lost_prey', 'forfeited', 'estate', 'tok_end', 'repro_net']
G = {g: {k: 0.0 for k in K} for g in ('winners', 'losers', 'rest', 'world')}
nl = {'winners': 0, 'losers': 0, 'rest': 0}; bad = 0; worst = 0
ep_iters = 25
for f in EP:
    rows = json.load(open(f))['rows']
    for r in rows:
        r['grow'] = r['end'] - r['start']
        lhs = r['tok_end'] - r['tok_before']; rhs = r['prey'] - r['lost_prey'] - r['forfeited'] + r['estate']
        worst = max(worst, abs(lhs - rhs))
        if abs(lhs - rhs) > 1e-6: bad += 1
    rows.sort(key=lambda r: -r['grow'])
    win = [r for r in rows[:10] if r['grow'] > 0]; lose = [r for r in rows[::-1][:10] if r['grow'] < 0]
    wid = {r['L'] for r in win}; lid = {r['L'] for r in lose}
    for r in rows:
        g = 'winners' if r['L'] in wid else 'losers' if r['L'] in lid else 'rest'
        nl[g] += 1
        for k in K: G[g][k] += r[k]; G['world'][k] += r[k]
print('epochs', len(EP), 'identity violations', bad, 'max abs diff', worst)
print('lineage counts', nl)
print('per agent per iteration (agents = nodes at start of each game)')
hdr = ['group', 'agent-iters', 'tok/agent', 'own', 'kin(same genotype)', 'kin(same lineage)', 'PREY', 'LOST(to other lineage)', 'forfeit(dead nodes)', 'ESTATE', 'net (end-before)', 'estate/prey']
print(' | '.join(hdr))
for g, d in G.items():
    n = d['n_start']
    print(' | '.join([g, f'{n:.0f}', f"{d['tok_before']/n:.3f}"] + [f"{d[k]/n:.4f}" for k in ('own', 'kin_geno', 'kin_lineage', 'prey', 'lost_prey', 'forfeited', 'estate')] + [f"{(d['tok_end']-d['tok_before'])/n:.4f}", f"{d['estate']/d['prey']:.1f}" if d['prey'] else 'inf']))
json.dump(G, open('/home/stefan/Documents/CodingProjects/GraphOfLife/.claude/worktrees/notebook/research/efficient/A2/a2_table.json', 'w'), indent=1)
