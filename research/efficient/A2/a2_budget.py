"""A2: where do tokens come from? Budget per lineage over iterations 12000..12499 (read-only).
Lineage = genotype (brain id) present at the start of the game of iteration 12000, plus all its
descendants found through parent_brain_ids. Run with GOL_RUNS_DIR set."""
import sys, json, collections, time
sys.path.insert(0, '/home/stefan/Documents/CodingProjects/GraphOfLife/.claude/worktrees/notebook')
import numpy as np
import gol_store
RUN = 'B1-10000-085a32-s4801'
T0, T1 = int(sys.argv[1]), int(sys.argv[2])
F = ['n_start', 'tok_before', 'own', 'kin_geno', 'kin_lineage', 'prey', 'lost_kin_lineage', 'lost_prey', 'forfeited', 'estate', 'tok_end', 'n_end', 'prev_end', 'repro_net']
acc = collections.defaultdict(lambda: collections.defaultdict(float))
world = collections.defaultdict(float)
agent_lost = 0.0; agent_stake_on_self = 0.0; agent_iters = 0
A0 = gol_store.read_frame(RUN, 2 * T0)
S = set(A0['brain_ids']); root = {b: b for b in S}; unresolved = set()
def learn(fr):
    for b, p in zip(fr['brain_ids'], fr['parent_brain_ids']):
        if b not in root:
            if p in root: root[b] = root[p]
            else: root[b] = b; unresolved.add(b)   # parent never seen: own lineage
start_nodes = collections.Counter(root[b] for b in A0['brain_ids'])
prevB_tok = None
t_start = time.time()
for t in range(T0, T1):
    A = gol_store.read_frame(RUN, 2 * t); B = gol_store.read_frame(RUN, 2 * t + 1)
    learn(A); learn(B)
    Aid = A['ids']; Abr = dict(zip(Aid, A['brain_ids'])); Atok = dict(zip(Aid, A['tokens']))
    Bid = B['ids']; Bbr = dict(zip(Bid, B['brain_ids'])); Btok = dict(zip(Bid, B['tokens']))
    R = {u: root[Abr[u]] for u in Aid}
    wins = {w['node']: w['winner'] for w in B['decisions']['winners']}
    incoming = collections.defaultdict(int)
    for r in B['decisions']['allocations']:
        u = r['agent']; agent_iters += 0
        for v, a in zip(r['targets'], r['alloc']):
            if a <= 0: continue
            incoming[v] += a
            w = wins[v]
            Ls, Lw = R[u], R[w]
            if u == v and w == v: acc[Lw]['own'] += a; world['own'] += a
            elif u == w: acc[Lw]['own'] += a; world['own'] += a
            elif Abr[u] == Abr[w]: acc[Lw]['kin_geno'] += a; world['kin_geno'] += a
            elif Ls == Lw: acc[Lw]['kin_lineage'] += a; world['kin_lineage'] += a
            else:
                acc[Lw]['prey'] += a; world['prey'] += a
                acc[Ls]['lost_prey'] += a; world['lost_prey'] += a
            if u != w:
                agent_lost += a
                if Ls == Lw and Abr[u] != Abr[w]: acc[Ls]['lost_kin_lineage'] += a; world['lost_kin_lineage'] += a
                elif Ls == Lw: pass
    dead = set(Aid) - set(Bid)
    for v in dead:
        if incoming.get(v, 0): acc[R[wins[v]]]['forfeited'] += incoming[v]; world['forfeited'] += incoming[v]
    for v in Bid:
        L = root[Bbr[v]]
        e = Btok[v] - incoming.get(v, 0)
        acc[L]['estate'] += e; world['estate'] += e
        acc[L]['tok_end'] += Btok[v]; acc[L]['n_end'] += 1
    for u in Aid:
        acc[R[u]]['n_start'] += 1; acc[R[u]]['tok_before'] += Atok[u]
    # reproduction phase net per lineage: tokens at start of game minus tokens at end of previous game
    if prevB_tok is not None:
        cur = collections.defaultdict(float)
        for u in Aid: cur[R[u]] += Atok[u]
        for L in set(cur) | set(prevB_tok):
            acc[L]['repro_net'] += cur.get(L, 0) - prevB_tok.get(L, 0); acc[L]['prev_end'] += prevB_tok.get(L, 0)
    pb = collections.defaultdict(float)
    for v in Bid: pb[root[Bbr[v]]] += Btok[v]
    prevB_tok = pb
    world['n_start'] += len(Aid); world['n_end'] += len(Bid); world['tok_before'] += sum(A['tokens']); world['tok_end'] += sum(B['tokens'])
    if t % 50 == 0: print(t, round(time.time() - t_start), 's', len(root), 'genotypes seen,', len(unresolved), 'unresolved', flush=True)
# last frame: lineages at the end
endB = collections.Counter(root[b] for b in B['brain_ids']); endTok = collections.Counter()
for b, k in zip(B['brain_ids'], B['tokens']): endTok[root[b]] += k
res = dict(window=[T0, T1], start_lineages=len(start_nodes), genotypes_seen=len(root), unresolved=len(unresolved),
           world={k: world[k] for k in world}, agent_level_lost_total=agent_lost,
           start_nodes=sum(start_nodes.values()), end_nodes=int(sum(endB.values())),
           lineages_alive_at_end=len(endB), lineages_ever_present=len(acc))
rows = []
for L, d in acc.items():
    rows.append(dict(L=int(L), start=start_nodes.get(L, 0), end=int(endB.get(L, 0)), **{k: d[k] for k in F}))
json.dump(dict(res=res, rows=rows), open(f'/home/stefan/Documents/CodingProjects/GraphOfLife/.claude/worktrees/notebook/research/efficient/A2/a2_rows_{T0}_{T1}.json', 'w'))
print(json.dumps(res, indent=1))
