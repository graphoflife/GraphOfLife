"""D1 extras (read-only): core drift, lottery z-scores, territories (B vs N), delay, income per head."""
import sys, json, math, collections
sys.path.insert(0, '/home/stefan/Documents/CodingProjects/GraphOfLife/.claude/worktrees/notebook')
import numpy as np, networkx as nx
import gol_store
RUN = 'B1-10000-085a32-s4801'
T0, T1 = 12000, 12100
out = {}
# --- lottery z: observed wins vs expected over all (node, staker) pairs, per share bin
d = json.load(open('/home/stefan/Documents/CodingProjects/GraphOfLife/.claude/worktrees/notebook/research/efficient/D1/d1_result.json'))
lb = d['lottery_bins']; e = np.array(lb['expected']); o = np.array(lb['observed'])
out['lottery_total_expected'] = float(e.sum()); out['lottery_total_observed'] = float(o.sum())
# --- per-iteration extras
core_next = []; core_foreign_next = []; core_foreign_how = collections.Counter()
terr = []  # (N, B, diam) per same-genotype component at end of game
ages_all = []; diam_big = []
share_largest_stake = []
ownstake = []  # border node owner's stake share vs unit
prevA = None
for t in range(T0, T1):
    A = gol_store.read_frame(RUN, 2 * t); B = gol_store.read_frame(RUN, 2 * t + 1)
    Aid = A['ids']; Abr = dict(zip(Aid, A['brain_ids']))
    GA = nx.Graph(); GA.add_nodes_from(Aid); GA.add_edges_from(map(tuple, A['edges']))
    if prevA is not None:
        pA, pbr, pG, pcore = prevA
        for v in pcore:
            if v in Abr:
                core_next.append(1)
                foreign = [x for x in GA.neighbors(v) if Abr[x] != Abr[v]]
                core_foreign_next.append(bool(foreign))
                if foreign:
                    for x in foreign:
                        if x in pbr and x in pG and pG.has_edge(v, x): core_foreign_how['old neighbour whose genotype id changed (a mutant implanted or child)' if pbr[x] != Abr[x] else 'old neighbour, old genotype'] += 1
                        elif x in pbr: core_foreign_how['new link to an existing node'] += 1
                        else: core_foreign_how['new node (a newborn linked to it)'] += 1
    core = [v for v in Aid if GA.degree(v) > 0 and all(Abr[x] == Abr[v] for x in GA.neighbors(v))]
    prevA = (A, Abr, GA, core)
    # territories at the end of the game
    Bid = B['ids']; Bbr = dict(zip(Bid, B['brain_ids']))
    GB = nx.Graph(); GB.add_nodes_from(Bid); GB.add_edges_from(map(tuple, B['edges']))
    same = nx.Graph(); same.add_nodes_from(Bid)
    same.add_edges_from((u, v) for u, v in GB.edges() if Bbr[u] == Bbr[v])
    for comp in nx.connected_components(same):
        if len(comp) < 2: continue
        bord = sum(1 for v in comp if any(Bbr[x] != Bbr[v] for x in GB.neighbors(v)))
        sub = same.subgraph(comp)
        diam = nx.diameter(sub) if len(comp) <= 400 else None
        terr.append((len(comp), bord, diam))
    ages_all += B['ages']
    if t % 10 == 0:
        big = max(nx.connected_components(GB), key=len); sub = GB.subgraph(big)
        # approx diameter by double sweep
        s = next(iter(big)); far = max(nx.single_source_shortest_path_length(sub, s).items(), key=lambda x: x[1])[0]
        diam_big.append((len(big), max(nx.single_source_shortest_path_length(sub, far).values())))
out['core_checked'] = len(core_next); out['core_got_foreign_neighbour_next_iteration'] = float(np.mean(core_foreign_next))
out['core_foreign_how'] = dict(core_foreign_how)
T = np.array([(n, b) for n, b, _ in terr], float)
out['territories'] = len(T)
for lo, hi in ((2, 3), (3, 6), (6, 12), (12, 30), (30, 100), (100, 10**9)):
    m = (T[:, 0] >= lo) & (T[:, 0] < hi)
    if m.sum(): out[f'terr_N[{lo},{hi})'] = dict(count=int(m.sum()), meanN=float(T[m, 0].mean()), meanB=float(T[m, 1].mean()), BoverN=float(T[m, 1].sum() / T[m, 0].sum()))
m = T[:, 0] >= 5
sl = np.polyfit(np.log(T[m, 0]), np.log(np.maximum(T[m, 1], 1)), 1)
out['loglog_slope_B_vs_N(N>=5)'] = float(sl[0]); out['largest_territory'] = int(T[:, 0].max())
dm = [(n, dd) for n, b, dd in terr if dd is not None and n >= 5]
out['territory_diam_mean(N>=5)'] = float(np.mean([x[1] for x in dm])); out['territory_diam_max'] = int(max(x[1] for x in dm))
a = np.array(ages_all); out['age_quantiles_end_of_game(10,25,50,75,90,99)'] = np.percentile(a, [10, 25, 50, 75, 90, 99]).tolist()
out['whole_graph_largest_piece_N_diam(sample)'] = diam_big[:10]
json.dump(out, open('/home/stefan/Documents/CodingProjects/GraphOfLife/.claude/worktrees/notebook/research/efficient/D1/d1_extra.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
