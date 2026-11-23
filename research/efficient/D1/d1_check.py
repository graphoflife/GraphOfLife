"""D1: check the laws of the rules on recorded game frames (read-only).
Run: GOL_RUNS_DIR=.../GraphOfLifeRuns python3 d1_check.py
World: B1-10000-085a32-s4801, iterations 12000..12099."""
import sys, json, math, collections
sys.path.insert(0, '/home/stefan/Documents/CodingProjects/GraphOfLife/.claude/worktrees/notebook')
import numpy as np, networkx as nx
import gol_store

RUN = 'B1-10000-085a32-s4801'
T0, T1 = 12000, 12100
fail = collections.defaultdict(list)      # law -> counterexamples
n_ok = collections.Counter()              # law -> cases that held
n_all = collections.Counter()

def check(law, ok, what=None):
    n_all[law] += 1
    if ok: n_ok[law] += 1
    elif len(fail[law]) < 5: fail[law].append(what)

def key(a, b): return (a, b) if a < b else (b, a)

stat = collections.defaultdict(list)
bins = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0001]
lot_exp = np.zeros(10); lot_obs = np.zeros(10); lot_n = np.zeros(10)
kin_exp = np.zeros(10); kin_obs = np.zeros(10); kin_n = np.zeros(10)
ways = collections.defaultdict(list)
core = collections.Counter()
prev = gol_store.read_frame(RUN, 2 * T0 - 1)   # P2 of iteration T0-1
for t in range(T0, T1):
    A = gol_store.read_frame(RUN, 2 * t)       # phase 1 (graph at the start of the game)
    B = gol_store.read_frame(RUN, 2 * t + 1)   # phase 2 (after the game and its cleanup)
    assert A['iteration'] == t and A['phase'] == 1 and B['phase'] == 2
    Aid = A['ids']; Atok = dict(zip(Aid, A['tokens'])); Abr = dict(zip(Aid, A['brain_ids']))
    Bid = B['ids']; Btok = dict(zip(Bid, B['tokens'])); Bbr = dict(zip(Bid, B['brain_ids']))
    Bpar = dict(zip(Bid, B['parent_brain_ids'])); Bage = dict(zip(Bid, B['ages'])); Bparent = dict(zip(Bid, B['parent_ids']))
    GA = nx.Graph(); GA.add_nodes_from(Aid); GA.add_edges_from(map(tuple, A['edges']))
    allocs = B['decisions']['allocations']; wins = B['decisions']['winners']
    # --- L1 conservation
    check('L1 tokens conserved (sum per frame = 10000)', sum(A['tokens']) == 10000 and sum(B['tokens']) == 10000 and B['summary']['tokens'] == 10000,
          (t, sum(A['tokens']), sum(B['tokens'])))
    staked_total = sum(sum(r['alloc']) for r in allocs)
    check('L1b every token is staked once (sum of stakes = tokens at start of game)', staked_total == sum(A['tokens']), (t, staked_total))
    check('L1c record per agent: agent tokens in record = tokens in phase-1 frame; every agent with tokens has a record',
          all(Atok[r['agent']] == r['tokens'] == sum(r['alloc']) for r in allocs) and len(allocs) == sum(1 for x in A['tokens'] if x > 0), t)
    check('L0 after a cleanup nobody holds 0 tokens (phase-1 frame)', min(A['tokens']) >= 1, (t, min(A['tokens'])))
    # --- stakes
    incoming = collections.defaultdict(int); stakers = collections.defaultdict(dict)
    off_target = 0; self_stake = 0; far_stake = 0
    for r in allocs:
        u = r['agent']
        expect = [u] + sorted(GA.neighbors(u))
        check('L2/L4 targets = the agent itself + its neighbours in the graph at the start of the game', r['targets'] == expect, (t, u))
        for v, a in zip(r['targets'], r['alloc']):
            if a > 0:
                incoming[v] += a; stakers[v][u] = a
                check('L2/L4 every stake lands on the staker or a neighbour (one link)', v == u or GA.has_edge(u, v), (t, u, v))
                if v == u: self_stake += a
                else: far_stake += a
        k = len(expect) - 1; tk = r['tokens']
        if tk in (1, 2, 3): ways[(tk, 'k')].append(k)
        for v in r['targets']: pass
    stat['self_stake_share'].append(self_stake / (self_stake + far_stake))
    # --- L3 keep every token + winners
    wmap = {w['node']: w for w in wins}
    check('L3 winners exist exactly for the nodes with some stake', set(wmap) == set(incoming), t)
    dead = set(Aid) - set(Bid)
    starved = {v for v in Aid if incoming.get(v, 0) == 0}
    check('L7a a node with no stake on it starves (and cleanup counts them)', starved <= dead and B['cleanup']['starved'] == len(starved), (t, len(starved), B['cleanup']['starved']))
    estate = {v: Btok[v] - incoming.get(v, 0) for v in Bid}
    check('L3 a survivor holds all that was staked on it, plus estate income (>=0)', min(estate.values()) >= 0, t)
    pool = sum(incoming.get(v, 0) for v in dead)
    check('L1d tokens of the dead = tokens shared out (cleanup.redistributed)', pool == B['cleanup']['redistributed'] and sum(estate.values()) == pool, (t, pool, B['cleanup']['redistributed'], sum(estate.values())))
    # --- winners and lottery
    atkwin = 0; failed_att = 0; failed_tokens = 0; strongest_lost = 0; multi = 0
    for v, w in wmap.items():
        st = stakers[v]
        check('LOT winner is one of the stakers of the node', w['winner'] in st, (t, v, w))
        check('LOT recorded amount = largest single stake', w['amount'] == max(st.values()), (t, v))
        if len(st) >= 2:
            multi += 1
            tot = sum(st.values())
            if st[w['winner']] < max(st.values()): strongest_lost += 1
            for s, a in st.items():
                p = a / tot; b = min(int(p * 10), 9)
                lot_exp[b] += p; lot_obs[b] += (s == w['winner']); lot_n[b] += 1
            # kin groups (same genotype id at start)
            grp = collections.defaultdict(int)
            for s, a in st.items(): grp[Abr[s]] += a
            for g, a in grp.items():
                p = a / tot; b = min(int(p * 10), 9)
                kin_exp[b] += p; kin_obs[b] += (Abr[w['winner']] == g); kin_n[b] += 1
        # owner kept / lost; failed attack
        if v in B['ids'] if False else False: pass
        if v in Btok:
            own = st.get(v, 0)
            others = sum(a for s, a in st.items() if s != v)
            if w['winner'] == v and others > 0:
                failed_att += 1; failed_tokens += others
            if w['winner'] != v: atkwin += 1
            # brain implant
            nb = Bbr[v]; wb = Abr[w['winner']]
            if w['winner'] == v:
                ok = nb == Abr[v]
            else:
                ok = nb == wb or Bpar[v] == wb
            check('L3b the winner implants its brain (same id, or a mutant child of it); the owner keeps its own brain when it wins', ok, (t, v, w))
    stat['nodes_with_2plus_stakers'].append(multi); stat['strongest_lost'].append(strongest_lost)
    stat['node_taken'].append(atkwin); stat['failed_attacks'].append(failed_att); stat['failed_tokens'].append(failed_tokens)
    # --- L7 links: cut those with no stake across and no payment this iteration
    crossed = set()
    for r in allocs:
        u = r['agent']
        for v, a in zip(r['targets'], r['alloc']):
            if a > 0 and v != u: crossed.add(key(u, v))
    paid = {key(a, b) for a, b in A['decisions']['upkeep']}
    Aedges = {key(a, b) for a, b in A['edges']}
    pedges = {key(a, b) for a, b in prev['edges']}
    new_edges = Aedges - pedges
    pred = {e for e in Aedges if e not in crossed and e not in paid and e not in new_edges}
    got = {key(a, b) for a, b in B['decisions']['pruned_edges']}
    check('L7b a link is cut after the game iff no stake crossed it and it was not paid/new this iteration', pred == got, (t, len(pred - got), len(got - pred)))
    check('L7b2 cutting happens in the game (phase 1 cuts nothing)', len(A['decisions']['pruned_edges']) == 0, t)
    # cull
    H = nx.Graph(); H.add_nodes_from(set(Aid) - starved); H.add_edges_from(e for e in Aedges - got if e[0] not in starved and e[1] not in starved)
    comps = sorted(nx.connected_components(H), key=len, reverse=True)
    check('L7c only the largest piece survives the cull', set(Bid) == comps[0] and B['cleanup']['orphaned'] == len(Aid) - len(starved) - len(comps[0]), (t, len(Bid), len(comps[0])))
    Bedges = {key(a, b) for a, b in B['edges']}
    check('L7d surviving links = links not cut, between survivors', Bedges == {e for e in Aedges - got if e[0] in comps[0] and e[1] in comps[0]}, t)
    # prune/orphan stats
    stat['n_start'].append(len(Aid)); stat['n_end'].append(len(Bid)); stat['starved'].append(len(starved)); stat['orphaned'].append(B['cleanup']['orphaned'])
    stat['cut_links'].append(len(got)); stat['links_start'].append(len(Aedges)); stat['links_end'].append(len(Bedges))
    # --- L9 estates: parents rule
    kids = [c for c in Bid if Bage[c] == 0 and Bparent[c] != -1]
    cnt = collections.Counter(Bparent[c] for c in kids)
    heirs = [u for u in Bid if u in cnt]
    recv = {u for u, e in estate.items() if e > 0}
    if heirs:
        check('L9 estate goes only to parents of children born this iteration that live (rule parents)', recv <= set(heirs), (t, len(recv - set(heirs))))
        stat['heirs'].append(len(heirs)); stat['pool_p2'].append(pool)
        # chi-square of draws vs weights
        w = np.array([cnt[u] for u in heirs], float); w /= w.sum()
        x = np.array([estate[u] for u in heirs], float); e = pool * w
        stat['chi2_over_df'].append(float(((x - e) ** 2 / np.maximum(e, 1e-9)).sum() / max(1, len(heirs) - 1)))
    else:
        check('L9 estate goes to all survivors when nobody is an heir', len(recv) > 0 or pool == 0, t)
    stat['pool_p1'].append(A['cleanup']['redistributed'])
    stat['estate_recipients'].append(len(recv))
    # --- tokens per agent
    for name, toks in (('A', A['tokens']), ('B', B['tokens'])):
        c = collections.Counter(min(x, 3) for x in toks); n = len(toks)
        stat[name + '_tok1'].append(c[1] / n); stat[name + '_tok2'].append(c[2] / n); stat[name + '_tok3'].append(c[3] / n)
        stat[name + '_mean'].append(sum(toks) / n)
    # --- core nodes: all neighbours same genotype id; do they get a foreign neighbour next iteration?
    core_now = [v for v in Aid if GA.degree(v) > 0 and all(Abr[x] == Abr[v] for x in GA.neighbors(v))]
    stat['core_nodes'].append(len(core_now))
    atk_core = sum(1 for v in core_now if v in wmap and any(Abr[s] != Abr[v] for s in stakers[v]))
    stat['core_attacked_by_other_genotype'].append(atk_core)
    # node-level: nodes whose all stakers are one genotype: cannot be lost to a stranger
    prev = B
# ---- ways to stake
def ways_n(tk, k): return math.comb(tk + k, k)
W = {}
for tk in (1, 2, 3):
    ks = np.array(ways[(tk, 'k')]); n = np.array([ways_n(tk, int(k)) for k in ks])
    W[tk] = dict(agents=len(ks), mean_k=float(ks.mean()), median_k=float(np.median(ks)), ways_at_mean_k=ways_n(tk, int(round(ks.mean()))), mean_ways=float(n.mean()), median_ways=float(np.median(n)),
                 sequences_at_mean_k=(int(round(ks.mean())) + 1) ** tk)
res = dict(laws={l: dict(held=n_ok[l], of=n_all[l], counterexamples=fail[l]) for l in n_all},
           stats={k: (float(np.mean(v)), float(np.min(v)), float(np.max(v))) for k, v in stat.items()},
           lottery_bins=dict(expected=lot_exp.tolist(), observed=lot_obs.tolist(), n=lot_n.tolist()),
           kin_bins=dict(expected=kin_exp.tolist(), observed=kin_obs.tolist(), n=kin_n.tolist()), ways=W)
json.dump(res, open('/home/stefan/Documents/CodingProjects/GraphOfLife/.claude/worktrees/notebook/research/efficient/D1/d1_result.json', 'w'), indent=1)
for l, d in res['laws'].items(): print(d['held'], '/', d['of'], l, d['counterexamples'] if d['held'] != d['of'] else '')
for k, v in res['stats'].items(): print(k, [round(x, 4) for x in v])
print('lottery', [(round(e, 1), o, n) for e, o, n in zip(lot_exp, lot_obs, lot_n)])
print('kin', [(round(e, 1), o, n) for e, o, n in zip(kin_exp, kin_obs, kin_n)])
print(W)
