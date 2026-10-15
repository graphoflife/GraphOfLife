"""
Prototype mechanics on top of the engine, for pilots only.

ProtoWorld is GraphOfLife with:
  * counters the pilots need (births, conquests, revolutions, deaths),
  * founder roots for every brain (which founder's line a genotype belongs to),
  * prototype mechanics, switched by the module dict P (one process = one run).

Mechanics (P keys):
  mor      "off" | "spawn" | "all"  mutation only at replication: at a child's
                                    birth ("spawn"), or also when a conquering
                                    brain is copied into a won node ("all").
                                    Removes the world-wide mutation after each game.
  memory   bool   two input rows per candidate: what it staked on me in the last
                  game, and what I staked on it (log1p tokens).
  seek     bool   edge proposals: two output rows SEEK (how much of my tokens to
                  spend searching) and two ACCEPT (yes/no to a proposed partner).
                  A searcher walks `budget` random steps from its node, paying one
                  token per step; if it ends on a stranger and both accept, they
                  are joined.
  toll     "pool" | "path"   where a searcher's tokens go: shared out uniformly, or
                  paid to the agent each step enters.
  seek_max int    the longest walk (tokens a single search may spend).
  incumbency float  an agent's stake on its own node counts this many times over
                  when the game decides who takes the node (1 = the engine's rule).
  juvenile int    a newborn cannot be taken for this many iterations after its birth.
  triangle_birth bool  a newborn is always joined to its parent and to the parent's
                  neighbour its LINK head favours most, so every birth closes a
                  triangle and no birth makes a leaf.
  birth_links int a newborn is joined to its parent and to this many of the parent's
                  neighbours (those its LINK head favours most), on top of its own choices.
  repair   bool   when an agent dies, its neighbours that were not linked to one another
                  are joined (groups of directly linked neighbours chained through their
                  least-connected members), so a death never splits a neighbourhood.
  safe_prune int  an unused connection lapses only if its two ends share a neighbour
                  (1), or two neighbours (2), so pruning never cuts the only local path
                  between two agents (1), nor leaves them hanging on one (2).
  newborn_home bool  an agent born this iteration stakes everything on its own node in
                  its first game: a child cannot be spent as a missile.
  valence  int    no agent may hold more than this many links (0 = no limit): a full
                  agent cannot have a child, and no new link is made to a full agent.
                  Mending after a death is exempt — it only replaces lost links.
  valence_mode str  "all": the cap also limits a newborn's two physics links;
                  "optional": it limits only the links a brain chooses.
  keep_one bool   an agent's last token always stays on its own node: no agent can
                  stake itself to death.
  memory_len int  the last L games' stakes between an agent and each candidate, both
                  ways, as 2L input rows (most recent first). A brain with a `mem`
                  attribute (grow brains) sees only its first `mem` games.
  max_age  int    an agent dies when it reaches this age (0 = never); its tokens are
                  shared out like any other estate, and its neighbourhood is mended.
  min_child int   the fewest tokens a child can be born with: a parent whose chosen
                  endowment is smaller has no child (1 = the engine's rule).
  window   int    a connection lapses after this many phases without tokens crossing
                  it (0 = the engine's rule, inactive_window).
  yield_tax float cooperation that produces: after each game every holding pays this
                  share into a pool, and the pool is paid out over the connections
                  on which both ends staked on each other, in proportion to the
                  smaller of the two stakes, half to each end. The total is unchanged.
  support_gain float  a supporting stake counts this many times over in its target's
                  defence (1 = at face value; with support).
  allies   bool   two output rows ALLY: does this agent want each neighbour as an ally
                  in this game? Two agents that both want it are allied. Stakes on a
                  node are then pooled by side: the owner and its allies are one side,
                  and other bidders that are allied with each other are one side each.
                  The side with the largest total takes the node — the owner keeps it
                  if its side wins; otherwise the side's largest bidder does. An ally
                  can never take an ally's node: a stake on it is a gift that defends.
  kin_pool int    the same pooling with kin for allies: two agents are on one side if
                  their genotypes are within this many mutations (0 = clones; -1 = off).
  Both pooling rules add one input row: is this candidate on my side (allied in the
  last game, or kin)?
  birth_allies bool  a newborn is also joined to every neighbour its parent was allied
                  with in the last game (how many is the brains' doing, not a number).
  ally_keep bool  a connection between agents allied in the game just played never
                  lapses for want of use; every other unused one lapses as the engine
                  (or safe_prune) says.
  engine   bool   play the engine's own phases, every rule as the run's settings say
                  (incumbency, keep_home, mutate_on_replication, birth_links, ... and
                  the switches gifting, handover, revolutions, redistribution), with
                  only the bookkeeping added. The other keys above are then unused,
                  except the four below. (equiv.py checks it against the engine.)
  contest  str    "plurality" (the engine's: the largest stake takes the node) or
                  "lottery": the node goes to a staker drawn with probability in
                  proportion to its stake as the game weighs it (Tullock's contest).
  pool     str    "" or "genotype": stakes on a node are pooled by genotype — every
                  stake by agents with one brain counts as one side's; the side with
                  most takes the node, kept by the owner if its genotype wins. Adds
                  the input row of the pooling rules (is this candidate my genotype?).
  kin_sense bool  the same input row — is this candidate's brain identical to mine? —
                  without any pooling: an agent can tell its clones, nothing more.
  fission  bool   birth by division: an agent whose brain would give a child at least
                  half its tokens divides into two equal halves (the child takes the
                  lower half); anything less is no child. No smallest child is needed:
                  a child is never poorer than the parent it leaves. (Engine mode.)
"""
from __future__ import annotations

import sys
from typing import Any, Dict, List, Tuple

sys.path.insert(0, "/home/stefan/Documents/CodingProjects/GraphOfLife Claude")

import numpy as np
import networkx as nx

from GraphOfLifeSimple import (GraphOfLife, _apportion, _share_of_first,
                               _choose_binary, stream_seed)
from gol_config import SimConfig

DEFAULTS = {"mor": "off", "memory": False, "seek": False, "toll": "pool", "seek_max": 8,
            "incumbency": 1.0, "juvenile": 0, "yield_tax": 0.0, "window": 0, "min_child": 1,
            "triangle_birth": False, "birth_links": 0, "repair": False, "safe_prune": False,
            "newborn_home": False, "max_age": 0, "valence": 0, "valence_mode": "all",
            "keep_one": False, "memory_len": 0, "keep_frac": 0.0,
            "support": False, "fog": False, "support_gain": 1.0, "allies": False,
            "kin_pool": -1, "birth_allies": False, "ally_keep": False,
            "engine": False, "contest": "plurality", "pool": "", "kin_sense": False,
            "fission": False}
P: Dict[str, Any] = dict(DEFAULTS)


def pooled() -> bool:
    return bool(P.get("allies")) or int(P.get("kin_pool", -1)) >= 0 or P.get("pool") == "genotype"


def set_proto(proto: Dict[str, Any]) -> None:
    P.clear()
    P.update({**DEFAULTS, **proto})


class ProtoConfig(SimConfig):
    """SimConfig whose brain grows the rows the prototype mechanics need."""

    def input_layout(self):
        layout = super().input_layout()
        row = max(stop for _, stop, _ in layout.values())
        if P.get("memory"):
            for name in ("stake_from_it", "stake_to_it"):
                layout[name] = (row, row + 1, "bit")
                row += 1
        for k in range(1, int(P.get("memory_len", 0)) + 1):
            for name in (f"hist_to_{k}", f"hist_from_{k}"):
                layout[name] = (row, row + 1, "bit")
                row += 1
        if pooled() or P.get("kin_sense"):
            layout["same_side"] = (row, row + 1, "bit")
            row += 1
        return layout

    def head_layout(self):
        layout = super().head_layout()
        if P.get("allies"):
            start = layout.pop("MESSAGE")[0]
            layout["ALLY"] = [start, start + 2]
            layout["MESSAGE"] = [start + 2, start + 2 + self.message_amount]
        if P.get("support"):
            start = layout.pop("MESSAGE")[0]
            layout["SUPPORT"] = [start, start + 2]
            layout["MESSAGE"] = [start + 2, start + 2 + self.message_amount]
        if P.get("seek"):
            start = layout.pop("MESSAGE")[0]
            layout["SEEK"] = [start, start + 2]
            layout["ACCEPT"] = [start + 2, start + 4]
            layout["MESSAGE"] = [start + 4, start + 4 + self.message_amount]
        return layout


class ProtoWorld(GraphOfLife):

    def __init__(self, G_init, cfg, _empty=False):
        self.root_of: Dict[int, int] = {}
        # The brain family tree: each genotype's parent and its depth below its founder.
        self.brain_parent: Dict[int, int] = {}
        self.brain_depth: Dict[int, int] = {}
        # Ages at death, collected until the runner reads and clears them.
        self.death_ages: List[int] = []
        self.c: Dict[str, int] = {}
        self.last_alloc: Dict[int, Dict[int, int]] = {}
        # Who was allied with whom in the last game (both ways).
        self.last_allies: Dict[int, set] = {}
        self._seekers: List[Tuple[int, int]] = []
        super().__init__(G_init, cfg, _empty=_empty)
        for b in self.brains.values():
            self.root_of.setdefault(b.brain_id, b.brain_id)

    # ---- bookkeeping -------------------------------------------------------
    def _count(self, key, n=1):
        self.c[key] = self.c.get(key, 0) + n

    def _new_brain(self):
        b = super()._new_brain()
        self.root_of[b.brain_id] = b.brain_id
        return b

    def _mutate_brain(self, brain):
        old = brain.brain_id
        if brain.mutate(self.next_brain_id, self.rng):
            self.next_brain_id += 1
            self.root_of[brain.brain_id] = self.root_of.get(old, old)
            self.brain_parent[brain.brain_id] = old
            self.brain_depth[brain.brain_id] = self.brain_depth.get(old, 0) + 1
            self._count("mutations")

    def _forget_agent(self, aid):
        born = self.born_at.get(aid)
        if born is not None and born >= 0:
            self.death_ages.append(self.iteration - born)
        super()._forget_agent(aid)

    def kin_distance(self, a, b, limit):
        """Mutations between two genotypes through their last common ancestor, or None past `limit`."""
        if a == b:
            return 0
        da, db = self.brain_depth.get(a, 0), self.brain_depth.get(b, 0)
        steps = 0
        while a != b:
            if da >= db:
                a = self.brain_parent.get(a)
                da -= 1
            else:
                b = self.brain_parent.get(b)
                db -= 1
            steps += 1
            if a is None or b is None or steps > limit:
                return None
        return steps

    def _window_phases(self):
        w = int(P.get("window", 0))
        return w if w > 0 else super()._window_phases()

    def _stale_edges(self):
        stale = super()._stale_edges()
        if P.get("ally_keep") and stale:
            allied = self.last_allies
            kept = [(a, b) for a, b in stale if b in allied.get(a, ())]
            if kept:
                self._count("ally_kept", len(kept))
                stale = [(a, b) for a, b in stale if b not in allied.get(a, ())]
        if not P.get("safe_prune") or not stale:
            return stale
        gone = set()
        out = []
        G = self.G
        for a, b in sorted(stale):
            na, nb = G[a], G[b]
            small, large = (na, nb) if len(na) <= len(nb) else (nb, na)
            need = 2 if P.get("safe_prune") == 2 else 1
            found = 0
            for c in small:
                if c in large and c != a and c != b:
                    if self._edge_key(a, c) not in gone and self._edge_key(b, c) not in gone:
                        found += 1
                        if found >= need:
                            break
            if found >= need:
                gone.add(self._edge_key(a, b))
                out.append((a, b))
        self._count("prune_spared", len(stale) - len(out))
        return out

    def _repair_around(self, nbrs):
        """Join the groups of a dead agent's neighbours, least-connected members first."""
        G = self.G
        nbrs = [x for x in nbrs if G.has_node(x)]
        if len(nbrs) < 2:
            return
        groups: List[List[int]] = []
        where: Dict[int, int] = {}
        for x in sorted(nbrs):
            joined = [where[y] for y in G[x] if y in where]
            if not joined:
                where[x] = len(groups)
                groups.append([x])
                continue
            target = min(joined)
            groups[target].append(x)
            where[x] = target
            for g in set(joined) - {target}:
                for y in groups[g]:
                    where[y] = target
                groups[target].extend(groups[g])
                groups[g] = []
        groups = [g for g in groups if g]
        pairs = list(zip(groups, groups[1:]))
        if len(groups) >= 3:
            pairs.append((groups[-1], groups[0]))
        for g1, g2 in pairs:
            a = min(g1, key=lambda z: (G.degree(z), z))
            b = min(g2, key=lambda z: (G.degree(z), z))
            self._add_edge(a, b)
            self._count("repairs")

    def _cleanup_and_redistribute(self):
        if not P.get("repair"):
            return super()._cleanup_and_redistribute()
        # The engine's cleanup with one change: each starved agent is removed on its
        # own, and its neighbourhood is mended before the next one goes.
        report = {"resurrected": False, "starved": 0, "orphaned": 0, "redistributed": 0}
        starved = sorted(u for u in self.G.nodes() if self.tokens.get(u, 0) <= 0)
        limit = int(P.get("max_age", 0))
        if limit:
            old = sorted(u for u in self.G.nodes()
                         if self.tokens.get(u, 0) > 0 and self.iteration - self.born_at.get(u, self.iteration) >= limit)
            self._count("aged", len(old))
            starved = sorted(set(starved) | set(old))
        pool = self.cfg.tokens_created_per_phase
        for u in starved:
            nbrs = list(self.G.neighbors(u))
            pool += max(0, self.tokens.get(u, 0))
            self._drop_edges([(u, v) for v in nbrs])
            self.G.remove_node(u)
            self._forget_agent(u)
            self._repair_around(nbrs)
        orphaned = set()
        if self.G.number_of_nodes() > 0:
            comps = sorted(nx.connected_components(self.G), key=len, reverse=True)
            for c in comps[1:]:
                orphaned.update(c)
        for u in orphaned:
            pool += max(0, self.tokens.get(u, 0))
        if orphaned:
            self._drop_edges([(u, v) for u in orphaned for v in self.G.neighbors(u)])
            self.G.remove_nodes_from(list(orphaned))
            for u in orphaned:
                self._forget_agent(u)
        report["starved"] = len(starved)
        report["orphaned"] = len(orphaned)
        survivors = list(self.G.nodes())
        if pool > 0 and survivors:
            draws = self.rng.multinomial(pool, np.full(len(survivors), 1 / len(survivors)))
            for u, extra in zip(survivors, draws):
                self.tokens[u] = self.tokens.get(u, 0) + int(extra)
            report["redistributed"] = int(pool)
        if self.G.number_of_nodes() == 0:
            aid = self.next_agent_id
            self.next_agent_id += 1
            self.G.add_node(aid)
            for book in self._books:
                book.clear()
            self.edge_active_at.clear()
            self._register_agent(aid, self.cfg.total_tokens, self._new_brain(), -1)
            report["resurrected"] = True
        return report

    def root(self, aid):
        b = self.brains[aid]
        return self.root_of.get(b.brain_id, b.brain_id)

    # ---- what a brain sees -------------------------------------------------
    def _inputs(self, u, candidates, log_deg, q_tok, q_deg, log_tok, at_risk):
        X = super()._inputs(u, candidates, log_deg, q_tok, q_deg, log_tok, at_risk)
        if P.get("fog"):
            lay = self.cfg.input_layout()
            for name in ("its_tokens", "its_degree", "its_token_quantiles", "its_degree_quantiles"):
                a, b, _ = lay[name]
                X[a:b, 1:] = 0.0
        if P.get("memory"):
            mine = self.last_alloc.get(u, {})
            extra = np.array([[np.log1p(mine.get(v, 0)) for v in candidates],
                              [np.log1p(self.last_alloc.get(v, {}).get(u, 0)) for v in candidates]])
            X = np.vstack([X, extra])
        L = int(P.get("memory_len", 0))
        if L:
            hist = getattr(self, "hist", [])
            seen = getattr(self.brains[u], "mem", L)
            rows = np.zeros((2 * L, len(candidates)))
            for k in range(min(L, len(hist), seen)):
                sent = hist[k]
                mine = sent.get(u, {})
                rows[2 * k] = [np.log1p(mine.get(v, 0)) for v in candidates]
                rows[2 * k + 1] = [np.log1p(sent.get(v, {}).get(u, 0)) for v in candidates]
            X = np.vstack([X, rows])
        if pooled():
            X = np.vstack([X, [[1.0 if v != u and self.same_side(u, v) else 0.0
                                for v in candidates]]])
        elif P.get("kin_sense"):
            own = self.brains[u].brain_id
            X = np.vstack([X, [[1.0 if v != u and self.brains[v].brain_id == own else 0.0
                                for v in candidates]]])
        return X

    def same_side(self, a, b, allies=None) -> bool:
        """Are agents a and b on one side: allied (in `allies`, else the last game), or kin?"""
        if P.get("allies"):
            book = self.last_allies if allies is None else allies
            return b in book.get(a, ())
        if P.get("pool") == "genotype":
            return self.brains[a].brain_id == self.brains[b].brain_id
        k = int(P.get("kin_pool", -1))
        return self.kin_distance(self.brains[a].brain_id, self.brains[b].brain_id, k) is not None

    def _resolve_sides(self, v, offers, allies):
        """
        Who takes node v when stakes are pooled by side.

        `offers` maps each bidder to its stake (the owner's already weighted). The
        owner's side is the owner and every bidder on its side; the other bidders
        are grouped into sides by being on one side with each other. Returns the
        new owner and the winning side's members.
        """
        same = lambda a, b: self.same_side(a, b, allies)
        owner_side = [v] + [b for b in offers if b != v and same(b, v)]
        taken = set(owner_side)
        rest = [b for b in offers if b not in taken]
        sides = [owner_side]
        while rest:
            group, frontier = [rest.pop(0)], None
            frontier = list(group)
            while frontier:
                x = frontier.pop()
                joined = [y for y in rest if same(x, y)]
                for y in joined:
                    rest.remove(y)
                group += joined
                frontier += joined
            sides.append(group)
        totals = [sum(offers.get(b, 0) for b in s) for s in sides]
        best = max(totals)
        tied = [i for i, t in enumerate(totals) if t == best]
        i = tied[0] if len(tied) == 1 else int(self.rng.choice(tied))
        if i == 0:
            return v, owner_side
        side = sides[i]
        top = max(offers[b] for b in side)
        leaders = [b for b in side if offers[b] == top]
        return (leaders[0] if len(leaders) == 1 else int(self.rng.choice(leaders))), side

    # ---- reproduction, with the search for partners after it ----------------
    def _spawn_child(self, parent, parent_tokens, child_tokens, candidates, Y):
        child_id, linked = self._spawn_child_rules(parent, parent_tokens, child_tokens, candidates, Y)
        if P.get("birth_allies"):
            G = self.G
            for v in [parent] + sorted(self.last_allies.get(parent, ())):
                if G.has_node(v) and v != child_id and not G.has_edge(child_id, v):
                    self._add_edge(child_id, v)
                    linked.append(v)
                    self._count("birth_ally_links")
        return child_id, linked

    def _spawn_child_rules(self, parent, parent_tokens, child_tokens, candidates, Y):
        K = int(P.get("valence", 0))
        if K:
            return self._spawn_child_valence(parent, parent_tokens, child_tokens, candidates, Y, K)
        child_id, linked = super()._spawn_child(parent, parent_tokens, child_tokens, candidates, Y)
        k = int(P.get("birth_links", 0)) or (1 if P.get("triangle_birth") else 0)
        if k:
            if not self.G.has_edge(child_id, parent):
                self._add_edge(child_id, parent)
                linked.append(parent)
            others = [c for c in range(1, len(candidates))
                      if self.G.has_node(candidates[c]) and candidates[c] != child_id]
            have = sum(1 for c in others if self.G.has_edge(child_id, candidates[c]))
            if have < k:
                L = Y[self.heads["LINK"], :]
                for c in sorted(others, key=lambda c: -(L[0, c] - L[1, c])):
                    if have >= k:
                        break
                    if not self.G.has_edge(child_id, candidates[c]):
                        self._add_edge(child_id, candidates[c])
                        linked.append(candidates[c])
                        have += 1
        return child_id, linked

    def _spawn_child_valence(self, parent, parent_tokens, child_tokens, candidates, Y, K):
        """The engine's birth, with a cap on links: the parent is always joined; then
        `birth_links` random neighbours of the parent that have room; then the LINK choices."""
        from GraphOfLifeSimple import _choose_binary
        child_id = self.next_agent_id
        self.next_agent_id += 1
        self.G.add_node(child_id)
        child_brain = self._copy_brain(self.brains[parent])
        self._mutate_brain(child_brain)
        self._register_agent(child_id, child_tokens, child_brain, int(parent))
        self.tokens[parent] = parent_tokens - child_tokens
        G = self.G
        linked = []
        self._add_edge(child_id, parent)
        linked.append(parent)
        if P.get("valence_mode") == "optional":
            # The physics links ignore the cap: a child always closes its triangles.
            room = [c for c in range(1, len(candidates)) if G.has_node(candidates[c])]
        else:
            room = [c for c in range(1, len(candidates))
                    if G.has_node(candidates[c]) and G.degree(candidates[c]) < K]
        k = int(P.get("birth_links", 0))
        if k and room:
            for c in self.rng.permutation(room)[:k]:
                v = candidates[int(c)]
                self._add_edge(child_id, v)
                linked.append(v)
        L, M = Y[self.heads["LINK"], :], Y[self.heads["LINK_MODE"], :]
        for c in range(1, len(candidates)):
            v = candidates[c]
            if G.degree(child_id) >= K:
                break
            if v in linked or not G.has_node(v) or G.degree(v) >= K:
                continue
            if _choose_binary(L[0, c], L[1, c], M[0, c], M[1, c], self.rng):
                self._add_edge(child_id, v)
                linked.append(v)
        return child_id, linked

    def _choose_gifts(self, u, purse, candidates, Y):
        if P.get("seek") and purse >= 1:
            frac = np.mean(Y[self.heads["SEEK"], :], axis=1)
            budget = int(np.floor(_share_of_first(frac[0], frac[1]) * purse))
            budget = min(budget, int(P.get("seek_max", 8)))
            if budget >= 1:
                self._seekers.append((int(u), budget))
        return super()._choose_gifts(u, purse, candidates, Y)

    def _engine_reproduction(self):
        """The engine's reproduction phase, copied, with the prototype hooks."""
        handovers, gifts = [], []
        outbox: Dict[int, Dict[int, List[float]]] = {}
        self.phase_count += 1
        nodes_before = self.G.number_of_nodes()
        tokens_before = dict(self.tokens)
        features = self._precompute_features()
        self._message_prepass("repro.messages", features)
        log_deg, neighs, q_tok, q_deg, log_tok, at_risk = features
        least = int(P.get("min_child", 1))
        for u in sorted(self.G.nodes()):
            tokens_u = int(self.tokens.get(u, 0))
            if tokens_u <= 0:
                continue
            candidates = [u] + list(neighs[u])
            Y = self._observe(u, candidates, log_deg, q_tok, q_deg, log_tok, at_risk)
            self._emit_messages(u, candidates, Y, outbox)
            frac = np.mean(Y[self.heads["REPRO_FRACTION"], :], axis=1)
            child_tokens = int(np.floor(_share_of_first(frac[0], frac[1]) * tokens_u))
            child_tokens = max(0, min(tokens_u, child_tokens))
            if P.get("keep_one") or P.get("keep_frac", 0) > 0:
                child_tokens = min(child_tokens, tokens_u - 1)
            K = int(P.get("valence", 0))
            if child_tokens >= max(1, least) and K and self.G.degree(u) >= K:
                child_tokens = 0
                self._count("births_crowded")
            if child_tokens >= max(1, least):
                child_id, links = self._spawn_child(u, tokens_u, child_tokens, candidates, Y)
                for v in self._choose_handovers(u, child_id, candidates, Y):
                    handovers.append((u, int(v), child_id))
            else:
                child_tokens = 0
            gifts.extend(self._choose_gifts(u, tokens_u - child_tokens, candidates, Y))
        for parent, v, child in handovers:
            if not (self.G.has_node(child) and self.G.has_edge(parent, v)):
                continue
            if child != v:
                self._add_edge(child, v)
            self._drop_edges([(parent, v)])
        self._drop_edges(list(nx.selfloop_edges(self.G)))
        for giver, taker, amount in gifts:
            if not self.G.has_edge(giver, taker):
                continue
            amount = min(amount, int(self.tokens.get(giver, 0)))
            if amount < 1:
                continue
            self.tokens[giver] = int(self.tokens.get(giver, 0)) - amount
            self.tokens[taker] = int(self.tokens.get(taker, 0)) + amount
            self._mark_flow(giver, taker)
        dead_edges = self._stale_edges() if self._prunes_after(1) else []
        self._drop_edges(dead_edges)
        self._deliver_messages(outbox)
        cleanup = self._cleanup_and_redistribute()
        return self._frame(phase=1, cleanup=cleanup, nodes_before=nodes_before,
                           tokens_before=tokens_before, decisions=None)

    def reproduction_phase(self, record_decisions):
        nxt = self.next_agent_id
        self._seekers = []
        frame = (GraphOfLife.reproduction_phase(self, record_decisions) if P.get("engine")
                 else self._engine_reproduction())
        self._count("births", self.next_agent_id - nxt)
        cl = frame["cleanup"]
        self._count("starved_r", cl["starved"])
        self._count("orphaned_r", cl["orphaned"])
        if P.get("seek") and self._seekers:
            self._search()
        return frame

    def _search(self):
        """Walk, pay, propose, and join where both ends agree."""
        pool = 0
        features = None
        for u, budget in self._seekers:
            if not self.G.has_node(u):
                continue
            budget = min(budget, int(self.tokens.get(u, 0)) - 1)   # never spend the last token
            if budget < 1:
                continue
            here = u
            steps = 0
            for _ in range(budget):
                nbrs = list(self.G.neighbors(here))
                if not nbrs:
                    break
                here = nbrs[self.rng.randint(len(nbrs))]
                steps += 1
                self.tokens[u] -= 1
                if P.get("toll") == "path":
                    self.tokens[here] = self.tokens.get(here, 0) + 1
                else:
                    pool += 1
            self._count("seek_tokens", steps)
            w = here
            if w == u or self.G.has_edge(u, w):
                self._count("seek_wasted")
                continue
            if features is None:
                features = self._precompute_features()
            log_deg, neighs, q_tok, q_deg, log_tok, at_risk = features
            if not (log_deg.get(u) is not None and log_deg.get(w) is not None):
                continue
            if self._accepts(u, w, features) and self._accepts(w, u, features):
                self._add_edge(u, w)
                self._count("links_made")
            else:
                self._count("links_refused")
        if pool > 0:
            nodes = list(self.G.nodes())
            draws = self.rng.multinomial(pool, [1 / len(nodes)] * len(nodes))
            for v, extra in zip(nodes, draws):
                self.tokens[v] += int(extra)

    def _mutual_yield(self, sent):
        """Tax every holding, pay the pool to connections staked on from both ends."""
        rate = float(P["yield_tax"])
        nodes = list(self.G.nodes())
        # The last token is never taxed: the tax must not be what starves an agent.
        held = np.array([max(0, self.tokens[u] - 1) for u in nodes], dtype=float)
        due = held * rate
        tax = np.floor(due).astype(int)
        tax += (self.rng.random_sample(len(nodes)) < (due - tax)).astype(int)
        pool = int(tax.sum())
        if pool <= 0:
            return
        pairs, weights = [], []
        for u, targets in sent.items():
            for v, s_uv in targets.items():
                if v <= u:
                    continue
                s_vu = sent.get(v, {}).get(u, 0)
                if s_vu > 0 and s_uv > 0 and self.G.has_edge(u, v):
                    pairs.append((u, v))
                    weights.append(min(s_uv, s_vu))
        if not pairs:
            return
        for u, t in zip(nodes, tax):
            self.tokens[u] -= int(t)
        shares = _apportion(np.array(weights, dtype=float), pool)
        for (u, v), amount in zip(pairs, shares):
            half = int(amount) // 2
            odd = int(amount) - 2 * half
            first, second = (u, v) if self.rng.random() < 0.5 else (v, u)
            self.tokens[first] += half + odd
            self.tokens[second] += half
        self._count("yield", pool)
        self._count("yield_pairs", len(pairs))

    def _accepts(self, u, w, features):
        log_deg, neighs, q_tok, q_deg, log_tok, at_risk = features
        Y = self._observe(u, [u, w], log_deg, q_tok, q_deg, log_tok, at_risk)
        acc = Y[self.heads["ACCEPT"], 1]
        mode = Y[self.heads["LINK_MODE"], 1]
        return _choose_binary(acc[0], acc[1], mode[0], mode[1], self.rng)

    # ---- the engine's own game, with the contest and pooling hooks ------------
    def _weighed(self, v, offers):
        """The engine's weighing; under pool=genotype, one offer per genotype."""
        self._contested = v
        weighed = super()._weighed(v, offers)
        if P.get("pool") != "genotype" or len(weighed) < 2:
            return weighed
        own = self.brains[v].brain_id
        sides: Dict[int, float] = {}
        rep: Dict[int, int] = {}
        for b in sorted(weighed):
            g = self.brains[b].brain_id
            sides[g] = sides.get(g, 0) + weighed[b]
            if g == own:
                rep[g] = v
            elif g not in rep or weighed[b] > weighed[rep[g]]:
                rep[g] = b
        if len(sides) < len(weighed):
            self._count("pooled_contests")
        return {rep[g]: total for g, total in sides.items()}

    def _endowment(self, parent, tokens, wanted):
        if not P.get("fission"):
            return super()._endowment(parent, tokens, wanted)
        half = tokens // 2
        if half < 1 or wanted < half:
            return 0
        if self.cfg.valence and self.G.degree(parent) >= self.cfg.valence:
            return 0
        return half

    def _resolve_winner(self, offers, revolutionaries, rng):
        if P.get("contest") == "lottery":
            keys = sorted(offers)
            w = np.array([float(offers[k]) for k in keys])
            pick = int(rng.choice(len(keys), p=w / w.sum())) if w.sum() > 0 else 0
            out = (int(keys[pick]), max(offers.values()), False)
        else:
            out = GraphOfLife._resolve_winner(offers, revolutionaries, rng)
        if P.get("engine"):
            v = self._contested
            if out[0] != v:
                self._count("conquests")
                if out[2]:
                    self._count("revolutions")
                if self.root(out[0]) == self.root(v):
                    self._count("kin_conquests")
        return out

    def blotto_phase(self, record_decisions: bool) -> Dict[str, Any]:
        if not P.get("engine"):
            return self._proto_blotto(record_decisions)
        frame = GraphOfLife.blotto_phase(self, record_decisions)
        self._count("starved_g", frame["cleanup"]["starved"])
        self._count("orphaned_g", frame["cleanup"]["orphaned"])
        return frame

    # ---- the game, copied from the engine with the prototype hooks -------------
    def _proto_blotto(self, record_decisions: bool) -> Dict[str, Any]:
        nodes_before = self.G.number_of_nodes()
        tokens_before = dict(self.tokens)
        self.phase_count += 1
        features = self._precompute_features()
        self._message_prepass("game.messages", features)
        log_deg, neighs, q_tok, q_deg, log_tok, at_risk = features
        outbox: Dict[int, Dict[int, List[float]]] = {}

        allocations_to: Dict[int, Dict[int, int]] = {v: {} for v in self.G.nodes()}
        revolution_to: Dict[int, Dict[int, int]] = {v: {} for v in self.G.nodes()}
        incoming_totals: Dict[int, int] = {v: 0 for v in self.G.nodes()}
        edge_flow: Dict[Tuple[int, int], int] = {tuple(sorted(e)): 0 for e in self.G.edges()}
        sent: Dict[int, Dict[int, int]] = {}
        support_to: Dict[int, int] = {}
        supported: Dict[int, Dict[int, int]] = {}
        wishes: Dict[int, set] = {}

        for u in sorted(self.G.nodes()):
            targets = [u] + list(neighs[u])
            Y = self._observe(u, targets, log_deg, q_tok, q_deg, log_tok, at_risk)
            self._emit_messages(u, targets, Y, outbox)
            if P.get("allies"):
                A = Y[self.heads["ALLY"], :]
                wishes[u] = {v for idx, v in enumerate(targets) if idx and A[0, idx] > A[1, idx]}
            tokens_u = int(self.tokens.get(u, 0))
            if tokens_u <= 0:
                continue
            scores = np.asarray(Y[self.heads["BLOTTO"], :], dtype=float)
            mode = np.mean(Y[self.heads["BLOTTO_MODE"], :], axis=1)
            spread = bool(mode[0] > mode[1])
            if P.get("newborn_home") and self.born_at.get(u) == self.iteration:
                alloc = np.zeros(len(targets), dtype=int)
                alloc[0] = tokens_u
                self._count("newborn_home")
            elif P.get("keep_one") or P.get("keep_frac", 0) > 0:
                # A part is set aside on the agent's own node; the rest is staked as chosen.
                kept = max(1, int(np.ceil(P.get("keep_frac", 0) * tokens_u)))
                rest = tokens_u - kept
                if spread:
                    alloc = _apportion(scores, rest) if rest > 0 else np.zeros(len(targets), dtype=int)
                else:
                    alloc = np.zeros(len(targets), dtype=int)
                    alloc[int(np.argmax(scores))] = rest
                alloc[0] += kept
            elif spread:
                alloc = _apportion(scores, tokens_u)
            else:
                alloc = np.zeros(len(targets), dtype=int)
                alloc[int(np.argmax(scores))] = tokens_u
            revolts_allowed = self.cfg.allow_revolutions
            rev_logits = Y[self.heads["REV_FRACTION"], :] if revolts_allowed else None
            sup_logits = Y[self.heads["SUPPORT"], :] if P.get("support") else None
            mine = sent.setdefault(u, {})
            for idx, v in enumerate(targets):
                amount = int(alloc[idx])
                if amount <= 0:
                    continue
                mine[v] = mine.get(v, 0) + amount
                incoming_totals[v] += amount
                bid = amount
                if sup_logits is not None and u != v:
                    sup = int(np.floor(_share_of_first(sup_logits[0, idx], sup_logits[1, idx]) * amount))
                    if sup > 0:
                        support_to[v] = support_to.get(v, 0) + sup
                        supported.setdefault(u, {})[v] = sup
                        bid = amount - sup
                if bid > 0:
                    allocations_to[v][u] = allocations_to[v].get(u, 0) + bid
                if revolts_allowed and bid > 0:
                    rev_share = _share_of_first(rev_logits[0, idx], rev_logits[1, idx])
                    rev_amount = int(np.floor(rev_share * bid))
                    if rev_amount > 0:
                        revolution_to[v][u] = revolution_to[v].get(u, 0) + rev_amount
                if u != v:
                    edge = tuple(sorted((u, v)))
                    if edge in edge_flow:
                        edge_flow[edge] += amount
                        self._mark_flow(u, v)
        self.last_flow = edge_flow
        self.last_support = supported
        if supported:
            self._count("support_tokens", sum(sum(d.values()) for d in supported.values()))
            self._count("support_mutual", sum(1 for a, d in supported.items() for b in d
                                              if a < b and supported.get(b, {}).get(a)))
        # Who staked what on whom, by staker: last_alloc[u][v] is u's stake on v.
        self.last_alloc = sent
        L = int(P.get("memory_len", 0))
        if L:
            self.hist = ([sent] + list(getattr(self, "hist", [])))[:L]

        allies_now: Dict[int, set] = {}
        if P.get("allies"):
            for u, want in wishes.items():
                mutual = {v for v in want if u in wishes.get(v, ())}
                if mutual:
                    allies_now[u] = mutual
            self._count("allied_pairs", sum(len(s) for s in allies_now.values()) // 2)
            self._count("ally_wishes", sum(len(s) for s in wishes.values()))

        new_tokens = dict(self.tokens)
        new_brains = dict(self.brains)
        mor = P.get("mor", "off")
        gain = float(P.get("support_gain", 1.0))
        for v in list(self.G.nodes()):
            offers = allocations_to[v]
            if not offers and not support_to.get(v):
                new_tokens[v] = 0
                new_brains[v] = self._copy_brain(self.brains[v])
                continue
            h = P.get("incumbency", 1.0)
            if (h != 1.0 and v in offers) or support_to.get(v):
                offers = dict(offers)
                offers[v] = offers.get(v, 0) * h + gain * support_to.get(v, 0)
            if pooled():
                winner, side = self._resolve_sides(v, offers, allies_now)
                by_revolt = False
                if sum(1 for b in side if offers.get(b, 0) > 0) >= 2:
                    self._count("pooled_wins")
                if winner == v:
                    self._count("side_gifts", sum(allocations_to[v].get(b, 0) for b in side if b != v))
            else:
                winner, max_amount, by_revolt = self._resolve_winner(offers, revolution_to[v], self.rng)
            if winner != v and P.get("juvenile", 0) and self.iteration - self.born_at[v] < P["juvenile"]:
                winner, by_revolt = v, False
                self._count("protected")
            new_brains[v] = self._copy_brain(self.brains[winner])
            new_tokens[v] = int(incoming_totals[v])
            if winner != v:
                self._count("conquests")
                if self.root(winner) == self.root(v):
                    self._count("kin_conquests")
                if by_revolt:
                    self._count("revolutions")
                if mor == "all":
                    self._mutate_brain(new_brains[v])

        self.tokens = new_tokens
        self.brains = new_brains
        self.last_allies = allies_now
        if P.get("yield_tax", 0.0) > 0:
            self._mutual_yield(sent)

        dead_edges = self._stale_edges() if self._prunes_after(2) else []
        self._drop_edges(dead_edges)
        self._count("pruned", len(dead_edges))
        self._deliver_messages(outbox)
        cleanup = self._cleanup_and_redistribute()
        self._count("starved_g", cleanup["starved"])
        self._count("orphaned_g", cleanup["orphaned"])

        if mor == "off":
            for brain in self.brains.values():
                self._mutate_brain(brain)
        self._prune_stale_messages()
        return self._frame(phase=2, cleanup=cleanup, nodes_before=nodes_before,
                           tokens_before=tokens_before, decisions=None)


def make_config(**overrides) -> ProtoConfig:
    cfg = ProtoConfig.for_new_run(**overrides)
    cfg.validate()
    return cfg


def make_world(cfg: SimConfig, proto: Dict[str, Any]) -> ProtoWorld:
    set_proto(proto)
    G0 = nx.watts_strogatz_graph(n=cfg.resolved_n(), k=cfg.resolved_k(),
                                 p=cfg.rewire_p, seed=stream_seed(cfg))
    return ProtoWorld(G0, cfg)
