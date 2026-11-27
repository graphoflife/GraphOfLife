"""Who took the two worlds of equals that died in Experiment 63: the brain line of their rich agents, traced
through the recorded frames. Reads frames; plays nothing.   python3 research/efficient/E50/autopsy_lines.py

A line is a brain and every brain copied from it, mutated or not (frames record each brain's parent). The line
followed is that of the last common ancestor of the agents that are rich (more than five median holdings) at the
first iteration with five rich agents or more."""
import gzip, json, os, sys
from collections import defaultdict
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ.setdefault("GOL_RUNS_DIR", "/home/stefan/Documents/CodingProjects/GraphOfLife/GraphOfLifeRuns")
sys.path.insert(0, ROOT)
import gol_plan, gol_store

#: condition, seed, first and last iteration read, iterations shown, a recorded game inside the takeover (or None)
WORLDS = [("30 numbers", 6302, 29100, 29350,
           [29250, 29270, 29280, 29290, 29300, 29310, 29320, 29330, 29340, 29345, 29348, 29350], None),
          ("50 numbers", 6302, 23150, 23620,
           [23250, 23350, 23400, 23450, 23500, 23510, 23520, 23530, 23540, 23550, 23560, 23600, 23620], 23500)]


def frame(run, iteration, phase):
    with gzip.open(os.path.join(gol_store.frames_dir(run), f"frame_{2 * iteration + phase - 1:05d}.json.gz")) as f:
        out = json.load(f)
    assert out["iteration"] == iteration and out["phase"] == phase
    return out


def rich(tokens):
    return np.asarray(tokens) > 5 * np.median(tokens)


def lineage(run, first, last):
    """
    The line that took the world, from the frames of iterations `first` to `last`: the brain it comes
    from, when each brain was first seen, whether a brain belongs to it, and the world before and
    after every game.
    """
    parent, born, before, after = {}, {}, {}, {}
    for it in range(first, last + 1):
        for phase, keep in ((1, before), (2, after)):
            f = frame(run, it, phase)
            for brain, of in zip(f["brain_ids"], f["parent_brain_ids"]):
                if brain not in parent:
                    parent[brain], born[brain] = of, it
            keep[it] = {k: f[k] for k in ("ids", "tokens", "brain_ids", "edges", "ages")}

    def chain(brain):
        out = [brain]
        while parent.get(out[-1]) in parent:
            out.append(parent[out[-1]])
        return out

    start = next(it for it in range(first, last + 1) if rich(after[it]["tokens"]).sum() >= 5)
    chains = [chain(b) for b, r in zip(after[start]["brain_ids"], rich(after[start]["tokens"])) if r]
    shared = set(chains[0]).intersection(*map(set, chains[1:]))
    origin = next(b for b in chains[0] if b in shared)
    member = {}

    def inline(brain):
        if brain not in member:
            member[brain] = origin in chain(brain)
        return member[brain]

    return origin, born, inline, before, after, start


def line_at(run, first, last, at):
    """The brains of the line in the world the run has at the start of iteration `at`, as a sorted list."""
    origin, born, inline, before, after, start = lineage(run, first, last)
    return sorted({b for b in after[at - 1]["brain_ids"] if inline(b)})


def trace(run, first, last, shown, game):
    origin, born, inline, before, after, start = lineage(run, first, last)
    print(f"  first iteration with five rich agents: {start}; the brain they all come from was first seen at {born[origin]}")
    print("  iteration: agents | the line's agents, its share of all tokens, links each | the others' tokens and links each | "
          "rich agents in the line / all rich")
    taken = strays = births = 0
    most = (0, None)
    for it in range(born[origin], last + 1):
        f, g = after[it], before[it]
        tokens, line = np.array(f["tokens"]), np.array([inline(b) for b in f["brain_ids"]])
        links = defaultdict(set)
        for u, v, *_ in g["edges"]:
            links[u].add(v); links[v].add(u)
        was = {u for u, b in zip(g["ids"], g["brain_ids"]) if inline(b)}
        now = {u for u, l in zip(f["ids"], line) if l}
        new = now - was
        taken += len(new)
        strays += sum(1 for v in new if not links[v] & was)
        births += sum(1 for b, age in zip(g["brain_ids"], g["ages"]) if age == 0 and inline(b))
        held = dict(zip(g["ids"], g["tokens"]))
        for u in was:
            turned = len(links[u] & new)
            if turned > most[0]:
                most = (turned, (it, held[u], len(links[u]), len(was)))
        if it in shown:
            degree = defaultdict(int)
            for u, v, *_ in f["edges"]:
                degree[u] += 1; degree[v] += 1
            d = np.array([degree[u] for u in f["ids"]])
            mean = lambda x: f"{x.mean():.0f}" if len(x) else "-"
            r = rich(tokens)
            print(f"   {it}: {len(tokens):4d} | {line.sum():4d}, {tokens[line].sum() / tokens.sum():.2f}, {mean(d[line])} | "
                  f"{mean(tokens[~line])}, {mean(d[~line])} | {int((r & line).sum())}/{int(r.sum())}")
    print(f"  nodes that turned to the line in a game: {taken}; of them next to no agent of the line before that game: {strays}; "
          f"agents born into the line: {births}")
    print(f"  most nodes turned in one game next to one agent of the line: {most[0]} (iteration, its tokens, its links, "
          f"the line's agents then: {most[1]})")
    if game:
        f, g = frame(run, game, 2), before[game]
        line = {u for u, b in zip(g["ids"], g["brain_ids"]) if inline(b)}
        mine = [a for a in f["decisions"]["allocations"] if a["agent"] in line]
        rest = [a for a in f["decisions"]["allocations"] if a["agent"] not in line]
        won = sum(1 for w in f["decisions"]["winners"] if w["winner"] in line)
        print(f"  the game of iteration {game}, whose stakes are recorded: {len(mine)} agents of the line, of whom "
              f"{sum(not a['spread'] for a in mine)} stake everything on one node ({sum(a['alloc'][0] == a['tokens'] for a in mine)} "
              f"on their own, {sum(a['alloc'][0] == 0 for a in mine)} on a neighbour's); they hold "
              f"{sorted(a['tokens'] for a in mine)} tokens and win {won} nodes. Of the {len(rest)} others, "
              f"{sum(not a['spread'] for a in rest)} stake on one node; their largest stake is "
              f"{int(np.median([max(a['alloc']) for a in rest]))} token as a median.")


if __name__ == "__main__":
    runs = {(s.condition, s.lab["seed"]): s.run_id for s in gol_plan.experiment_runs("E63")}
    for condition, seed, first, last, shown, game in WORLDS:
        print(f"\n{condition}, seed {seed} ({runs[(condition, seed)]})")
        trace(runs[(condition, seed)], first, last, shown, game)
