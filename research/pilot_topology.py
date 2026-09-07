#!/usr/bin/env python3
"""
Two theses about this graph, against runs that already exist.

Both come from `Graphs.md` §6 and neither needs a new run or a new engine
metric — everything used here is already in the cached series.

  T1  The shortcut budget is spent.
      No edge is ever created between nodes more than two hops apart, so the
      Watts-Strogatz long-range edges the world starts with can only be lost.
      Prediction: bridge share and clustering rise, loop density falls, and
      path length pulls away from the log(n) a small-world graph would hold.

  T2  Bridges lead the cull.
      Cleanup keeps only the largest component, so a bridge with a tenth of
      the population behind it is a tenth-of-the-population extinction waiting
      for the zero-flow prune to reach that edge. Prediction: bridge share at
      t-k predicts orphan share at t, above what the trend alone explains.

Nothing here concludes anything. It reports numbers and the controls they have
to beat, per `Research.md` §6: a measurement without a control has meant
nothing every previous time in this project.

    python3 research/pilot_topology.py [run_id ...]

With no arguments it reads every run under GraphOfLifeRuns/ that has a series.
"""

import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RUNS_DIR = os.environ.get(
    "GOL_RUNS_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "GraphOfLifeRuns"))

# How much of a run counts as "early" and "late" when quoting the two ends.
END_FRACTION = 0.10

# Lags tested for T2, in frames. Two frames is one iteration.
LAGS = range(0, 9)

# Shifted-series controls per lag. Enough to put a standard deviation on the
# null without the run taking longer than the thing it is checking.
SHUFFLES = 200


# ---------------------------------------------------------------------------
# Small statistics, kept here so the script runs with no dependencies
# ---------------------------------------------------------------------------

def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def _pearson(xs, ys):
    """Correlation, or nan if either side never moves."""
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = _mean(xs), _mean(ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return float("nan")
    return sxy / math.sqrt(sxx * syy)


def _ranks(xs):
    """Ranks with ties averaged, so Spearman is Pearson on these."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    out = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        shared = (i + j) / 2.0
        for k in range(i, j + 1):
            out[order[k]] = shared
        i = j + 1
    return out


def _spearman(xs, ys):
    return _pearson(_ranks(xs), _ranks(ys))


def _diff(xs):
    """First difference — what changed, rather than where it stands."""
    return [b - a for a, b in zip(xs, xs[1:])]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load(run_id):
    path = os.path.join(RUNS_DIR, run_id, "series.json")
    with open(path) as f:
        cache = json.load(f)
    rows = cache.get("rows") or []
    return cache, [r for r in rows if r.get("nodes")]


def column(rows, key, default=0.0):
    return [float(r.get(key) if r.get(key) is not None else default) for r in rows]


def derived(rows):
    """
    The size-free forms.

    Raw counts are useless here because the population grows by an order of
    magnitude inside a run — a rising bridge count could be nothing but a
    rising node count. Every quantity below is a share or a ratio.
    """
    out = []
    for r in rows:
        n = r.get("nodes") or 0
        m = r.get("edges") or 0
        before = r.get("nodes_before") or n
        if n < 2 or m < 1:
            continue
        out.append({
            "iteration": r.get("iteration", 0),
            "phase": r.get("phase", 0),
            "nodes": n,
            "bridgeShare": (r.get("bridges") or 0) / m,
            "loopDensity": r.get("loopDensity") or 0.0,
            "transitivity": r.get("transitivity") or 0.0,
            "meanDegree": r.get("meanDegree") or 0.0,
            "meanPathLength": r.get("meanPathLength") or 0.0,
            # A small-world graph holds mean path length near log(n)/log(k).
            # Dividing by log(n) removes the growth and leaves the shape.
            "pathOverLogN": ((r.get("meanPathLength") or 0.0) / math.log(n)) if n > 2 else 0.0,
            "diameterOverLogN": ((r.get("diameter") or 0) / math.log(n)) if n > 2 else 0.0,
            # Taken by the engine before the cull. The only version of this that
            # can be read as a cause of the cull rather than a consequence —
            # everything else in this row is post-cleanup. None on any run
            # recorded before the engine started writing it.
            "cutRiskBefore": r.get("cutRiskBefore"),
            "cutRisk": r.get("cutRisk"),
            "orphanShare": (r.get("orphaned") or 0) / before if before else 0.0,
            "starvedShare": (r.get("starved") or 0) / before if before else 0.0,
        })
    return out


# ---------------------------------------------------------------------------
# T1 — the shortcut budget
# ---------------------------------------------------------------------------

T1_KEYS = ("bridgeShare", "loopDensity", "transitivity", "meanDegree",
           "meanPathLength", "pathOverLogN", "diameterOverLogN")

# Which way the thesis says each one should move. The point of writing this
# down is that a metric moving the *wrong* way is evidence against, and it is
# too easy to read any movement as confirmation after the fact.
T1_EXPECTED = {
    "bridgeShare": "up",
    "loopDensity": "down",
    "transitivity": "up",
    "meanDegree": "down",
    "meanPathLength": "up",
    "pathOverLogN": "up",
    "diameterOverLogN": "up",
}


def thesis_one(rows):
    cut = max(1, int(len(rows) * END_FRACTION))
    early, late = rows[:cut], rows[-cut:]
    iters = [r["iteration"] for r in rows]

    print("\n  T1  the shortcut budget is spent")
    print(f"      first {cut} against last {cut} of {len(rows)} sampled frames")
    print(f"      {'metric':<20} {'early':>10} {'late':>10} {'change':>9}"
          f"  {'rho':>6}  expected")
    for key in T1_KEYS:
        a, b = _mean([r[key] for r in early]), _mean([r[key] for r in late])
        rho = _spearman(iters, [r[key] for r in rows])
        change = ((b - a) / a * 100) if a else float("nan")
        went = "up" if rho > 0 else "down"
        mark = "  ok" if went == T1_EXPECTED[key] else "  AGAINST"
        print(f"      {key:<20} {a:>10.4f} {b:>10.4f} {change:>8.1f}%"
              f"  {rho:>+6.2f}  {T1_EXPECTED[key]:<5}{mark}")


def watts_strogatz_control(rows):
    """
    What the same-sized graph would look like if it had stayed what it started as.

    The run begins as watts_strogatz_graph and the thesis is that it drifts off
    that shape. Drift is only visible against the shape, so this rebuilds one
    at the final size and the final mean degree and measures the same things.
    """
    try:
        import networkx as nx
    except ImportError:
        print("      (networkx not available — control skipped)")
        return

    last = rows[-1]
    n = int(last["nodes"])
    k = max(2, int(round(last["meanDegree"])))
    if k % 2:
        k += 1                      # watts_strogatz wants an even ring degree
    if k >= n:
        print("      (degree exceeds node count — control skipped)")
        return

    G = nx.watts_strogatz_graph(n=n, k=k, p=0.2, seed=1)
    m = G.number_of_edges()
    bridges = sum(1 for _ in nx.bridges(G))
    comps = nx.number_connected_components(G)
    # Path length on a big graph is sampled; the engine's own estimate is a
    # double sweep, so this is the comparable rough number rather than exact.
    nodes = list(G.nodes())
    random.seed(1)
    sample = random.sample(nodes, min(40, len(nodes)))
    total = pairs = 0
    for s in sample:
        for _, d in nx.single_source_shortest_path_length(G, s).items():
            total += d
            pairs += 1
    mean_path = total / pairs if pairs else 0.0

    print(f"\n      control: watts_strogatz(n={n}, k={k}, p=0.2), the shape it started as")
    print(f"      {'bridgeShare':<20} {bridges / m:>10.4f}   (run ends at {last['bridgeShare']:.4f})")
    print(f"      {'loopDensity':<20} {(m - n + comps) / m:>10.4f}"
          f"   (run ends at {last['loopDensity']:.4f})")
    print(f"      {'transitivity':<20} {nx.transitivity(G):>10.4f}"
          f"   (run ends at {last['transitivity']:.4f})")
    print(f"      {'pathOverLogN':<20} {mean_path / math.log(n):>10.4f}"
          f"   (run ends at {last['pathOverLogN']:.4f})")


# ---------------------------------------------------------------------------
# T2 — bridges lead the cull
# ---------------------------------------------------------------------------

def thesis_two(rows, stride):
    """
    Lag correlation, on levels and on differences, within one phase.

    Three things this has to get right, each of which got it wrong first:

    Phases are separated. Frames alternate reproduction and Blotto, and the two
    have different cleanup pressure, so a pooled series alternates in sign with
    lag and the alternation reads as structure. Correlating a phase against
    itself removes it.

    Lag is quoted in real frames, not in sampled rows. The series is stored at
    a stride, so one row back is `stride` frames back — labelling the axis by
    row index silently claims a resolution the file does not have.

    Levels are reported but should not be read. Both series trend across a run,
    so any two rising quantities correlate; the differenced version asks the
    question that matters, which is whether a *rise* in bridge share goes with
    a worse cull afterwards.

    The control circularly shifts the orphan series by a random offset. That
    keeps both marginal distributions and every autocorrelation intact and
    destroys only the alignment between them, which is the thing being claimed.
    """
    # Use the pre-cull reading where the run has one. Without it the leading
    # series is measured in the same frame as the cull it is supposed to
    # predict, so lag 0 cannot separate cause from effect at all and the whole
    # question is only half askable.
    pre = all(r["cutRiskBefore"] is not None for r in rows)
    lead_key = "cutRiskBefore" if pre else "bridgeShare"

    print("\n  T2  bridges lead the cull")
    print(f"      leading series: {lead_key}"
          + ("" if pre else "  — this run predates the pre-cull reading, so lag 0"
                            " is confounded and only positive lags mean anything"))
    for phase in sorted({r["phase"] for r in rows}):
        block = [r for r in rows if r["phase"] == phase]
        if len(block) < 40:
            continue
        # Rows of one phase are 2 frames apart before the stride is applied.
        step = stride if stride > 1 else 2
        bridge = [r[lead_key] for r in block]
        orphan = [r["orphanShare"] for r in block]
        d_bridge, d_orphan = _diff(bridge), _diff(orphan)

        name = {1: "reproduction", 2: "blotto"}.get(phase, f"phase {phase}")
        print(f"\n      phase {phase} ({name}), {len(block)} rows,"
              f" {step} frames apart")
        print(f"      {'lag':>10} {'levels':>9} {'differenced':>12}"
              f" {'null mean':>10} {'null sd':>8}  {'z':>6}")

        rng = random.Random(7)
        for lag in LAGS:
            if lag >= len(d_bridge) - 10:
                break
            lead_lv, resp_lv = bridge[:len(bridge) - lag], orphan[lag:]
            lead, resp = d_bridge[:len(d_bridge) - lag], d_orphan[lag:]
            real = _pearson(lead, resp)

            null = []
            span = len(resp)
            for _ in range(SHUFFLES):
                off = rng.randrange(span)
                null.append(_pearson(lead, [resp[(i + off) % span] for i in range(span)]))
            null = [c for c in null if c == c]
            mu = _mean(null)
            sd = math.sqrt(_mean([(c - mu) ** 2 for c in null])) if len(null) > 1 else 0.0
            z = (real - mu) / sd if sd > 0 else float("nan")

            print(f"      {lag * step:>7} fr {_pearson(lead_lv, resp_lv):>+9.3f}"
                  f" {real:>+12.3f} {mu:>+10.3f} {sd:>8.3f}  {z:>+6.1f}")

    print("\n      Read the differenced column against the null, not the levels.")
    print("      A column that alternates sign lag by lag is very likely an")
    print("      artefact rather than a signal: differencing an autocorrelated")
    print("      series induces negative correlation at lag 1, and that shows up")
    print("      here as +,-,+,- of roughly equal size. Pre-whitening both sides")
    print("      before correlating is the fix, and is not done yet — until it is,")
    print("      treat an alternating column as no result.")
    if not pre:
        print("      Lag 0 cannot settle the thesis either way here: `bridges` is")
        print("      counted after cleanup, in the same frame as `orphaned`, so a")
        print("      cull that removes a whole side of a bridge changes both at")
        print("      once. Re-run this against a simulation recorded since the")
        print("      engine started taking the reading before the cull.")


# ---------------------------------------------------------------------------

def main():
    wanted = sys.argv[1:]
    if not wanted:
        wanted = sorted(d for d in os.listdir(RUNS_DIR)
                        if os.path.isfile(os.path.join(RUNS_DIR, d, "series.json")))
    if not wanted:
        print("no runs with a cached series under GraphOfLifeRuns/")
        return

    print(__doc__.strip().split("\n\n")[0])

    for run_id in wanted:
        cache, raw = load(run_id)
        rows = derived(raw)
        if len(rows) < 40:
            print(f"\n{run_id}: only {len(rows)} usable frames, skipped")
            continue
        # A series built before the structural metrics existed still loads and
        # still has the keys, filled with zeros. Reporting nan over nan on it
        # looks like a failed measurement rather than an absent one.
        if not any(r["bridgeShare"] or r["transitivity"] for r in rows):
            print(f"\n{run_id}: series v{cache.get('version')} predates the "
                  f"structural metrics — nothing to read, rebuild it first")
            continue

        stride = cache.get("stride", 1)
        print(f"\n{'=' * 78}\n{run_id}"
              f"   series v{cache.get('version')}  stride {stride}"
              f"  {len(rows)} frames"
              f"  iterations {rows[0]['iteration']}..{rows[-1]['iteration']}"
              f"  nodes {rows[0]['nodes']}..{rows[-1]['nodes']}")
        if cache.get("version") != 18:
            print("      (built by an older formula set; rebuild to refresh)")

        thesis_one(rows)
        watts_strogatz_control(rows)
        thesis_two(rows, stride)

    print("\nNo conclusions here by design — see Graphs.md §6. The controls are\n"
          "the point: a trend that a shifted series matches is not a finding.")


if __name__ == "__main__":
    main()
