"""Experiment 71, read afterwards: how fast the brains of a world turn over under a head count.
Reads the runs' frames; plays nothing.   python3 research/efficient/E50/e71_lines.py

Every brain names the brain it was copied from. From a starting iteration, every brain alive then founds a line; a
later brain belongs to the line of the founder it descends from. The reading: how many iterations pass until one
line holds every agent, and how many lines are left on the way. Read from several starts in each world."""
import gzip, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e56
import gol_plan, gol_store

STARTS = range(22500, 30000, 1500)
MARKS = (25, 100, 250, 500)                                   # lines left after this many iterations


def lines(run_id, end=30000):
    """For each start: the lines alive at it, the lines left after each mark, the iterations to one line (None if
    more than one is left at the end), and that line's share of the agents at the start."""
    frames = os.path.join(gol_store.run_dir(run_id), "frames")
    open_, out, before, unknown = {}, {}, {}, 0               # open_[start] = {brain: founder} over the last two frames
    for n in range(2 * STARTS[0], 2 * end):
        with gzip.open(os.path.join(frames, f"frame_{n:05d}.json.gz")) as f:
            frame = json.load(f)
        it, brains, parents = n // 2, frame["brain_ids"], frame["parent_brain_ids"]
        if n % 2 == 0 and it in STARTS:
            open_[it] = [{}, {b: b for b in brains}]
            out[it] = {"lines": len(set(brains)), "left": {}, "to_one": None}
            before[it] = {b: brains.count(b) / len(brains) for b in set(brains)}
        for start, (older, last) in list(open_.items()):
            now = {}
            for b, pb in zip(brains, parents):
                now[b] = last.get(b) or older.get(b) or last.get(pb) or older.get(pb)
                unknown += now[b] is None
            open_[start] = [last, now]
            if n % 2 == 1:
                alive = set(now.values())
                if it + 1 - start in MARKS:
                    out[start]["left"][it + 1 - start] = len(alive)
                if len(alive) == 1:
                    founder = alive.pop()
                    out[start].update(to_one=it + 1 - start, founder_share=before[start].get(founder))
                    del open_[start]
    return out, unknown


if __name__ == "__main__":
    result = {}
    for spec in gol_plan.experiment_runs("E71"):
        label = spec.condition.rsplit(",", 1)[0]
        result[label], unknown = lines(spec.run_id)
        print(f"{label} (brains of unknown line: {unknown})", flush=True)
        for start, r in result[label].items():
            left = ", ".join(f"{k} after {m}" for m, k in r["left"].items())
            one = f"one line after {r['to_one']} iterations (it held {r['founder_share']:.1%} of the agents at the start)" if r["to_one"] else "more than one line at the end"
            print(f"  from {start:,}: {r['lines']} lines; {left}; {one}", flush=True)
    with open(os.path.join(e56.HERE, "e71_lines.json"), "w") as f:
        json.dump(result, f, indent=1)
