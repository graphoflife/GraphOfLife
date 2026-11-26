"""Experiment 67: the two worlds of equals that died, played again from the last world each kept. Is it
the same world? Reads the frames of the two plays; plays nothing.   python3 research/efficient/E50/e67.py"""
import gzip, json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ.setdefault("GOL_RUNS_DIR", "/home/stefan/Documents/CodingProjects/GraphOfLife/GraphOfLifeRuns")
sys.path.insert(0, ROOT)
import gol_plan, gol_store

HERE = os.path.dirname(os.path.abspath(__file__))
SAME = ("ids", "tokens", "brain_ids", "parent_brain_ids")


def after_game(run, iteration):
    """The frame after the game of `iteration`, or None if the run has none."""
    path = os.path.join(gol_store.frames_dir(run), f"frame_{2 * iteration + 1:05d}.json.gz")
    if not os.path.exists(path):
        return None
    with gzip.open(path) as f:
        frame = json.load(f)
    assert frame["iteration"] == iteration and frame["phase"] == 2
    return frame


if __name__ == "__main__":
    out = {}
    for spec in gol_plan.experiment_runs("E67"):
        first, start = spec.lab["from"]["run"], spec.lab["from"]["at"]
        ends = {name: gol_store.load_meta(run)["iteration"] for name, run in (("first", first), ("again", spec.run_id))}
        differs = games = None
        for games, it in enumerate(range(start, max(ends.values()) + 1)):
            a, b = after_game(first, it), after_game(spec.run_id, it)
            if a is None and b is None:
                break
            if a is None or b is None or any(a[k] != b[k] for k in SAME):
                differs = it
                break
        kept = gol_store.archived(spec.run_id)
        out[spec.condition] = {"first": first, "again": spec.run_id, "from": start, "ends": ends,
                               "games_compared": games, "first_difference": differs,
                               "worlds_kept": [len(gol_store.archived(first)), len(kept)]}
        print(f"{spec.condition}: {first} from {start:,}; the first play ends at {ends['first']:,}, the second at "
              f"{ends['again']:,}; {games:,} games compared; the plays first differ at {differs}; "
              f"worlds kept {len(gol_store.archived(first))} and {len(kept)} ({kept[0]:,} to {kept[-1]:,})")
    with open(os.path.join(HERE, "e67.json"), "w") as f:
        json.dump(out, f, indent=1)
