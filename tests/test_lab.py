"""
The lab: plans read strictly, runs shared, workers on their frozen engine, and
a queue worked through to the end however often it is stopped or cut off.

    python3 tests/test_lab.py

Everything happens in a scratch runs folder with plans of its own, on worlds
small enough that a worker process spends most of its life importing numpy.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import gol_lab        # noqa: E402
import gol_record     # noqa: E402
import gol_store      # noqa: E402
from gol_config import SimConfig   # noqa: E402

with open(os.path.join(HERE, "book", "experiments", "B1.json")) as _f:
    B1 = json.load(_f)

#: The real baseline, shrunk to a world a test can run in a second, and never
#: declared extinct, so a world that dwindles still goes on as long as asked.
TINY = {**B1["settings"], "total_tokens": 400, "n_nodes": 30, "k_neighbors": 4,
        "hidden_layers": [6], "message_amount": 2, "random_input_amount": 2,
        "extinction_threshold": 0}


def plan(name, conditions=({"name": "straight"},), seeds="1..2", iterations=6, **runs):
    """A plan on the tiny baseline B9."""
    return {"id": name, "title": name, "thesis": {},
            "runs": {"baseline": "B9", "world": {"total_tokens": 400}, "seeds": seeds,
                     "iterations": iterations, "record": {"checkpoint_every": 2,
                                                          "heavy_every": 2},
                     "conditions": list(conditions), **runs}}


@contextlib.contextmanager
def lab(*plans, baseline=TINY):
    """A runs folder and a plans folder of their own, and an engine that counts as committed."""
    with tempfile.TemporaryDirectory() as tmp:
        runs, plans_dir = os.path.join(tmp, "runs"), os.path.join(tmp, "plans")
        os.makedirs(runs)
        os.makedirs(plans_dir)
        with open(os.path.join(plans_dir, "B9.json"), "w") as f:
            json.dump({"id": "B9", "settings": baseline}, f)
        for p in plans:
            with open(os.path.join(plans_dir, f"{p['id']}.json"), "w") as f:
                json.dump(p, f)
        saved = gol_store.BASE_DIR, gol_lab.PLANS, gol_lab._git
        gol_store.BASE_DIR, gol_lab.PLANS = runs, plans_dir
        # The tests run on whatever is in the working tree, committed or not.
        gol_lab._git = lambda *args: "" if args[0] == "status" else "test-commit"
        try:
            yield tmp
        finally:
            gol_store.BASE_DIR, gol_lab.PLANS, gol_lab._git = saved


def work(spec, until, *extra, engine=None):
    """Run one worker for one spec, as the lab would, and return its exit code."""
    engine = engine or gol_lab.snapshot()
    job = gol_lab.Job(spec, until, engine,
                      os.path.exists(os.path.join(gol_store.run_dir(spec.run_id), "meta.json")),
                      None, 0)
    proc = gol_lab._start(job)
    return proc.wait(timeout=120)


def refused(fn, *words):
    try:
        fn()
    except gol_lab.LabError as exc:
        for word in words:
            assert word in str(exc), (word, str(exc))
        return
    raise AssertionError(f"accepted what it should have refused: {words}")


# ---------------------------------------------------------------------------
# Plans
# ---------------------------------------------------------------------------

def test_a_plan_with_an_unknown_setting_is_refused():
    """
    SimConfig.from_dict drops keys it does not know, which is right for a stored
    run and wrong for a plan: a misspelt setting would run the baseline under
    the name of an experiment. The lab refuses it, and anything else a plan
    cannot say.
    """
    with lab(plan("E91", [{"name": "a", "set": {"mutation_rate": 0.1}}]),
             plan("E92", [{"name": "a", "sets": {"mutation_probability": 0.1}}]),
             plan("E93", seed_list=[1])):
        refused(lambda: gol_lab.experiment_runs("E91"), "mutation_rate")
        refused(lambda: gol_lab.experiment_runs("E92"), "sets")
        refused(lambda: gol_lab.experiment_runs("E93"), "seed_list")


def test_a_condition_may_not_set_seed_or_infrastructure():
    """The seed belongs to the seeds and what is recorded to the lab, not to a condition."""
    with lab(plan("E91", [{"name": "a", "set": {"seed": 3}}]),
             plan("E92", [{"name": "a", "set": {"checkpoint_every": 5}}]),
             plan("E93", [{"name": "a", "set": {"total_tokens": 900}}])):
        refused(lambda: gol_lab.experiment_runs("E91"), "seed")
        refused(lambda: gol_lab.experiment_runs("E92"), "checkpoint_every")
        refused(lambda: gol_lab.experiment_runs("E93"), "total_tokens")


def test_a_baseline_names_every_mechanic_and_parameter():
    """
    A baseline that left a setting out would inherit whatever the code's
    default is on the day, and a default that moved would move the baseline
    with it. B1 names them all, and its strain is the one its settings give.
    """
    settings = gol_lab.baseline("B1")
    assert SimConfig.from_dict({**settings, "seed": 1}).strain_id() == B1["strain"]
    with lab(baseline={k: v for k, v in TINY.items() if k != "rewire_p"}):
        refused(lambda: gol_lab.baseline("B9"), "rewire_p")


def test_every_strain_an_experiment_uses_is_registered():
    """Every algorithm an experiment runs is named in research/strains.md, as the scheme asks."""
    registered = set(gol_lab.registered_strains())
    for name in gol_lab.experiments():
        for spec in gol_lab.experiment_runs(name):
            strain = SimConfig.from_dict(spec.config).strain_id()
            assert strain in registered, f"{name} runs {strain}, which strains.md does not list"


def test_a_plan_can_list_world_sizes_and_a_condition_its_own():
    """
    A plan may run its conditions at several sizes of world, and a condition
    at sizes of its own. Every run is still named for what it is, size
    included, and a size the engine cannot build is refused before anything
    runs, saying which.
    """
    sizes = plan("E91", [{"name": "a"},
                         {"name": "b", "set": {"mutation_probability": 0.5}, "sizes": [400]}],
                 seeds="1..2")
    sizes["runs"]["world"] = {"total_tokens": [300, 400]}
    broken = plan("E92")
    broken["runs"]["world"] = {"total_tokens": [400, 20]}
    odd = plan("E93", [{"name": "a", "sizes": [400.5]}])
    with lab(sizes, broken, odd):
        specs = gol_lab.experiment_runs("E91")
        ids = [s.run_id for s in specs]
        assert ids[:4] == ["B9-300-s001", "B9-300-s002", "B9-400-s001", "B9-400-s002"], ids
        assert len(ids) == 6 and all(i.startswith("B9-400-") for i in ids[4:]), ids
        assert specs[0].config["total_tokens"] == 300 and "300 tokens" in specs[0].name
        assert specs[0].lab["world"] == {"total_tokens": 300}
        refused(lambda: gol_lab.experiment_runs("E92"), "20 tokens")
        refused(lambda: gol_lab.experiment_runs("E93"), "whole numbers")


def test_a_plan_can_keep_its_runs_to_one_worker():
    """
    A plan may say how many of its runs run at once, whatever the lab's
    workers: its runs then go one after another while another experiment's
    use the rest. The estimate is made for the workers it will really have.
    """
    alone = plan("E91", seeds="1..3", workers=1)
    beside = plan("E92", [{"name": "other", "set": {"mutation_probability": 0.5}}], seeds="1..2")
    with lab(alone, beside, plan("E93", workers=0)):
        refused(lambda: gol_lab.experiment_runs("E93"), "workers")
        assert gol_lab.experiment_status("E91", gol_lab.read_control() | {"workers": 3},
                                         gol_lab.costs(), False)["workers"] == 1
        gol_lab.request(run="E91", workers=3)
        gol_lab.request(run="E92")
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            assert gol_lab.run_lab() == 0

        # The lab's own record of what it started and saw end, in order.
        mine = {f"B9-400-s{seed:03d}" for seed in (1, 2, 3)}
        going, most, others = set(), 0, 0
        for line in log.getvalue().splitlines():
            words = line.split()
            if "started" in words:
                run_id = words[words.index("started") + 1]
                going.add(run_id)
                others += run_id not in mine
            elif "ended" in words:
                going.discard(words[words.index("ended") - 1])
            most = max(most, len(going & mine))
        assert most == 1, f"{most} of a one-worker plan's runs ran at once"
        assert others == 2, "the other experiment's runs did not get the lab"
        assert gol_lab.status()["experiments"]["E92"]["state"] == "finished"

    # On one worker the order cannot change the time, so the quickest go first;
    # the experiment still gets the first place in the queue its longest run had.
    sweep = plan("E91", seeds="1", workers=1)
    sweep["runs"]["world"] = {"total_tokens": [100, 400, 200]}
    with lab(sweep, plan("E92", [{"name": "big"}], seeds="7")):
        order = [j.spec.run_id for j in gol_lab.jobs(["E91", "E92"], "e")]
        assert order[0] == "B9-100-s001", order
        assert [i for i in order if i.endswith("s001")] == ["B9-100-s001", "B9-200-s001",
                                                            "B9-400-s001"], order


def test_how_a_world_grows_with_its_tokens_is_fitted():
    """
    A power law is a straight line on logarithmic axes: its slope is found,
    with an interval that holds the true one, and points that bend away from
    a line are told from points on one. A size sweep in the lab reads every
    run's mean over the plan's window against its world's tokens.
    """
    import numpy as np
    import gol_analysis
    rng = np.random.default_rng(1)
    x = np.repeat([1e3, 2e3, 4e3, 8e3, 16e3, 32e3], 3)
    straight = 2 * x ** 0.8 * np.exp(rng.normal(0, 0.05, x.size))
    fit = gol_analysis.power_fit(x, straight, x)
    assert abs(fit["exponent"] - 0.8) < 0.05 and fit["interval"][0] < 0.8 < fit["interval"][1], fit
    assert fit["bendInterval"][0] < 0 < fit["bendInterval"][1], "a straight line was called bent"
    bent = x * np.exp(-0.2 * (np.log10(x) - 3.5) ** 2) * np.exp(rng.normal(0, 0.02, x.size))
    assert gol_analysis.power_fit(x, bent, x)["bendInterval"][1] < 0, "a bend went unseen"

    sweep = plan("E91", [{"name": "baseline"}], seeds="1..2", iterations=6)
    sweep["runs"]["world"] = {"total_tokens": [200, 400, 800]}
    sweep["analyse"] = {"kind": "scaling", "reference": "baseline",
                        "window": {"from": 3, "to": 5}, "check": {"from": 1, "to": 2},
                        "stats": [{"stat": "nodes", "name": "agents", "y": "agents"}]}
    with lab(sweep) as tmp, _book(tmp) as book:
        gol_lab.request(run="E91", workers=3)
        with contextlib.redirect_stdout(io.StringIO()):
            assert gol_lab.run_lab() == 0
        results = gol_analysis.analyse("E91")
        agents = results["scaling"]["agents"]["baseline"]
        assert sorted(agents["bySize"]) == ["200", "400", "800"], agents["bySize"]
        assert len(agents["points"]) == 6 and agents["fit"]["sizes"] == 3
        assert [step["from"] for step in agents["local"]] == [200, 400]
        rows = gol_record.read_stats("B9-800-s001")
        mean = np.mean([r["nodes"] for r in rows if r["phase"] == 2 and 3 <= r["iteration"] <= 5])
        point = next(p for p in agents["points"] if p["tokens"] == 800 and p["seed"] == 1)
        assert abs(point["value"] - mean) < 1e-9, (point, mean)
        figure = _svg(os.path.join(book, "figures", "E91", "agents.svg"))
        # 200 to 800 tokens on a log axis is ruled at 200 and 500; a linear one would have 300.
        assert "<circle" in figure and ">500<" in figure and ">300<" not in figure, \
            "the runs were not drawn on log axes"


def test_two_experiments_asking_for_the_same_run_share_it():
    """
    A run is named for what it is, so two experiments that need the same run
    get one, taken as far as the further of them asks — and a condition that
    is a run made again on purpose gets one of its own.
    """
    with lab(plan("E91", iterations=4),
             plan("E92", [{"name": "straight"}, {"name": "cut", "fault_at": 2},
                          {"name": "fast", "set": {"mutation_probability": 0.9}},
                          {"name": "apart", "replicate": True}],
                  iterations=6),
             {"id": "E93", "runs": "same as E91"}):
        runs = gol_lab.wanted(["E91", "E92", "E93"])
        fast = [gol_lab.run_id_for("B9", {"total_tokens": 400},
                                   {"mutation_probability": 0.9}, seed) for seed in (1, 2)]
        assert sorted(runs) == sorted(fast + ["B9-400-s001", "B9-400-s001-cut",
                                              "B9-400-s002", "B9-400-s002-cut",
                                              "B9-400-s001-apart", "B9-400-s002-apart"]), \
            sorted(runs)
        assert all(run_id.startswith("B9-400-") and run_id != "B9-400-s001" for run_id in fast)
        shared = runs["B9-400-s001"]
        assert shared.targets == [4, 6] and shared.experiments == ["E91", "E92", "E93"]
        assert runs["B9-400-s001-cut"].fault_at == 2


def test_a_plan_can_grow_but_not_change():
    """
    More seeds or longer runs only add to what exists. A baseline edited after
    its runs were made would put new settings on old runs, and those runs are
    stopped and named rather than carried on as something they are not.
    """
    with lab(plan("E91", seeds="1", iterations=2)) as tmp:
        spec = gol_lab.experiment_runs("E91")[0]
        assert work(spec, 2) == 0
        assert gol_lab.run_state(spec, None)["state"] == "done"

        grown = plan("E91", seeds="1..2", iterations=3)
        with open(os.path.join(tmp, "plans", "E91.json"), "w") as f:
            json.dump(grown, f)
        states = [gol_lab.run_state(s, None)["state"] for s in gol_lab.experiment_runs("E91")]
        assert states == ["waiting", "waiting"], states

        with open(os.path.join(tmp, "plans", "B9.json"), "w") as f:
            json.dump({"id": "B9", "settings": {**TINY, "mutation_probability": 0.3}}, f)
        state = gol_lab.run_state(gol_lab.experiment_runs("E91")[0], None)
        assert state["state"] == "blocked" and "other settings" in state["reason"], state


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------

def test_a_worker_imports_only_its_snapshot():
    """
    A worker that found its engine somewhere other than beside it would run
    today's code under a snapshot's name. It refuses — and one where it should
    be records the snapshot's hash and commit in the run's provenance.
    """
    with lab(plan("E91", seeds="1", iterations=1)) as tmp:
        spec = gol_lab.experiment_runs("E91")[0]
        assert work(spec, 1) == 0
        with open(os.path.join(gol_store.run_dir(spec.run_id), "provenance.json")) as f:
            session = json.load(f)["sessions"][0]
        assert session["engine"]["hash"] == gol_lab.engine_hash()
        assert session["engine"]["commit"] == "test-commit"
        assert session["exit"] == "target" and session["to"] == 1
        assert session["environment"]["threads"]["OPENBLAS_NUM_THREADS"] == "1", \
            "a worker has to keep the matrix library to one thread"

        alone = os.path.join(tmp, "alone")
        os.makedirs(alone)
        shutil.copy(os.path.join(HERE, "gol_worker.py"), alone)
        done = subprocess.run([sys.executable, "-B", os.path.join(alone, "gol_worker.py"),
                               spec.run_id, "--until", "2"],
                              env={**os.environ, "PYTHONPATH": HERE,
                                   "GOL_RUNS_DIR": gol_store.BASE_DIR},
                              capture_output=True, text=True, timeout=120)
        assert done.returncode == gol_lab.EXIT_REFUSED, done.stderr
        assert "refusing" in done.stderr


def test_a_changed_environment_is_refused():
    """
    A run started on one numpy does not go on under another: a last-bit
    difference in a sum is a different history. The worker refuses and says
    which library moved, and the lab treats the run as blocked.
    """
    with lab(plan("E91", seeds="1", iterations=4)):
        spec = gol_lab.experiment_runs("E91")[0]
        assert work(spec, 2) == 0
        path = os.path.join(gol_store.run_dir(spec.run_id), "provenance.json")
        with open(path) as f:
            provenance = json.load(f)
        provenance["sessions"][0]["environment"]["numpy"] = "0.0.1"
        with open(path, "w") as f:
            json.dump(provenance, f)

        assert work(spec, 4) == gol_lab.EXIT_REFUSED
        state = gol_lab.run_state(spec, None)
        assert state["state"] == "blocked" and "numpy 0.0.1" in state["reason"], state


def test_a_terminated_worker_resumes_to_the_same_run():
    """
    Pausing the lab sends its workers SIGTERM. A worker finishes the iteration
    in hand, saves, and says it was paused; the next one picks up from there,
    and the run is the run a worker left alone would have made.
    """
    with lab(plan("E91", [{"name": "straight"}, {"name": "paused", "stops": []}],
                  seeds="1", iterations=40)):
        straight, paused = gol_lab.experiment_runs("E91")
        assert work(straight, 40) == 0

        engine = gol_lab.snapshot()
        proc = gol_lab._start(gol_lab.Job(paused, 40, engine, False, None, 0))
        meta = os.path.join(gol_store.run_dir(paused.run_id), "meta.json")
        deadline = time.time() + 60
        while time.time() < deadline:
            if os.path.exists(meta) and gol_store.load_meta(paused.run_id).get("iteration", 0) >= 3:
                break
            time.sleep(0.05)
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(timeout=60) == 0
        stopped_at = gol_store.load_meta(paused.run_id)
        assert 3 <= stopped_at["iteration"] < 40, stopped_at["iteration"]
        assert stopped_at["checkpoint_iteration"] == stopped_at["iteration"]

        assert work(paused, 40) == 0
        with open(os.path.join(gol_store.run_dir(paused.run_id), "provenance.json")) as f:
            exits = [s["exit"] for s in json.load(f)["sessions"]]
        assert exits == ["paused", "target"], exits
        folder = gol_store.BASE_DIR
        same = gol_lab.compare((folder, straight.run_id), (folder, paused.run_id))
        assert same["same"], same


def test_an_engine_that_does_not_remake_a_run_is_caught():
    """
    Before runs made by different snapshots are compared with each other, the
    newer snapshot has to make the older one's runs again, exactly. An engine
    whose agents hear their noise a hundred times louder is caught.
    """
    with lab(plan("E91", seeds="1", iterations=3)):
        spec = gol_lab.experiment_runs("E91")[0]
        assert work(spec, 3) == 0
        engine = gol_lab.snapshot()
        assert gol_lab.reproduce(spec.run_id, engine, 3)["same"]

        other = gol_lab.engine_dir("0123456789abcdef")
        shutil.copytree(gol_lab.engine_dir(engine), other)
        source = os.path.join(other, "GraphOfLifeSimple.py")
        with open(source) as f:
            text = f.read()
        with open(source, "w") as f:
            f.write(text.replace("rng.uniform(-2.0, 2.0,", "rng.uniform(-200.0, 200.0,"))
        answer = gol_lab.reproduce(spec.run_id, "0123456789abcdef", 3)
        assert not answer["same"] and "firstFrame" in answer, answer


# ---------------------------------------------------------------------------
# The lab
# ---------------------------------------------------------------------------

def test_the_lab_works_through_a_queue():
    """
    Queued, run, and left alone: every run reaches what its plan asks, a run
    stopped on the way is continued in a new process, a run cut off without a
    checkpoint is resumed from the last one, and every one of them records
    exactly what the run left alone records.
    """
    conditions = [{"name": "straight"}, {"name": "stopped", "stops": [3]},
                  {"name": "cut", "fault_at": 5}, {"name": "threads", "threads": "default"}]
    with lab(plan("E91", conditions, seeds="1..2", iterations=8)):
        gol_lab.request(run="E91", workers=3)
        with contextlib.redirect_stdout(io.StringIO()):
            assert gol_lab.run_lab() == 0

        status = gol_lab.status()["experiments"]["E91"]
        assert status["state"] == "finished", status
        assert status["done"] == status["total"] == 8 * 8

        folder = gol_store.BASE_DIR
        for seed in (1, 2):
            reference = f"B9-400-s{seed:03d}"
            for variant in ("stopped", "cut", "threads"):
                same = gol_lab.compare((folder, reference), (folder, f"{reference}-{variant}"))
                assert same["same"], (variant, same)

            def exits(run_id):
                with open(os.path.join(gol_store.run_dir(run_id), "provenance.json")) as f:
                    return [(s["from"], s["exit"]) for s in json.load(f)["sessions"]]
            assert exits(f"{reference}-stopped") == [(0, "target"), (3, "target")]
            with open(os.path.join(gol_store.run_dir(f"{reference}-threads"),
                                   "provenance.json")) as f:
                threads = json.load(f)["sessions"][0]["environment"]["threads"]
            assert threads["OPENBLAS_NUM_THREADS"] is None, threads
            assert exits(f"{reference}-cut") == [(0, "fault"), (5, "target")]
            assert gol_store.load_meta(f"{reference}-cut")["checkpoint_iteration"] == 8


def test_what_a_simulation_costs_is_fitted_to_what_runs_recorded():
    """
    The estimates are fitted, not guessed: time is every recorded second over
    every agent-iteration it bought, so the biggest stretches of a run count
    for most, and each measure says whether it was measured. Agents per token
    waits until a world has settled, which a short run never does.
    """
    from GraphOfLifeSimple import brain_shape
    import gol_record
    with lab(plan("E91", seeds="1..2", iterations=5)):
        gol_lab.request(run="E91", workers=2)
        with contextlib.redirect_stdout(io.StringIO()):
            assert gol_lab.run_lab() == 0
        costs = gol_lab.fit_costs()
        seconds = work = 0.0
        for spec in gol_lab.experiment_runs("E91"):
            weights = brain_shape(SimConfig.from_dict(spec.config))["weights"]
            for row in gol_record.read_stats(spec.run_id):
                if row.get("_seconds"):
                    seconds += row["_seconds"]
                    work += row["nodes"] * weights / 1e4
        assert abs(costs["secondsPerAgentIteration"] - seconds / work) < 1e-12, costs
        assert costs["measured"]["secondsPerAgentIteration"] and costs["runs"] == 2
        assert not costs["measured"]["agentsPerToken"], "a five-iteration run taught the size of a world"
        assert costs["agentsPerToken"] == gol_lab.CALIBRATION["agentsPerToken"]
        assert gol_lab.costs() == costs, "the fit was not kept for the estimates"

        # A run of the kind measured is estimated from that kind's own speed,
        # not from all runs scaled by the size of their brains.
        spec = gol_lab.experiment_runs("E91")[0]
        kind = costs["kinds"][gol_lab.kind_of(spec.config)]
        nodes = sum(r["nodes"] for s in gol_lab.experiment_runs("E91")
                    for r in gol_record.read_stats(s.run_id) if r.get("_seconds"))
        assert abs(kind["secondsPerAgent"] - seconds / nodes) < 1e-12, kind
        guess = gol_lab.predict(spec.config, 10, costs)
        assert abs(guess["seconds"] - 10 * guess["agents"] * kind["secondsPerAgent"]) < 1e-9
        other = {**spec.config, "message_amount": 7}
        assert gol_lab.kind_of(other) != gol_lab.kind_of(spec.config)
        assert gol_lab.kind_of({**spec.config, "total_tokens": 999, "seed": 4}) \
            == gol_lab.kind_of(spec.config), "world size and seed made another kind"


def test_a_disk_that_would_fill_pauses_the_lab_and_says_why():
    """
    A run that would not fit on the disk is not started, to fail halfway. The
    lab pauses before it begins and the control file says what it needed.
    """
    with lab(plan("E91", seeds="1", iterations=4)):
        real = gol_lab.shutil.disk_usage
        gol_lab.shutil.disk_usage = lambda path: real(path)._replace(free=gol_lab.DISK_MARGIN)
        try:
            gol_lab.request(run="E91")
            with contextlib.redirect_stdout(io.StringIO()):
                assert gol_lab.run_lab() == 0
        finally:
            gol_lab.shutil.disk_usage = real
        control = gol_lab.read_control()
        assert control["paused"] and "disk" in control["reason"], control
        assert not os.path.exists(gol_store.run_dir("B9-400-s001"))


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def test_bands_count_who_is_left():
    """
    A band over runs that do not all last is a band over fewer runs at the end,
    and says so: the median of two is between them, of one is that one, and
    the count of runs behind each point falls when one stops.
    """
    import numpy as np
    import gol_analysis
    longer = (np.arange(10.0), np.arange(10.0))
    shorter = (np.arange(5.0), np.arange(5.0) + 100)
    band = gol_analysis.bands([longer, shorter], points=10)
    assert band["alive"] == [2] * 5 + [1] * 5, band["alive"]
    assert band["y"][0] == 50.0 and band["y"][9] == 9.0, band["y"]
    assert band["lo"][9] == band["hi"][9] == 9.0

    # One run measured every iteration and two every fifth: every stretch
    # holds all three, rather than most of them holding the first alone.
    often = (np.arange(50.0), np.arange(50.0))
    seldom = [(np.arange(0.0, 50.0, 5), np.arange(0.0, 50.0, 5)) for _ in range(2)]
    band = gol_analysis.bands([often] + seldom)
    assert band["alive"] == [3] * 10, band["alive"]


@contextlib.contextmanager
def _book(tmp):
    """Results and figures written somewhere other than the real book."""
    import gol_analysis
    saved = gol_analysis.BOOK
    gol_analysis.BOOK = os.path.join(tmp, "book")
    try:
        yield gol_analysis.BOOK
    finally:
        gol_analysis.BOOK = saved


def _strict(path):
    """Read JSON that a browser can read: no NaN, no infinity."""
    def refuse(token):
        raise AssertionError(f"{path} holds {token}")
    with open(path) as f:
        return json.load(f, parse_constant=refuse)


def _svg(path):
    """An SVG figure, which no undefined number may have reached."""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    assert text.startswith("<svg") and "nan" not in text.lower() and "inf" not in text.lower(), path
    return text


def test_figures_hold_no_nan():
    """
    The analysis of a sweep writes a figure per statistic it was asked for and
    a results file with each condition's ending compared with the reference,
    paired by seed, interval and all — and nothing in either is a NaN, which
    the page could not read. Graph statistics are absent on most iterations,
    so there are plenty to turn up.
    """
    import gol_analysis
    sweep = plan("E91", [{"name": "baseline"}, {"name": "fast",
                                                "set": {"mutation_probability": 0.9}}],
                 seeds="1..3", iterations=6)
    sweep["analyse"] = {"kind": "series", "reference": "baseline", "seedsNeeded": True,
                        "figures": [{"name": "nodes", "stat": "nodes"},
                                    {"name": "bridges", "stat": "bridges"}],
                        "endpoints": ["nodes", "edges"]}
    with lab(sweep) as tmp, _book(tmp) as book:
        gol_lab.request(run="E91", workers=3)
        with contextlib.redirect_stdout(io.StringIO()):
            assert gol_lab.run_lab() == 0
        results = gol_analysis.analyse("E91")
        assert results["figures"] == ["nodes", "bridges"]
        for name in ("nodes", "bridges"):
            figure = _svg(os.path.join(book, "figures", "E91", f"{name}.svg"))
            assert "of 3 runs" in figure, "the band was not drawn over all three runs"
        stored = _strict(os.path.join(book, "results", "E91.json"))
        fast = stored["comparisons"]["nodes"]["fast"]
        assert fast["paired"] and fast["n"] == 3 and fast["indicative"], fast
        assert fast["interval"][0] <= fast["difference"] <= fast["interval"][1], fast
        assert all(stored["endings"]["nodes"][c]["seedsNeeded"]["10%"] > 0 for c in ("baseline", "fast"))
        assert len(stored["citation"]["runs"]) == 6
        assert stored["citation"]["runs"][0]["environment"]["numpy"]
        assert stored == _strict(os.path.join(book, "results", "E91.json"))
        again = gol_analysis.analyse("E91")
        assert again["comparisons"] == results["comparisons"], "the same runs gave other numbers"

        # A statistic undefined in some runs — a spectral gap of a graph in
        # pieces, a ratio over nothing — reaches the writer as NaN or infinity.
        import numpy as np
        odd = os.path.join(book, "odd.json")
        gol_analysis._write(odd, {"gap": float("nan"), "ratio": [np.float64("inf"), 1.0]})
        assert _strict(odd) == {"gap": None, "ratio": [None, 1.0]}


def test_a_world_that_died_is_counted_apart():
    """
    A world that dies out did not end up anywhere: the last fifth of its life
    is its dying. Where runs ended up is measured over the runs that reached
    the end, over the stretch the plan names, and the dead are counted as an
    outcome of their own; so is any other stretch, over the runs that lived
    through it. Lineages read off the frames say how big the biggest
    genotype got, how long genotypes and agents last and how far back the
    living share an ancestor, and are compared between conditions.
    """
    import gol_analysis
    sweep = plan("E91", [{"name": "baseline"},
                         {"name": "doomed", "set": {"extinction_threshold": 10_000}}],
                 seeds="1..3", iterations=6)
    sweep["analyse"] = {"kind": "series", "reference": "baseline", "endpoints": ["nodes"],
                        "figures": [{"name": "nodes", "stat": "nodes", "seeds": [1],
                                     "seedsOf": "baseline"}],
                        "settledFrom": 2, "seedsNeeded": True, "lineage": True,
                        "windows": [{"name": "start", "from": 0, "to": 2},
                                    {"name": "lowest", "from": 1, "of": "min"}]}
    with lab(sweep) as tmp, _book(tmp) as book:
        gol_lab.request(run="E91", workers=3)
        with contextlib.redirect_stdout(io.StringIO()):
            assert gol_lab.run_lab() == 0
        results = gol_analysis.analyse("E91")
        assert results["reached"] == {"baseline": 3, "doomed": 0}, results["reached"]
        assert [x["seed"] for x in results["extinct"]["doomed"]] == [1, 2, 3]
        assert results["endings"]["nodes"]["doomed"]["n"] == 0, "a dead world's dying was averaged in"
        assert results["endingsFrom"] == 2
        drawn = _svg(os.path.join(book, "figures", "E91", "nodes.svg"))
        assert "baseline, seed 1" in drawn and "doomed, seed" not in drawn, drawn
        ended = results["endings"]["nodes"]["baseline"]
        assert ended["n"] == 3 and sorted(ended["bySeed"]) == [1, 2, 3], ended
        assert set(ended["seedsNeeded"]) == {"5%", "10%", "20%"}
        start = results["windows"]["start"]["nodes"]
        assert start["baseline"]["n"] == 3 and start["doomed"]["n"] == 0, start
        lowest = results["windows"]["lowest"]["nodes"]["baseline"]
        assert lowest["max"] <= ended["max"] + 1e-9, "the lowest of a stretch above its mean"

        lineage = results["lineage"]["baseline"]
        assert lineage["reached"] == 3
        for run in lineage["runs"].values():
            assert 0 < run["topShare"]["max"] <= 1 and run["end"] == 5
            assert run["genotypeLife"]["n"] > 0 and run["genotypeLife"]["median"] >= 1
            assert run["topShare"]["longestOverATenth"] <= 6
            assert set(run["ancestor"]) >= {"allFrom500", "moves", "oneFounder"}
        assert 0 <= lineage["largestFrom100"]["0.1"] <= 3
        assert results["lineageComparisons"]["ancestor.moves"]["doomed"]["n"] == 0


def test_the_common_ancestor_is_the_newest_that_holds_the_share():
    """
    Climbing the tree of genotypes from the living finds, for each share of
    the living agents, the newest genotype they all descend from — and none
    while they descend from more than one founder.
    """
    from gol_analysis import shared_ancestors
    #        1       2        founders
    #        |       |
    #        3       6
    #       / \
    #      4   5
    parent = {1: -1, 2: -1, 3: 1, 4: 3, 5: 3, 6: 2}
    assert shared_ancestors(parent, {4: 2, 5: 1}) == [3, 3, 4]
    assert shared_ancestors(parent, {4: 1, 5: 1}) == [3, 3, 5]
    assert shared_ancestors(parent, {4: 9, 6: 1}) == [None, 4, 4]
    assert shared_ancestors(parent, {4: 3, 6: 1}) == [None, None, 4]
    assert shared_ancestors(parent, {4: 1, 6: 1}) == [None, None, 6]
    assert shared_ancestors(parent, {3: 2}) == [3, 3, 3]


def test_an_identity_experiment_reports_every_variant():
    """
    The reproducibility experiment's analysis compares every variant with the
    reference run of its seed, and says so per run, with how much it compared.
    """
    import gol_analysis
    identity = plan("E91", [{"name": "straight"}, {"name": "stopped", "stops": [3]},
                            {"name": "cut", "fault_at": 4}], seeds="1..2", iterations=6)
    identity["analyse"] = {"kind": "identity", "reference": "straight"}
    with lab(identity) as tmp, _book(tmp):
        gol_lab.request(run="E91", workers=3)
        with contextlib.redirect_stdout(io.StringIO()):
            assert gol_lab.run_lab() == 0
        results = gol_analysis.analyse("E91")
        assert results["allSame"], results["comparisons"]
        assert len(results["comparisons"]) == 4
        assert all(c["frames"] == 12 and c["rows"] == 12 for c in results["comparisons"])


def _main() -> int:
    tests = sorted((name, fn) for name, fn in globals().items()
                   if name.startswith("test_") and callable(fn))
    failures = []
    started = time.perf_counter()
    for name, fn in tests:
        try:
            fn()
            print(".", end="", flush=True)
        except Exception:
            failures.append((name, traceback.format_exc()))
            print("F", end="", flush=True)
    print(f"\n\n{len(tests) - len(failures)} passed, {len(failures)} failed "
          f"in {time.perf_counter() - started:.1f}s")
    for name, trace in failures:
        print(f"\n--- {name} ---\n{trace}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_main())
