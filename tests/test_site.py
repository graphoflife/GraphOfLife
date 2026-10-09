"""
The two ways the site is served, and what each sends the page.

    python3 tests/test_site.py

gol_server.py serves the page from the repository, with the engine behind
it; build_site.sh assembles a static site whose browser runs the engine
itself, in a worker (web/py/gol_browser.py). Both have to ship the same
files, stamp what a returning visitor would otherwise keep, answer the page
the same way, and offer every setting the engine has.
"""

from __future__ import annotations

import math
import os
import re
import sys
import tempfile

import runner  # first: the repository on the path, and a runs folder of the tests' own

import gol_record
import gol_series
from gol_config import SimConfig
from GraphOfLifeSimple import new_world
from worlds import advanced_run, scratch_runs, small, unrecorded_run


# ---------------------------------------------------------------------------
# What the site ships
# ---------------------------------------------------------------------------

def test_the_server_and_the_build_ship_the_same_python():
    """
    The page fetches engine files from /py/. build_site.sh copies them there
    when it assembles the static site; gol_server.py serves them from the
    repository root, because when it is serving web/ off the disk there is
    nowhere to copy them to.

    Two lists of the same thing, so they drift. They already did: the
    Explanation fetches the script it walks through, which the build shipped
    and the server did not, so the tab worked once published and reported that
    it could not load on localhost.
    """
    import re
    import gol_server

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = open(os.path.join(root, "build_site.sh")).read()
    match = re.search(r"for f in ([^;]+); do", script)
    assert match, "could not find the copy loop in build_site.sh"
    copied = set(match.group(1).split())

    served = set(gol_server.SHIPPED_PY)
    assert copied == served, (
        f"build_site.sh copies {sorted(copied)} but gol_server serves "
        f"{sorted(served)}")

    for name in served:
        assert os.path.isfile(os.path.join(root, name)), f"{name} does not exist"

    # The same arrangement for documents the page renders, which live outside
    # web/ for the same reason and would fail the same way: rendering on the
    # published site and reporting that it cannot be read on localhost.
    #
    # Checked by name rather than by the whole destination path, because the
    # build copies them in a loop and the full string never appears. The
    # stamping table is checked separately, and is the half that matters — a
    # document that is copied and not stamped is a document a returning visitor
    # keeps the old version of.
    stamped = re.search(r"declare -A stamp_in=\((.*?)\n\)", script, re.S)
    assert stamped, "could not find the stamp_in table in build_site.sh"
    for url, source in gol_server.SHIPPED_DOCS.items():
        assert os.path.isfile(os.path.join(root, source)), f"{source} does not exist"
        assert os.path.basename(source) in script, (
            f"gol_server serves {url} from {source}, and build_site.sh never "
            f"copies it into the site")
        assert url in stamped.group(1), (
            f"build_site.sh copies {url} but never stamps it, so a cached copy "
            f"survives a deploy")


def test_the_browser_is_sent_every_module_its_python_imports():
    """
    The worker writes PY_FILES into the interpreter before it imports
    gol_browser, and the site has to hold every one of them. A module that
    gol_browser reaches and nobody sends fails only in a browser, as an import
    error on someone else's machine — so the list is checked against what the
    imports actually reach, and every file on it against what the build ships.
    """
    from closure import closure
    import gol_server

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    web_py = os.path.join(root, "web", "py")
    reached = closure(os.path.join(web_py, "gol_browser.py"), [web_py, root])
    needed = {os.path.basename(path) for path in reached.values()} | {"gol_browser.py"}

    worker = open(os.path.join(root, "web", "js", "sim-worker.js")).read()
    listed = re.search(r"const PY_FILES = \[(.*?)\];", worker, re.S)
    assert listed, "could not find PY_FILES in sim-worker.js"
    sent = set(re.findall(r"'([^']+\.py)'", listed.group(1)))
    assert needed <= sent, (
        f"gol_browser.py reaches {sorted(needed - sent)}, which the worker never sends")

    shipped = set(gol_server.SHIPPED_PY) | set(os.listdir(web_py))
    assert sent <= shipped, (
        f"the worker fetches {sorted(sent - shipped)}, which the site does not hold")


def test_every_asset_a_script_fetches_by_name_is_cache_stamped():
    """
    A returning visitor must never run new code against an old asset.

    index.html's script and stylesheet tags are stamped wholesale, but anything
    a script fetches by name — a worker, what a worker imports, the teaching
    script, a recording — is invisible to that pass and has to be named in
    build_site.sh. Nothing about adding a new one makes you remember, and the
    failure is silent and only hits people who have been here before: the page
    is new, the file behind it is last week's.

    So the two are checked against each other. Any asset reference in web/js
    that build_site.sh does not stamp fails here rather than in someone's
    cache.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = open(os.path.join(root, "build_site.sh")).read()

    # Only the stamping table counts, not the whole script. Looking for the
    # path anywhere in the file passed for a `cp` line that shipped a document
    # and never stamped it — mentioning an asset is not the same as versioning
    # it, and the whole failure this guards against is a file that is shipped
    # and cached.
    table = re.search(r"declare -A stamp_in=\((.*?)\n\)", script, re.S)
    assert table, "could not find the stamp_in table in build_site.sh"
    build = table.group(1)

    # A quoted path into one of the shipped directories, or a bare file next to
    # the script — which is what importScripts() takes.
    quoted = re.compile(r"""['"]((?:js|py|data|css|book)/[\w./-]+|[\w-]+\.js)['"]""")
    interesting = (".js", ".py", ".json", ".bin", ".css", ".md")

    def ours(line, at):
        """
        False for a name glued onto a base URL.

        `importScripts(PYODIDE + 'pyodide.js')` is fetched from a CDN that
        versions itself in its own path; there is nothing of ours in it to
        stamp.
        """
        return not line[:at].rstrip().endswith("+")

    unstamped = []
    js_dir = os.path.join(root, "web", "js")
    for name in sorted(os.listdir(js_dir)):
        if not name.endswith(".js"):
            continue
        source = open(os.path.join(js_dir, name)).read()
        for line in source.splitlines():
            # Only where a file is actually being fetched.
            if not re.search(r"importScripts\(|new Worker\(|fetch\(|SOURCE|SCRIPT|RUN:|INDEX:|workerUrl",
                             line):
                continue
            for match in quoted.finditer(line):
                ref = match.group(1)
                if not ref.endswith(interesting) or not ours(line, match.start()):
                    continue
                if ref not in build:
                    unstamped.append(f"{name}: {ref}")

    assert not unstamped, (
        "these are fetched by name but build_site.sh does not stamp them, so a "
        "cached copy will survive a deploy:\n  " + "\n  ".join(unstamped))


def test_the_server_and_the_build_ship_the_same_book():
    """The published book and the one served locally are the same kinds of file from the same folder."""
    import gol_server

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = open(os.path.join(root, "build_site.sh")).read()
    copy = re.search(r'cd "\$\{here\}/book" && find \. -type f (.+?)\)', script)
    assert copy, "could not find where build_site.sh copies the book"
    built = tuple(sorted(re.findall(r"-name '\*(\.\w+)'", copy.group(1))))
    assert built == tuple(sorted(gol_server.BOOK_TYPES)), (built, gol_server.BOOK_TYPES)


# ---------------------------------------------------------------------------
# The server
# ---------------------------------------------------------------------------

def test_a_run_is_reported_as_it_is_not_as_it_was_written():
    """
    Whether a run is going is a live fact; its status on disk is what was last
    written. The server settles the two, both ways, so the page shows the
    status it is given: written down as running with nothing advancing it is
    interrupted, and advancing is running whatever was last written.
    """
    import gol_server

    meta = {"id": "no-such-run", "status": "running", "config": {}}
    try:
        gol_server.POOL.is_running = lambda run_id: False
        assert gol_server.Handler._decorate(meta)["status"] == "interrupted"
        assert gol_server.Handler._decorate({**meta, "status": "stopped"})["status"] == "stopped"

        gol_server.POOL.is_running = lambda run_id: True
        assert gol_server.Handler._decorate({**meta, "status": "idle"})["status"] == "running"
    finally:
        del gol_server.POOL.is_running


def test_every_request_turns_what_goes_wrong_into_an_answer():
    """
    GET, POST and DELETE each turned exceptions into answers their own way,
    and no two agreed. One dispatcher does it now: a request that makes no
    sense is a 400, a missing run a 404, a browser that hung up is left alone,
    and anything else is a 500 with a message rather than a dropped connection.
    """
    import contextlib
    import io
    import gol_server

    class Request:
        path = "/api/runs/x?from=1"

        def __init__(self):
            self.answered = []

        def _error(self, message, status=400):
            self.answered.append((status, message))

    for raised, answer in ((ValueError("bad count"), (400, "bad count")),
                           (FileNotFoundError(), (404, "run not found")),
                           (KeyError("config"), (500, "KeyError: 'config'")),
                           (BrokenPipeError(), None)):
        request, seen = Request(), []

        def route(path, raised=raised):
            seen.append(path)
            raise raised

        with contextlib.redirect_stderr(io.StringIO()):
            gol_server.Handler._dispatch(request, route)
        assert seen == ["/api/runs/x"], seen
        assert request.answered == ([answer] if answer else []), (raised, request.answered)


def test_the_book_route_serves_only_the_book():
    """
    The book is served as a folder, since it grows with every chapter — and a
    folder served off the disk is a way to read anything beside it unless the
    path is held inside it after every link is followed, and only the kinds of
    file a book is made of are given out.
    """
    import gol_server

    found = gol_server.Handler._engine_file("book/experiments/E01.json")
    assert found and found.endswith(os.path.join("book", "experiments", "E01.json")), found
    for asked in ("book/../gol_server.py", "book/../research/Research.md",
                  "book/experiments/../../gol_lab.py", "book/missing.json",
                  "book/experiments/E01.json.py", "bookish/E01.json"):
        assert gol_server.Handler._engine_file(asked) is None, asked

    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, "book"))
        with open(os.path.join(tmp, "secret.json"), "w") as f:
            f.write("{}")
        os.symlink(os.path.join(tmp, "secret.json"), os.path.join(tmp, "book", "link.json"))
        with open(os.path.join(tmp, "book", "tool.py"), "w") as f:
            f.write("")
        saved = gol_server.BASE_DIR, gol_server.BOOK_DIR
        gol_server.BASE_DIR, gol_server.BOOK_DIR = tmp, os.path.join(tmp, "book")
        try:
            assert gol_server.Handler._engine_file("book/link.json") is None, \
                "a link inside the book reached a file outside it"
            assert gol_server.Handler._engine_file("book/tool.py") is None, \
                "the book gave out a file a book is not made of"
        finally:
            gol_server.BASE_DIR, gol_server.BOOK_DIR = saved


def test_a_lab_run_shows_running_and_refuses_start():
    """
    A run that belongs to an experiment is advanced by the lab, on the engine
    it was made with. Held by the lab it shows as running; and the server will
    not start it, stop it or delete it, which would run it on today's engine or
    pull it out from under the lab, and says where to go instead.
    """
    import gol_server
    import gol_store

    class Request:
        headers = {}
        _held_elsewhere = staticmethod(gol_server.Handler._held_elsewhere)

        def __init__(self, path):
            self.path = path
            self.answered = []

        def _read_json(self):
            return {}

        def _error(self, message, status=400):
            self.answered.append((status, message))

        def _send_json(self, payload, status=200):
            self.answered.append((status, payload))

    with scratch_runs():
        run_id = gol_store.create_run("lab run", small(seed=91), run_id="E9-s001",
                                      lab={"baseline": "B1", "engine": "x"})["id"]
        with gol_store.hold(run_id):
            assert gol_server.Handler._decorate(gol_store.load_meta(run_id))["running"]
            for action, route in (("start", gol_server.Handler._route_post),
                                  ("stop", gol_server.Handler._route_post),
                                  (None, gol_server.Handler._route_delete)):
                path = f"/api/runs/{run_id}" + (f"/{action}" if action else "")
                request = Request(path)
                route(request, path)
                status, message = request.answered[0]
                assert status == 409 and "Book" in message, (action, request.answered)
        assert os.path.isdir(gol_store.run_dir(run_id)), "a run held by the lab was deleted"

        request = Request(f"/api/runs/{run_id}/start")
        gol_server.Handler._route_post(request, f"/api/runs/{run_id}/start")
        assert request.answered[0][0] == 409, "an experiment's run was started outside the lab"


def test_the_server_keeps_the_matrix_library_to_one_thread():
    """
    A brain's matrix products are added up in another order when the library
    splits them across threads, and the last bit of a message moves: a run made
    on all threads is not the run the lab makes on one (Chapter 2). The server
    asks for one before numpy is loaded, unless the environment says otherwise.
    """
    import subprocess

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ask = ("import gol_server, os; "
           "print(os.environ['OPENBLAS_NUM_THREADS'], os.environ['OMP_NUM_THREADS'])")
    bare = {k: v for k, v in os.environ.items() if not k.endswith("_NUM_THREADS")}
    said = subprocess.run([sys.executable, "-B", "-c", ask], cwd=here, env=bare,
                          capture_output=True, text=True, check=True).stdout.split()
    assert said == ["1", "1"], said
    chosen = subprocess.run([sys.executable, "-B", "-c", ask], cwd=here,
                            env={**bare, "OPENBLAS_NUM_THREADS": "4"},
                            capture_output=True, text=True, check=True).stdout.split()
    assert chosen[0] == "4", "the server overrode a thread count it was given"


def test_the_defaults_endpoint_carries_the_brain_presets():
    """The form fills itself in from the engine, so the engine has to say."""
    import gol_server

    payload = gol_server.Handler._defaults()
    assert "brain_presets" in payload, "the form has nowhere to read the presets from"
    assert set(payload["brain_presets"]) == set(SimConfig.BRAIN_PRESETS)


def test_the_strip_is_answered_from_the_record_or_worked_out_alike():
    """
    The strip under the canvas asks the server for one frame's statistics. A
    frame the run recorded deep enough is answered from its stats.jsonl line,
    found by its offset; anything deeper is worked out from the frame by the
    same frame_stats, and both answers are the row the run would record.
    """
    import gol_framestats
    import gol_record
    import gol_server
    import gol_store

    with scratch_runs():
        run_id = advanced_run(small(seed=84), 6, {"heavy_every": 3})
        rows = gol_record.read_stats(run_id)
        light = next(r for r in rows if not r["_heavy"])
        heavy = next(r for r in rows if r["_heavy"])

        assert gol_server.recorded_row(run_id, light["_frame"]) == light
        assert gol_server.frame_row(run_id, light["_frame"]) == light, "a light question missed the record"
        assert gol_server.frame_row(run_id, heavy["_frame"], True, True) == heavy

        index = light["_frame"]
        frame = gol_store.read_frame(run_id, index)
        deep = gol_server.frame_row(run_id, index, True, False)
        assert deep == gol_framestats.strip(frame, None, True, False)
        assert deep == {k: v for k, v in gol_series.frame_stats(frame, None, True).items()}
        assert all(deep[k] is not None or light.get(k) is None for k in gol_series.HEAVY_KEYS
                   if k in ("bridges", "triangles", "cycleRank"))
        flow = gol_server.frame_row(run_id, index, False, True)
        assert flow["lightningScore"] == deep["lightningScore"] and flow["bridges"] is None, \
            "the flow on its own should come without the graph walk"

        # A line out of step with its frame is not handed out as that frame's.
        assert gol_server.recorded_row(run_id, 999) is None


# ---------------------------------------------------------------------------
# The browser's worker
# ---------------------------------------------------------------------------

def _browser_module():
    """gol_browser, which lives with the page rather than beside the engine."""
    import importlib.util

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location(
        "gol_browser", os.path.join(here, "web", "py", "gol_browser.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_two_browser_worlds_stepped_in_turn_record_what_each_records_alone():
    """The same for the page's worker, which holds every run in one interpreter."""
    config = {"total_tokens": 2000, "n_nodes": 40, "k_neighbors": 4,
              "hidden_layers": [6], "message_amount": 2, "random_input_amount": 2}

    def alone(seed):
        worlds = _browser_module().Worlds()
        worlds.create("x", {**config, "seed": seed})
        return [worlds.step("x")["frames"] for _ in range(4)]

    worlds = _browser_module().Worlds()
    worlds.create("a", {**config, "seed": 5})
    worlds.create("b", {**config, "seed": 6})
    together = {"a": [], "b": []}
    for _ in range(4):
        for run in ("a", "b"):
            together[run].append(worlds.step(run)["frames"])

    assert together["a"] == alone(5) and together["b"] == alone(6), \
        "two runs in one worker moved each other"


def test_a_browser_checkpoint_says_which_iteration_it_holds():
    """
    The worker recorded a checkpoint under the iteration in the run's stored
    record, which a slice in flight could have left behind the world. The
    checkpoint says for itself now, from the world it was written from, and a
    world restored from it is at that iteration.
    """
    import json

    browser = _browser_module().Worlds()
    config = {"total_tokens": 2000, "n_nodes": 40, "k_neighbors": 4,
              "hidden_layers": [6], "seed": 3}
    # The run keeps the configuration create settles on, as the worker does.
    stored = browser.create("a", config)["config"]
    browser.step("a", 3)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "a.npz")
        saved = browser.checkpoint("a", path)
        assert saved["iteration"] == 3 and saved["bytes"] == os.path.getsize(path), saved
        assert json.loads(_browser_module().to_json(saved)) == saved

        again = _browser_module().Worlds()
        restored = again.restore("a", stored, path)
        assert restored["iteration"] == saved["iteration"], restored


def test_the_worker_answers_in_json_with_what_is_not_a_number_as_null():
    """
    An answer crosses from Python to the page as JSON, which cannot write a
    NaN or an infinity, so those arrive as null and read as missing. Answers
    holding none are written straight out rather than rebuilt first, and must
    come out exactly as the rebuilt ones did.
    """
    import json

    browser = _browser_module()
    write = lambda value: json.dumps(value, separators=(",", ":"), allow_nan=False)

    clean = {"frames": [{"ids": [3, 4], "tokens": [0.5, 2.0]}], "pair": (1, 2.5), "extinct": False}
    assert browser.answer(lambda: clean, "[]") == write(browser._finite(clean))

    odd = {"ratio": float("nan"), "range": [1.0, float("inf"), -float("inf")],
           "pair": (float("nan"), 3)}
    assert json.loads(browser.answer(lambda: odd, "[]")) == \
        {"ratio": None, "range": [1.0, None, None], "pair": [None, 3]}

    assert browser.answer(lambda a, b: {"sum": a + b}, "[2, 3]") == '{"sum":5}'


def test_the_browser_and_the_server_summarise_a_run_alike():
    """
    Two backends, one history.

    The worker used to assemble its replies itself and keep nothing, so the
    two backends had their own rules for what a reply holds. Both go through
    gol_series.History now; this holds them to answering the same request with
    the same numbers. The family count is the one difference, and it is on
    purpose: it needs every iteration in order, which only the server reads.
    """
    import gol_store

    browser = _browser_module().Worlds()

    same = lambda a, b: a == b or (isinstance(a, float) and isinstance(b, float)
                                   and math.isnan(a) and math.isnan(b))

    with tempfile.TemporaryDirectory() as tmp:
        original = gol_store.BASE_DIR
        gol_store.BASE_DIR = tmp
        try:
            run_id, _, written = unrecorded_run(24)
            for points in (2, 5, None):
                server = gol_record.build_series(run_id, points=points, heavy=False)
                plan = browser.series_plan(run_id, written, points, ["nodes"])
                frames = [{"index": f, "frame": gol_store.read_frame(run_id, f)}
                          for it in plan["iterations"] for f in (2 * it, 2 * it + 1)
                          if f < written]
                reply = browser.series_absorb(run_id, frames, plan["heavy"])
                for key in ("done", "complete", "heavy", "totalPoints", "frames", "stride"):
                    assert reply[key] == server[key], (points, key, reply[key], server[key])
                for key in server["keys"]:
                    if key == "cladesInWindow":
                        continue
                    ours, theirs = reply["series"].get(key), server["series"][key]
                    assert ours is not None and all(map(same, ours, theirs)), (points, key)
        finally:
            gol_store.BASE_DIR = original


# ---------------------------------------------------------------------------
# The form
# ---------------------------------------------------------------------------

def test_the_interface_offers_every_setting_the_engine_has():
    """
    A field nobody can set is a field nobody knows about.

    Every knob on SimConfig should have somewhere in the form to set it, and
    every checkbox should say what it does — the pre-pass and messages were
    both added without one, and an unexplained checkbox is a checkbox nobody
    touches. The settings shown beside every run are read off this same form
    (RunsView.settingGroups), so they cannot fall a setting behind it either.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    page = open(os.path.join(root, "web", "index.html")).read()

    # Not offered on purpose: the seed graph's rewire probability and the
    # run-control knobs are set elsewhere or left at their defaults.
    from dataclasses import fields as dataclass_fields
    missing = [f.name for f in dataclass_fields(SimConfig)
               if f'data-cfg="{f.name}"' not in page]
    assert not missing, f"no form field for: {', '.join(missing)}"

    for name in ("exchange_messages", "message_prepass", "allow_handover",
                 "allow_revolutions"):
        block = page.split(f'data-cfg="{name}"')[1].split("</div>")[0]
        assert "<small>" in block, f"the {name} checkbox has no explanation under it"


def test_every_mode_is_one_table_the_form_and_the_engine_share():
    """
    A setting that names a rule — which brain, when to prune, how long a
    connection may idle, how the dead are shared out — has one table in
    gol_config. validate() refuses anything outside it, the engine looks its
    rule up in it, and the form offers exactly what it holds. The engine used
    to fall back onto some other rule for a value it did not know.
    """
    import GraphOfLifeSimple as engine

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    page = open(os.path.join(root, "web", "index.html")).read()
    assert set(engine.BRAIN_KINDS) == set(SimConfig.MODES["brain_kind"])
    for setting, allowed in SimConfig.MODES.items():
        select = re.search(rf'<select[^>]*data-cfg="{setting}"[^>]*>(.*?)</select>', page, re.S)
        assert select, f"the form offers no choice of {setting}"
        offered = re.findall(r'<option value="([^"]+)"', select.group(1))
        assert sorted(offered) == sorted(allowed), (
            f"the form offers {setting} = {offered}, the engine knows {list(allowed)}")
        try:
            SimConfig(**{setting: "no-such-rule"}).validate()
        except ValueError as exc:
            assert setting in str(exc), str(exc)
        else:
            raise AssertionError(f"an unknown {setting} was accepted")

    # Past validation — a config built by hand — the engine refuses it too.
    world = new_world(small())
    world.cfg.prune_after = "no-such-rule"
    try:
        world._prunes_after(1)
    except KeyError:
        pass
    else:
        raise AssertionError("the engine fell back onto some rule for an unknown prune_after")


if __name__ == "__main__":
    sys.exit(runner.main(globals()))
