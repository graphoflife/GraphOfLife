# Is a run reproducible?

## In short

Yes — on the same machine with the same libraries, a run is decided by its
settings and its seed and by nothing else. Stopped and reloaded in a new
process, cut off without warning and resumed, or given every processor
thread: all nine variants matched their reference run exactly, frame for
frame, down to the last number of the final state. One thing did break
reproducibility before this chapter: two simulations running side by side in
the server changed each other from their very first frame. That was fixed
first, and is now tested.

## Thesis

```thesis E01
```

Everything after this chapter compares runs with each other. That only means
something if a run is decided by its settings and its seed and by nothing
else — not by when it was stopped, not by a crash, not by how many threads the
computer happened to give it.

## Method

```experiment E01
```

Every run starts from the baseline B1 with 10,000 tokens — a hundred
founders — and runs for 30 iterations, by which time it holds two to four
thousand agents. Each of three seeds is run four ways:

- **straight** — from start to iteration 30 in one go. This is the reference.
- **stopped at 10** — stopped at iteration 10, saved, and continued later in a
  *new* process, from the checkpoint on disk.
- **cut off at 15** — the process dies at iteration 15 without saving, the way
  it would if the power went. Its last checkpoint is from iteration 10, so it
  resumes from there and lives iterations 10 to 15 a second time.
- **all BLAS threads** — the matrix library may use every core. Normally the
  lab keeps it to one thread per run.

For each variant, every recorded frame, every row of statistics and the final
checkpoint — every array of it, including the state of the random number
generator — is compared with the straight run of the same seed. Files cannot
be compared as they are, because compressed files carry the time they were
written; their contents are compared instead.

Two more checks were made outside the lab, with `tools/trajectory_digest.py`,
which runs six small worlds (each kind of brain, the original defaults and
today's, the other way of sharing out the dead, and the control that ignores
its brain) and fingerprints everything they record:

- **in company**: each world advanced in turn with a second world in the same
  process — the way the server runs two simulations at once;
- **the engine before**: the same, on the engine as it was before this book
  began (commit `d22ff47`).

## Results

All nine variants recorded exactly what their reference recorded.

| Variant | Seed 1 | Seed 2 | Seed 3 | Compared, each |
|---|---|---|---|---|
| stopped at 10 | identical | identical | identical | 60 frames, 60 rows, 29 arrays |
| cut off at 15 | identical | identical | identical | 60 frames, 60 rows, 29 arrays |
| all BLAS threads | identical | identical | identical | 60 frames, 60 rows, 29 arrays |

The worlds were not small by then: at iteration 30 they held 3,907, 2,576 and
2,019 agents (seeds 1, 2 and 3), after peaks of 6,596, 3,923 and 2,871. The 29
arrays of a checkpoint include every weight of every brain, every agent's
tokens, every connection, every message in flight, and the random number
generator's state.

The provenance kept with each run shows what each variant went through: the
stopped runs were made in two sessions, 0 to 10 and 10 to 30, each in its own
process; the cut-off runs too, the first ending at iteration 15 with no
checkpoint and the second starting again from iteration 10.

**In company, before this book**, every one of the six small worlds recorded
something different from the very first frame, because every world in a
process drew its random numbers from numpy's single shared stream, and
starting a world reseeded that stream for all of them. The server runs
simulations as threads of one process, so any two runs that were going at the
same time disturbed each other, and neither could have been made again from
its seed. The browser had the same problem.

**The fix** gave every world a random stream of its own. A stream seeded the
same way is the same stream, so a run on its own is exactly the run it always
was: all six fingerprints are unchanged by the fix. In company, each world now
records exactly what it records alone.

## Conclusion

A run is a function of its settings and its seed. Stopping it, crashing it,
or giving it more threads changes nothing — so the lab can pause, resume and
recover runs freely, and every later chapter can compare runs knowing that a
difference between them comes from what was changed and not from how they
were run.

Three limits remain, and they matter for how results are reported:

- **Other machines.** This was tested on one computer with one set of
  libraries. A different processor or a different numpy can round a sum in
  the last bit differently, and this simulation turns a last-bit difference
  into a different history. So every run records its Python, numpy, networkx,
  BLAS library, processor and instruction set, and the lab will not continue a
  run under different ones.
- **The browser.** The version of the engine that runs in the page uses other
  builds of numpy and networkx and is not expected to match the server.
- **Older runs.** Runs made by hand before 2 October 2026, while another run
  was going in the same server, cannot be made again from their seed. Runs
  made one at a time can.

## Details

| | |
|---|---|
| Runs | 12: `B1-10000-s001` … `-s003`, each also as `-stopped-at-10`, `-cut-off-at-15`, `-all-blas-threads` |
| Strain | `gol-1+allow_gifting+brain_kind=float16+inactive_window=iteration+prune_after=reproduction` |
| Engine | snapshot `04f533dd0bf4b7c1`, commit `f936c96` |
| Environment | Python 3.12.3, numpy 2.5.1, networkx 3.6.1, OpenBLAS (scipy-openblas 0.3.33), Intel Core Ultra 7 258V, numpy dispatch AVX2 |
| Results | `book/results/E01.json`, with every run's seed, sessions, time and memory |
| Fingerprints | the six worlds of `tools/trajectory_digest.py`: `0f89434c…`, `1fc6c25a…`, `ad0d4552…`, `b5aeb19b…`, `ed3539b3…`, `8fd127e6…`, the same before and after the fix |
| Tests | `tests/test_engine.py` and `tests/test_lab.py` hold all of this: two worlds in turn, two runs in two threads, a stop and resume through the store, a run cut between checkpoints, a worker terminated mid-run |
