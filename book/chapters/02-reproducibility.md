# Is a run reproducible?

## In short

Yes. Every run the lab or the server makes is reproducible, exactly. Made
again from its settings and its seed — in one go, stopped and reloaded in a
new process, or cut off without warning and resumed — it records the same
frames and the same statistics and ends in the same state, down to every
brain weight and the random number generator. All twelve runs of this chapter
were also made again from nothing, each in a fresh process, and came out the
same.

The experiment found the one condition this rests on: the matrix library has
to use the same number of threads. On all of the computer's threads instead
of one, two of three runs kept the same history but ended with different last
digits in the messages their agents were about to read. So every run is made
on one thread: the lab always did this, and since this chapter so does the
server. Before this chapter, two simulations running side by side in the
server also disturbed each other from their very first frame; that was fixed
first.

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
founders — and runs for 30 iterations, by which time it holds one to three
thousand agents. Each of three seeds is run four ways:

- **straight** — from start to iteration 30 in one go. This is the reference.
- **stopped at 10** — stopped at iteration 10, saved, and continued later in a
  *new* process, from the checkpoint on disk.
- **cut off at 15** — the process dies at iteration 15 without saving, the way
  it would if the power went. Its last checkpoint is from iteration 10, so it
  resumes from there and lives iterations 10 to 15 a second time.
- **all BLAS threads** — the matrix library may use every core. Normally the
  lab keeps it to one thread per run.

All twelve are runs of their own, shared with no other experiment. For each
variant, every recorded frame, every row of statistics and the final
checkpoint — every array of it, including the state of the random number
generator and the messages in flight — is compared with the straight run of
the same seed. Files cannot be compared as they are, because compressed files
carry the time they were written; their contents are compared instead.

Three more checks were made outside the lab:

- **in company:** with `tools/trajectory_digest.py`, six small worlds (each
  kind of brain, every mechanic on, the baseline, the original defaults, the
  other way of sharing out the dead, and the control that ignores its brain)
  were each advanced in turn with a second world in the same process — the way
  the server runs two simulations at once — and fingerprinted;
- **the engine before:** the same, on the engine as it was before this book
  began (commit `d22ff47`);
- **threads, longer:** seeds 2 and 3 run for 150 iterations on one thread and
  on all threads, compared iteration by iteration.

## Results

**Stopping, resuming and crashing change nothing.** Every one of these six
variants recorded exactly what its reference recorded.

| Variant | Seed 1 | Seed 2 | Seed 3 | Compared, each |
|---|---|---|---|---|
| stopped at 10 | identical | identical | identical | 60 frames, 60 rows, 29 arrays |
| cut off at 15 | identical | identical | identical | 60 frames, 60 rows, 29 arrays |
| all BLAS threads | identical | messages differ | messages differ | 60 frames, 60 rows, 29 arrays |

The worlds were of a fair size by then: at iteration 30 they held 1,298,
2,478 and 1,633 agents (seeds 1, 2 and 3), after peaks of 2,831, 4,944 and
3,890. The 29 arrays of a checkpoint include every weight of every brain,
every agent's tokens, every connection, every message in flight, and the
random number generator's state. The provenance kept with each run shows what
it went through: the stopped runs were made in two sessions, 0 to 10 and 10
to 30, each in its own process; the cut-off runs too, the first ending at
iteration 15 without a checkpoint and the second starting again from 10.

**Threads change the last digits of the messages.** On all threads, seeds 2
and 3 recorded the same 60 frames and the same 60 rows of statistics as their
references — every agent, every token, every connection, every decision —
and ended with the same brains, tokens and connections. The single difference
was in the numbers in flight: the messages agents had just written to each
other differed in their last bits. A message is a sum of many products inside
a brain, and on several threads the library adds those up in a different
order. Run again on all threads, seed 2 gave exactly what it gave the first
time on all threads; run again on one, exactly what it gave on one. So each
thread count is reproducible, and the two are not the same as each other.
Over 150 iterations, seeds 2 and 3 on one thread and on all threads still
never recorded a different frame: a last-bit difference in a message had not
yet changed a single decision. Nothing promises it never will.

**Made again from nothing.** Some hours after the experiment, all twelve runs
were made again from their settings and seeds, each in a fresh process and
an empty folder, and compared frame by frame and row by row over all 30
iterations with what had been recorded: every one was identical. They were
made a second time on the copy of the engine the next experiments use — taken
after the new-simulation form's defaults were changed — and were identical
again, so that change did not touch what a run does.

This did not show on the first baseline, whose messages had five numbers
instead of thirty: run the same way that morning, all nine variants were
identical (those runs are archived). The bigger brains of the baseline B1 do
products large enough for the library to split.

**In company, before this book**, every one of the six small worlds recorded
something different from its very first frame, because every world in a
process drew its random numbers from numpy's single shared stream, and
starting a world reseeded that stream for all of them. The server runs
simulations as threads of one process, so any two runs that were going at the
same time disturbed each other, and neither could have been made again from
its seed. The browser had the same problem. **The fix** gave every world a
random stream of its own; a stream seeded the same way is the same stream, so
a run on its own is exactly the run it always was — all six fingerprints are
the same before and after — and in company each world now records exactly
what it records alone.

## Conclusion

A run is reproducible: it is decided by its settings and its seed, given one
thread for the matrix library — which every run in this project now gets.
Stopping it or crashing it changes nothing, so the lab can pause, resume and
recover runs freely, and every later chapter can compare runs knowing that a
difference between them comes from what was changed, not from how they were
run.

The thesis as it was written fails in one clause, the one about threads, and
that is the most useful thing this chapter found. The lab always kept the
matrix library to one thread; the server did not, so a run started from the
Simulations tab was, in its last digits, not the run the lab would make. Since
this chapter the server keeps to one thread as well, at a cost of about 3% of
speed for a run on its own, and every run is the same run wherever on this
machine it is made.

The limits that remain:

- **Other machines.** A different processor or a different numpy can round a
  sum differently in the last bit, exactly as threads do. So every run
  records its Python, numpy, networkx, BLAS library, processor and
  instruction set, and the lab will not continue a run under different ones.
- **The browser.** The version of the engine that runs in the page uses other
  builds of numpy and networkx and is not expected to match the server.
- **Older runs.** Runs made by hand before 2 October 2026 while another run
  was going in the same server cannot be made again from their seed; and runs
  made by the server on all threads may be made again by the lab only until
  the first message rounds differently enough to change a decision.

## Details

| | |
|---|---|
| Runs | 12: `B1-10000-s001-straight` … `-s003-straight`, each also as `-stopped-at-10`, `-cut-off-at-15`, `-all-blas-threads` |
| Strain | `gol-1+brain_kind=float16`, with `message_amount=30` and `mutation_probability=0.2` |
| Engine | snapshot `f39288e90b003df0`, commit `ccba58f` |
| Made again | all 12 runs, from nothing in fresh processes (`python3 gol_lab.py verify E01 --runs 12 --iterations 30`), on `f39288e90b003df0` and on `f820369a1583af53`: identical in every frame and row |
| Environment | Python 3.12.3, numpy 2.5.1, networkx 3.6.1, OpenBLAS (scipy-openblas 0.3.33), Intel Core Ultra 7 258V, numpy dispatch AVX2 |
| Results | `book/results/E01.json`, with every run's seed, sessions, time and memory |
| Fingerprints | the six worlds of `tools/trajectory_digest.py`: `0f89434c…`, `1afc5884…`, `ad0d4552…`, `b5aeb19b…`, `ed3539b3…`, `8fd127e6…`, the same before and after the fix |
| Tests | `tests/test_engine.py` and `tests/test_lab.py` hold all of this: two worlds in turn, two runs in two threads, a stop and resume through the store, a run cut between checkpoints, a worker terminated mid-run, and the server keeping the matrix library to one thread |
