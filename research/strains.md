# Strain registry

Every algorithm variant an experiment has been run on. The scheme is defined in
`Research.md` §5.

**A strain is never removed and never redefined.** If a mechanic has to change
meaning, it gets a new name. This file is what makes a result from a year ago
still reproducible when six more mechanics exist.

```
gol-<SPEC>[+<mechanic>[=<value>]]...
```

- `SPEC` is bumped **only** for a change that cannot be expressed as an
  optional flag — a bug fix that alters results, or a change of a frozen
  default.
- Mechanics whose value equals the frozen default are **omitted**, which is why
  adding a new mechanic never changes an existing strain's name.
- Mechanics are listed alphabetically. A boolean that is on appears bare.

---

## SPEC 1

The algorithm as of `9cee9fd`. Implemented in `gol_config.py` as `SPEC`,
`MECHANICS`, `PARAMETERS` and `INFRASTRUCTURE`; `SimConfig.strain_id()` spells
the name. `tests/test_engine.py` asserts that every setting is classified as
exactly one of the three, so a new setting cannot be added without deciding
which it is.

### Mechanics — implemented, and settable

| mechanic | frozen default | notes |
|---|---|---|
| `brain_kind` | `float` | also `float16`, `binary` |
| `exchange_messages` | `true` | |
| `message_prepass` | `true` | |
| `allow_handover` | `true` | |
| `allow_revolutions` | `true` | the non-transitivity generator |
| `tokens_created_per_phase` | `0` | a magnitude, but 0 against anything else is a closed economy against one that mints — a different algorithm, not a different setting |
| `random_decisions` | `false` | the control: agents never read their inputs and every decision is taken from noise instead |
| `allow_gifting` | `false` | an agent may hand tokens to a neighbour during reproduction; the transfer is also flow, so it is the one way to deliberately keep a link that would otherwise lapse. Adds three output heads and the input flag saying whether a link is about to |
| `prune_after` | `blotto` | when unused links are cut: `blotto`, `reproduction` or `both`. Cutting after reproduction is what gives a gift somewhere to land |
| `inactive_window` | `phase` | how far back "used" looks: the phase just ended, or the last whole `iteration`. A newly made link counts as used either way |
| `redistribution` | `uniform` | how the estate of the dead is shared out. `by_tokens` weights it by what each survivor already holds, making every cull a concentration event |

### What a new run is offered

The frozen defaults above say what `gol-1` means and never move. What the
new-simulation form arrives filled in with is a separate thing —
`NEW_RUN_DEFAULTS` in `gol_config.py` — and it is free to move as the evidence
does. Since 2026-10-02 it offers `gol-1+brain_kind=float16` — half-precision
brains, every other mechanic at its frozen value — with two parameters
changed: `message_amount=30` and `mutation_probability=0.2`. Parameters are
cited beside a strain, not in it. This is also the book's baseline B1.

Until then it offered

```
gol-1+allow_gifting+inactive_window=iteration+prune_after=reproduction
```

and runs started in that time announce themselves as that algorithm; every run
on disk keeps the name it was given. Verified by hashing three seeds over
fifteen iterations against the engine as it stood before those mechanics
existed: with all four at their frozen values the frames are identical.

### Mechanics — reserved, not yet implemented

Named and defaulted now on purpose, so the first experiment to turn one on does
not also get to choose what it is called. **They are not config fields**, so
they cannot be set — asking for one raises rather than silently producing a
strain name for a run that ignored it.

| mechanic | frozen default | section |
|---|---|---|
| `mutate_on_replication` | `false` | Research.md §4.1 |
| `germline` | `false` | §4.2 |
| `token_colours` | `1` | §4.3 |
| `mutual_flow_yield` | `0` | §4.4 |
| `edge_proposal` | `false` | §4.5 |
| `growable_layers` | `false` | §4.6 |
| `local_rules` | `false` | §4.7 |
| `contracts` | `false` | §4.8 |
| `structure_replication` | `false` | §4.9 |

Implementing one means adding the field with exactly the default above and
adding it to `MECHANICS`. Nothing else — every existing strain keeps its name,
because a mechanic at its default is never listed.

### Parameters and infrastructure

`total_tokens`, `n_nodes`, `k_neighbors`, `rewire_p`, `hidden_layers`,
`brain_bits`, `message_amount`, `random_input_amount`, `mutation_probability`,
`mutation_noise_std`, `mutation_sparsity`, `extinction_threshold` and `seed`
are **parameters**. Cited with an experiment, not part of the strain —
otherwise every seed would be its own algorithm.

`checkpoint_every`, `export_every` and `export_decisions` are
**infrastructure** and affect only what is recorded.

### Where the strain is written

On every store that can be found on its own:

- **run metadata** — `gol_store.create_run`, and `gol_browser.create` for the
  browser backend, so both backends label a run the same way.
- **the checkpoint** — a `strain` array in the `.npz`. A checkpoint gets
  copied, shared and resumed elsewhere; without this it is a world with no way
  to say what rules it lived under.
- **the series cache** — `series.json` is the file an analysis actually loads,
  and a chart made from it should not have to go back to the run directory to
  learn which algorithm it is of.

---

## Strains used

| strain | mechanics on | first used | what for |
|---|---|---|---|
| `gol-1` | — (all defaults) | `9cee9fd` | The baseline. Everything in `Research.md` Appendix A. |
| `gol-1+random_decisions` | `random_decisions` | this commit | The control. Every other mechanic identical, so a difference between this and `gol-1` is attributable to the agents reading their inputs and to nothing else. |
| `gol-1+allow_gifting+brain_kind=float16+inactive_window=iteration+prune_after=reproduction` | `allow_gifting`, `brain_kind=float16`, `inactive_window=iteration`, `prune_after=reproduction` | the book's lab | The book's first baseline B1, on the morning of 2026-10-02: what a new run was offered then, on half-precision brains. Replaced the same day; its only runs, a first Experiment 1, are archived. |
| `gol-1+brain_kind=float16` | `brain_kind=float16` | the book's lab | The book's baseline B1 (`book/experiments/B1.json`) since 2026-10-02, and exactly what a new run is offered: half-precision brains, every other mechanic at its frozen value, with `message_amount=30` and `mutation_probability=0.2`. |
| `gol-1+brain_kind=float16+random_decisions` | `brain_kind=float16`, `random_decisions` | the book's lab | The control of the book's Chapter 37 (Do the brains matter?): the baseline B1 with every decision taken from noise instead of from the brains. |

---

## Errata

Bugs fixed without a new SPEC, because no recorded run used the strains they
touched. The commit that fixed each is what separates the two meanings.

- **Binary brains with gifting** (`brain_kind=binary` with `allow_gifting`),
  fixed on 2026-10-09 in the commit that made `BinaryBrain.encode` read
  `SimConfig.input_kinds()`. The encoder cut every observation after one flag,
  but gifting puts two there: the at-risk flag was laddered as if it were a
  magnitude, and the last magnitude reached the brain as a raw number. The row
  count was unchanged, so no shape check saw it. No run of a binary brain with
  gifting had been made by the lab or the book. A checkpoint now records how
  many flags its brains read, and a binary+gifting checkpoint written before
  the fix is refused on resume, saying why.

---

## Recording an experiment

Four fields. The strain makes results comparable; the commit makes them
reproducible.

```
strain:  gol-1
setup:   total_tokens=2500 n_nodes=50 k_neighbors=6 hidden_layers=[12,10]
         mutation_probability=0.5
seeds:   1..30
commit:  9cee9fd
```

The strain does not pin bugs, and a bug that changes results is exactly the
thing that makes two runs of "the same" algorithm disagree. That is what
`commit` is for, and why it is not optional.
