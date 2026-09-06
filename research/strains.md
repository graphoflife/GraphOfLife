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
