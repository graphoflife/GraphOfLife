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

The algorithm as of `9cee9fd`. Frozen defaults:

| mechanic | frozen default | notes |
|---|---|---|
| `brain_kind` | `float` | also `float16`, `binary` |
| `exchange_messages` | `true` | |
| `message_prepass` | `true` | |
| `allow_handover` | `true` | |
| `allow_revolutions` | `true` | the non-transitivity generator |
| `mutate_on_replication` | `false` | *not yet implemented* — §4.1 |
| `germline` | `false` | *not yet implemented* — §4.2 |
| `token_colours` | `1` | *not yet implemented* — §4.3 |
| `mutual_flow_yield` | `0` | *not yet implemented* — §4.4 |
| `edge_proposal` | `false` | *not yet implemented* — §4.5 |
| `growable_layers` | `false` | *not yet implemented* — §4.6 |
| `local_rules` | `false` | *not yet implemented* — §4.7 |
| `contracts` | `false` | *not yet implemented* — §4.8 |
| `structure_replication` | `false` | *not yet implemented* — §4.9 |

The unimplemented rows are listed now on purpose: they fix the names and the
defaults before anything uses them, so the first experiment to turn one on does
not also get to choose what it is called.

`total_tokens`, `n_nodes`, `k_neighbors`, `rewire_p`, `hidden_layers`,
`brain_bits`, `message_amount`, `random_input_amount`, `mutation_probability`,
`mutation_noise_std`, `mutation_sparsity`, `extinction_threshold` and `seed`
are **parameters**, not mechanics. They are cited with an experiment but are
not part of the strain — otherwise every seed would be its own algorithm.

`checkpoint_every`, `export_every` and `export_decisions` are
**infrastructure** and affect only what is recorded.

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
