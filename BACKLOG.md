# Backlog

Work that is known about and not done yet. Written 2026-09-04, after a read
through the whole codebase.

This is not a wishlist. Everything here is either a defect that has been
confirmed by reading the code, a gap that has already cost something, or a
decision that was deferred on purpose and should not be forgotten. Anything
speculative has been left out.

Each item says what is wrong, why it matters, and what finishing it looks
like. Sizes are rough: **small** is under an hour, **medium** is an afternoon.

---

## 1. Tests for the view logic — do this first

**Medium.** The single highest-leverage thing in this file.

There are thirty tests over roughly 3,500 lines of Python engine and **none**
over roughly 7,000 lines of browser code. `node --check` in CI is a syntax
check, not a test. Every defect found in the last several working sessions was
in the JavaScript, and every one of them was caught by a person looking at the
screen: the gaze sort that made the supply ball jitter, the conquest colour
leaking into the staking phase, a code region that lit a hundred and forty-six
lines, arrows hanging fifty-nine pixels off the card, piles that filled before
anything had reached them.

That is the bottleneck. The layer that changes most often has no net under it,
so the net is a human being.

It is cheap right now, which is the reason to do it before anything else:

- `web/js/stepview.js` is a plain `const StepView = {...}` at the top level.
- The timing and accounting functions — `_staking`, `_inheritance`,
  `_conquest`, `_supply`, `mutating`, `gaze` — touch **no** DOM at all. There
  are eleven canvas references in the whole 1,048-line file and all of them are
  in the drawing paths.
- Node 18 is already installed and already used by CI.
- `tests/test_stats_parity.py` establishes the pattern of running this
  project's JavaScript from a test, so there is a shape to copy.

No new dependency, no refactor.

### What the tests should assert

These are not invented. Every one was measured by hand in a browser while the
feature was being built, confirmed correct, and then lost when the session
ended. They are exactly the properties that break silently.

**Token accounting** — the ones where being wrong is invisible on screen. A
pile that fills too early still looks like a pile filling.

- Across the game step, the total held by all agents is the full supply at
  rest, dips mid-flight to what is drawn on the links, and returns to the full
  supply on arrival. Measured 500 → 96 → 500 on the recorded run.
- The new balances derived from `marks.staked` equal the engine's own count
  for every agent. Measured: 53 of 53 matched, 0 differed.
- In reproduction the parent loses what it gave as the tokens leave and the
  child gains it as they land. Measured 32 → 16 and 0 → 16 on the first pair.
- One token draws one dot; the run-wide reference draws fifteen; no agent ever
  draws more dots than it holds tokens.

**Elimination waves.**

- Every removed agent holding nothing is in wave one, every removed agent still
  holding something is in wave two, and the two sets together are exactly
  `marks.removed`. On the recorded stage: 1417 at 0 tokens in wave one, 1418 at
  16 tokens in wave two, 55 agents → 54 → 53.

**Motion that must not be synchronised or reversed.**

- Gaze phases are spread, not in step: about a twentieth of the population
  changes target in any tenth of a second, not all of it at once. Measured 34
  distinct phases across 43 eyes, 5 of 43 switching per 0.1s.
- The supply ball's angle never decreases across a step. Measured 0 backward
  steps over 130 frames; it was the *absence* of this test that let the jitter
  ship.

**Layout arithmetic** (needs a DOM, so either jsdom or leave it as a manual
check — see the note in section 8 about the browser pane).

- A code region that fits the panel is centred; one that does not starts at the
  top. Regions that fit measured 0px off centre.
- The navigation stays inside the note card on every step, with and without an
  emblem.

**Done looks like:** `tests/test_view.js`, run by `node tests/test_view.js`,
following the same hand-rolled harness as `tests/test_engine.py` (collect
`test_*` functions, run them, print dots and failures). Added to the CI `test`
job next to the two Python suites.

---

## 2. `_pick_index` can kill a long run

**Small.** `GraphOfLifeSimple.py`, `_pick_index` (~line 205).

```python
probs = (vals / total) if total > 0.0 else np.full(len(vals), 1.0 / len(vals))
return int(np.random.choice(len(scores), p=probs))
```

`np.random.choice` rejects a probability vector that does not sum to 1 within
its tolerance. Dividing by a float sum does not guarantee that. It is rare, but
this is the only remaining known defect that can **end a run outright** rather
than degrade a picture, and runs here go to tens of thousands of iterations —
which is exactly the regime where a one-in-a-million event is a certainty.

**Done looks like:** renormalise (`probs /= probs.sum()`) or sample from the
cumulative sum directly, plus a test that feeds it a vector known to be off by
an ulp.

---

## 3. Drift test between the teaching script and the walk-through

**Small.**

The Explanation finds the lines it highlights by searching `explain_minimal.py`
for literal text held in `web/js/explain.js` (`STEPS[].code`). Rename a
function or reword a comment in the teaching script and the walk-through
silently highlights the wrong lines, or nothing. There is no error and nothing
on screen to say why.

Two tests already check that the teaching script runs and conserves tokens.
Neither checks that the anchors still find anything.

**Done looks like:** a test that loads `explain_minimal.py` and every anchor
pair out of `explain.js`, and asserts each pair resolves to exactly one region
and that no region exceeds some sane size — the hundred-and-forty-six-line
region that prompted the two-region support would have been caught by the
second half of that check.

---

## 4. Smaller confirmed defects

All read in the code, none currently causing visible harm.

**`report["redistributed"]` lies when nobody survives.**
`GraphOfLifeSimple.py:1136`. The pool is reported as redistributed even when
`survivors` was empty and it was actually discarded in favour of minting
`cfg.total_tokens` fresh for the resurrected agent. Cosmetic today; misleading
the moment anything plots it. **Small.**

**Unescaped interpolation in `runs.js:74`.** The Pyodide progress `detail`
string goes into `innerHTML` without escaping. Same-origin and local, so not
exploitable as things stand — but two hundred lines away the same file
carefully runs run names through `escapeHtml`. Inconsistent care is how the
real one eventually ships. **Small.**

**`_deliver_messages` returns early on an empty outbox**, so when messaging is
off or nobody wrote anything, the previous phase's messages persist instead of
clearing. `_prune_stale_messages` catches most of the consequences. **Small.**

---

## 4b. Two things measured in the brains, not yet acted on

Neither is a defect. Both are numbers nobody chose, and both would change how
a population behaves if they were chosen deliberately.

**A third of what a float brain responds to is noise.** The inputs are not
normalised, so how loud an input is depends on the range it happens to live in.
Measured across a phase, the first layer's variance splits: magnitudes 50% over
28 inputs, noise **35% over 5**, messages 15% over 20. Per input, a noise draw
is about four times louder than a magnitude and nine times louder than a
message. That is an accident of `uniform(-2, 2)` against the spread of the
others, not a decision that gambling should be a third of what an agent
attends to. Sigmoid saturation is fine — 8.6% pinned in the first hidden layer
and none after — so the network itself is healthy. **Small** to change, and it
changes behaviour, so it wants an opinion before a patch.

**A binary brain's output layer is coarse.** Its counts run about -12 to 12
against the float brain's continuum, giving 18 distinct BLOTTO scores where
float gives 936. The width comes from the last hidden layer: 64 units of
-1/0/+1 with two thirds zero sums to roughly ±8. The consequences are visible
in the decisions — paired heads land exactly equal 8.9% of the time, which
`_choose_binary` answers with a coin, and both non-positive 35.7% of the time,
which `_share_of_first` answers with an even split. The float brain never ties
and falls back 28.8%.

Widening the last hidden layer is the lever. Worth measuring against survival
before changing the preset.

---

## 4c. Done — the ladder has a band and a place inside it

Kept here because the reasoning is worth having, and because it names the one
thing it did **not** fix.

One ladder resolved 15 levels across a range of e^12 in tokens, which made a
level a factor of **2.23x** — an agent could not tell a hundred tokens from a
hundred and eighty. The same sixteen rows are now twelve for the band and four
for the place inside it: **36 levels, a factor of 1.40x**.

The obvious alternative — encode the number in the bits the way a float does —
was measured and rejected. A binary unit computes `sum(w_i * bit_i)` with `w`
in {-1, 0, +1} and thresholds it. That sum cannot weight a row by two to its
position, so place value puts the magnitude where nothing can read it:

| encoding | codes | mean rank corr | units that read it |
|---|---|---|---|
| one ladder | 15 | 0.530 | 73% |
| **band + place (shipped)** | **36** | **0.491** | **71%** |
| positional | 3878 | 0.189 | 26% |
| Gray | 3878 | 0.125 | 28% |

Gray is the control that rules out the usual explanation: it fixes the
"127 and 128 share no bits" adjacency problem and is *worse*. The constraint is
not adjacency, it is that a `±1` sum has to be monotone in the value.

**What this did not fix.** The binary brain's decisions are still coarse:
paired heads land exactly equal about 9% of the time and staking still has
about 17 distinct scores. Those come from the *output* layer, whose range is
set by the last hidden layer — 64 units of -1/0/+1 with two thirds zero sums to
roughly ±8. Nothing about the input encoding touches it. Widening the last
hidden layer is the lever, and it wants a survival comparison before the preset
moves.

---

## 5. Structural, for the long term

**No linter or formatter.** Style is held by discipline alone, which works
while there is one author and stops working the moment there is not. Ruff for
the Python and ESLint for the JavaScript, both with a near-empty config, would
cost an afternoon and catch the boring half of a review automatically.
**Medium.**

**`requirements.txt` uses floors, not pins.** `numpy>=2.0` and
`networkx>=3.0` mean CI and a local machine can silently be running different
libraries, so a green CI does not mean anything specific. A lockfile, or at
minimum upper bounds, makes a passing build reproducible. **Small.**

**IndexedDB has no migration path.** `web/js/runstore.js`'s
`onupgradeneeded` only creates stores that do not exist. There is no path for
*changing* one, so the first schema change has to be written from scratch
against data already sitting in people's browsers. Not worth inventing now —
but worth knowing it is unwritten, and worth writing against the first real
change rather than after it. **Deferred on purpose.**

**Nothing checks that `explain_minimal.py` and the engine agree.** The
teaching script is presented as the same algorithm. Two tests check it runs and
that its revolution goes to the strongest rebel. Nothing checks the two
implementations produce the same *kind* of behaviour, so they can drift apart
while both remain individually correct. Hard to test well; worth thinking about
before the teaching script is edited again. **Medium.**

---

## 6. Product decisions that were deferred, not dropped

**Explanation: why each mechanic was chosen.** Every step explains what
happens. The intent was for each to also carry a short note on *why* the
algorithm does it that way — which is the part a reader cannot get from the
code. Explicitly postponed during the build.

**The Analysis tab is an empty placeholder.** `web/index.html:524` describes
what it is for — comparing runs against each other and against the settings
that produced them — and nothing implements it. It needs a specification before
it needs code.

**README's opening line still says "A new kind of Artificial Life Algorithm".**
The front page was deliberately changed to "An Artificial Life Algorithm". The
README was left alone because only the main menu was asked for. Either is fine;
they should match.

---

## 7. What has not been reviewed

Stated so that silence is not mistaken for a clean bill of health.

The Python was read thoroughly: engine, store, server, series, tools. The
JavaScript was read selectively — worker lifecycles, error paths, asset
handling, and the files under active edit.

**Not audited line by line:** `web/js/stats.js` (1,207 lines),
`web/js/render.js` (804), `web/js/force.js` (870), `web/js/viewer.js` (725),
`web/js/viewer-panels.js` (584), `web/js/graphstats.js` (759). Together that is
roughly 4,900 lines, most of the browser codebase by volume. `graphstats.js` is
partly protected by the parity test against `gol_series.py`; the rest is not
covered by anything.

Absence of findings there is absence of looking.

---

## 8. A note on verifying browser work

Worth writing down because it has caused two wrong conclusions already.

The in-app browser pane does not composite frames, which means:

- `requestAnimationFrame` never fires, so any animation loop must be driven by
  hand to observe anything.
- `scroll-behavior: smooth` and `scrollTo({behavior: 'smooth'})` never make
  progress — every scroll measurement reads zero until instant scrolling is
  forced.
- Screenshots time out; canvas contents have to be read back and inspected
  directly.

Two separate rounds of measurement were invalid before this was understood: a
set of scroll positions that all read zero, and a set of elimination captures
taken by fast-forwarding past the removals and then rewinding the clock, which
showed agents that had already faded out.

If a browser measurement looks impossible, check this first.
