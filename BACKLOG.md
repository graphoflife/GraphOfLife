# Backlog

Work that is known about and not done yet. Written 2026-09-04 after a read
through the whole codebase; every item re-checked against the code on
2026-09-06, which is when the ones marked **done** were worked and the ones
marked **was wrong** were found not to be true.

This is not a wishlist. Everything here is either a defect that has been
confirmed by reading the code, a gap that has already cost something, or a
decision that was deferred on purpose and should not be forgotten. Anything
speculative has been left out.

Each item says what is wrong, why it matters, and what finishing it looks
like. Sizes are rough: **small** is under an hour, **medium** is an afternoon.

**Three of the original items were wrong**, and they are kept below with what
is actually the case rather than deleted — an item that reads as confirmed and
is not costs an afternoon each time somebody picks it up. The lesson is in
section 9.

---

## 1. Tests for the view logic — **done**

`tests/test_view.js`, fifteen tests, in CI. What follows is the case that was
made for it; it held up, and it was cheaper than estimated — the six timing
functions needed **no** stubs at all, and `show()` needed one three-method
layout. Every hand-measured number below reproduced exactly against
`web/data/explain-run.json`.

Two things came up while writing it. The dot-count rule was defined inside
`_tokens`, a drawing function, where no test could reach it without copying the
formula; it is now `StepView.dots(held, ref)` and the painter calls it. And two
of the first assertions were **wrong, not the code** — the world does dip below
its full supply during a handover, because that is the tokens in flight, and
one token does fire at t=0, because the ball starts pointing at the first
agent. Both now assert the real invariant: the dip is exactly what is in the
air and never more, and shots leave in the order the ball comes round.

Still open from section 7: nothing tests `stats.js`, `render.js`, `force.js`,
`viewer.js`, `viewer-panels.js` or `graphstats.js`.

---

<details>
<summary>The original case for doing it</summary>

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

</details>

---

## 2. `_pick_index` can kill a long run — **was wrong; the code was dead**

The reasoning held and the conclusion did not. `np.random.choice` does reject a
probability vector that does not sum to 1 within tolerance, and dividing by a
float sum does not guarantee that. But `_pick_index` had **no callers** — its
last two went in `48109c6`, with the rewiring, and the function was left
behind. It could not end a run because it never ran.

It was read as a live defect for two days because the code was read and the
call sites were not. Deleted, along with the same dead branch in
`explain_minimal.py`'s `pick()`, whose one caller only ever asked for the
`argmax`. Nothing else in the repo passes `p=` to `np.random.choice`.

`_apportion` builds its `probs` the same way and is very much alive — but it
feeds largest-remainder, not `np.random.choice`, so there is no tolerance check
to fail.

---

## 3. Drift between the teaching script and the walk-through — **done, and it had already happened**

Written as a risk. It was not a risk; it was a live defect on the public site,
and it had been since the anchor was written.

The Explanation finds the lines it lights by searching `explain_minimal.py` for
literal text held in `web/js/explain.js`. Step 8's second region ran from
`'def resolve(staked: dict'` to `'    return hegemon'` — and matching was a
plain substring test, so the four leading spaces meant nothing and the search
stopped at the `return hegemon` **eight** columns in, the early exit for when
nobody revolts. The lit region ended twenty-two lines into a thirty-seven line
function and **stopped short of the mob walk the step's own words describe**.
The reader was pointed at the early return and told about the revolution.

Three things came out of it:

- Indentation now counts when an anchor has any (`Explain.carries`). An anchor
  written with four spaces is naming a line at that depth; throwing that away
  was the bug. Verified against all twenty anchors: it fixes step 8 and changes
  no other region.
- A missing anchor is warned about instead of silently lighting nothing. Two
  comments in that file already claimed it "says so". Neither was true.
- `tests/test_explain_anchors.js` resolves every step against the real script
  in CI, using the shipping `Explain.regions` rather than a copy of it. It
  checks that each anchor still matches, that a tail is unique **after its
  head** — global uniqueness is the wrong bar, and the resolver's actual search
  window is what matters — and that no region grows past sixty lines, which is
  what the hundred-and-forty-six-line region would have tripped.

The lesson is in section 9.

---

## 4. Smaller defects — one fixed, two were wrong

**`report["redistributed"]` lied when nobody survived — done.** The pool was
reported as redistributed even when `survivors` was empty and it had actually
been dropped in favour of minting `cfg.total_tokens` fresh for the resurrected
agent. Called cosmetic here, which was too kind: it reaches `gol_series.py`,
the viewer's General panel, and a per-run chart, and `test_stats_parity.py`
held both implementations to the same wrong answer. Now counted only when it is
really shared out, with two tests — one for each side of the branch — and the
statistic's own description corrected, since it also folds in
`tokens_created_per_phase` and never said so.

**Unescaped interpolation in `runs.js:74` — was wrong twice over.** Line 74 was
never that code, on the day this was written or since; and `escapeHtml`, which
the item held up as the careful example elsewhere in the same file, has never
existed anywhere in this repo. The underlying worry is also unfounded: the
progress `detail` goes through `textContent`, and every value it can hold is a
literal from our own source.

**`_deliver_messages` returns early on an empty outbox — was wrong.** The
guard is `if not outbox: return`, and the loop it skips is
`for u, notes in outbox.items()`, which over an empty dict iterates zero times.
Removing the guard would change nothing observable. The stated consequence —
the previous phase's messages persisting — does not follow from it either:
delivery overwrites per key and never cleared anyone, guard or no guard.

---

## 4b. Two things measured in the brains — **now measured against survival, and neither change is supported**

Neither is a defect. Both are numbers nobody chose, and the argument below was
that choosing them deliberately would change how a population behaves.

**It does not.** `research/pilot_brain_inputs.py`, thirty seeds each, sixty
iterations:

```
noise inputs      extinct   median n        last hidden   extinct   median n
  0                 1/30        400           10            8/30        388
  2                 0/30        363           32           10/30        322
  5                 0/30        393           64           13/30        324
 10                 1/30        421          128            9/30        320
```

The noise column is a clean null: taking noise away **entirely** is
indistinguishable from doubling it, and the medians wander without direction.
The loudness is real — 38.5% of the first layer's variance over five inputs,
one noise draw about 4.4x a magnitude, which reproduces what is claimed below —
but nothing downstream cares. Leave it alone.

The binary column is the more useful result, because it points the other way
from the recommendation. Widening the last hidden layer **does** buy
expressiveness: on identical observations, distinct staking scores go 1 → 7 →
6 → 10 and exact ties fall from 26.9% to about 4%. It buys no survival.
Extinction is flat to worse and the median population drifts *down*. At n=30
the binomial standard deviation is about 2.6, so 8 against 13 does not
separate — but there is no reading of this table in which widening helps, and
"widening the last hidden layer is the lever" should not be acted on from
memory later.

What that leaves is a real question rather than a patch: a brain that can say
more precisely what it wants does no better, which is either something about
this game or something about how little of the decision the score actually
carries. Worth a look before either preset moves.

Two methodological notes from doing it, both of which changed the answer:
measure on a **warmed** world, since at iteration 0 every founder holds an
equal share and every magnitude is nearly constant, which made noise look like
half of what a brain hears; and compare architectures on the **same**
observations, since giving each its own world compares populations rather than
brains — and two of them had died before the measurement was taken.

The original entries follow, with the numbers they were written from.

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
before changing the preset. *(Measured. It is not the lever — see the table
above.)*

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

**`requirements.txt` used floors, not pins — done.** Ceilings added, so a
green CI means something specific. Worth knowing what this does *not* cover:
`web/js/sim-worker.js` runs the same engine in the browser under Pyodide, with
numpy from Pyodide's own bundle and networkx from an unconstrained
`micropip.install()`. That is a third environment, it is the one that actually
ships, and nothing checks it against these bounds. Noted in the file itself.
A lockfile would still be better than bounds. **Small, still open.**

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

**Nothing compares one run against another.** The tab that was going to do it
is now **Research**, and it holds the lineage forest, the flow modules and the
literature — all of which look at a single run, or at no run at all. The
original intent, comparing runs against each other and against the settings
that produced them, is still unimplemented and still needs a specification
before it needs code. It is also what the ablation method in the Literature
page calls for, which makes it the higher-value half.

**README's opening line disagreed with the front page — done.** Both now read
"An Artificial Life Algorithm".

---

## 7. What has not been reviewed

Stated so that silence is not mistaken for a clean bill of health.

The Python was read thoroughly: engine, store, server, series, tools. The
JavaScript was read selectively — worker lifecycles, error paths, asset
handling, and the files under active edit.

**Not audited line by line:** `web/js/stats.js` (1,224 lines),
`web/js/render.js` (804), `web/js/force.js` (870), `web/js/viewer.js` (725),
`web/js/viewer-panels.js` (584), `web/js/graphstats.js` (759). Together that is
roughly 4,900 lines, most of the browser codebase by volume. `graphstats.js` is
partly protected by the parity test against `gol_series.py`; the rest is not
covered by anything. `stats.js` is the only one over a thousand lines, and the
only one of the six that has changed since this was written.

`stepview.js` was on this list and is off it: `tests/test_view.js` covers its
timing and accounting, though not its drawing.

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
showed agents that had already faded out. A third since: `window.innerWidth`
reads **0**, so every layout measurement taken through the pane — element
widths, whether something wrapped, whether a column collapsed — is worthless.
Computed styles are still trustworthy, and that is how a CSS specificity bug
was caught after the geometry proved unreadable.

If a browser measurement looks impossible, check this first.

---

## 9. What the wrong entries cost

Three items in this file were confirmed defects that were not defects, and one
was a risk that had already happened. They stayed that way for two days across
several sessions.

The common thread is that each was written from **reading the code and not the
callers**. `_pick_index` was analysed correctly and had no callers.
`_deliver_messages`' guard was quoted correctly and skips a loop that does
nothing. The `runs.js` item named a line that never held that code and a helper
that has never existed here. Meanwhile the one genuine live defect — a step of
the Explanation pointing at the wrong half of a function on the public site —
was filed as a hypothetical, because nobody resolved a single anchor against
the script.

So, for anything added below:

- **Grep for the callers before calling something a defect.** Reachability is
  part of the claim, not a detail.
- **Run the thing once.** Resolving the anchors took one command and would have
  turned a "small, worth doing eventually" into "the site is wrong now".
- Quote a line number *and* enough of the line to find it again when the
  numbers move, because they do.
