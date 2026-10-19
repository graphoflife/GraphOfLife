# Every era against three

> [!question] Questions of this chapter
> - In the longest worlds the book has, do later brains still get better at
>   the end, or had evolution stopped climbing?
> - Do later brains lose again to early ones, as they would if evolution
>   went round in circles?

<!-- experiment E28 -->

## Why ask

Every long world the book has played says the same thing in its contests:
later brains beat earlier ones, by less and less. In Experiment 18's two
longest worlds, brains mixed at birth and played to 50,000 iterations
([Chapter 81](81-five-mixed-worlds-fifty-thousand-iterations.md)), the
brains of 25,000 beat those of 10,000, and those of 50,000 then scored
0.54 and 0.42 against those of 25,000.

Contests between neighbours in time cannot tell three different stories
apart:
- **a plateau**, where evolution has stopped;
- **a slow climb**, where each step gains a little and many little steps add
  up;
- **a circle**, where each step beats the one before and loses to one long
  gone.

Rock, paper and scissors each beat the one before, for ever, and get
nowhere ([What would count as open-ended evolution](../notes/open-ended-test.md)).

Playing every era against the same few fixed eras tells the three apart:
- **On a plateau**, the scores against a fixed era stop rising.
- **On a slow climb**, they keep rising, however little each step gained.
- **In a circle**, they rise and fall again as the eras go round.

This experiment plays every 2,500th era of the two worlds against three
fixed ones: 2,500, 10,000 and 25,000.

## The thesis

<!-- thesis E28 -->
> [!quote] The thesis of Experiment 28, written down before any of its runs existed
> **The claim.** Evolution in Experiment 18's two longest worlds had stopped climbing by iteration 25,000. Against the brains of 25,000, the brains of the ten eras after it (27,500 to 50,000) do no better than a contest's own noise: their median score lies within the band an era scores against itself. Against the brains of 2,500, every era from 5,000 on still wins, beyond that band.
>
> **Why it would be so.** The two worlds' own contests (Chapter 81), 100 iterations each on both eras' worlds with two coins: in seed 5 the brains of 25,000 scored a median of 0.56 against those of 10,000, and those of 50,000 scored 0.54 against those of 25,000; in seed 6, 0.72 and then 0.42. Later brains beat earlier ones by less and less in every long world the book has played (Chapter 92; Experiment 19, 0.78 falling to 0.55). If the game these brains play still had room within their reach, the eras after 25,000 would pull ahead of 25,000 as the eras before it pulled ahead of 2,500.
>
> **It holds if:** In both worlds the median score of the eras 27,500 to 50,000 against 25,000 (each era's score the mean of its two contests) lies within the noise band, 0.5 plus or minus twice the standard deviation of the world's twelve contests of an era against itself (2,500, 10,000 and 25,000, four coins each); and every era from 5,000 to 50,000 scores above that band against 2,500.
>
> **It fails if:** In either world the median score of the eras 27,500 to 50,000 against 25,000 lies above the band (evolution was still climbing) or below it (it went back), or some era from 5,000 on fails to score above the band against 2,500.
<!-- /thesis -->

## The trials

Each contest is the book's lifetime contest
([Chapter 46](46-how-to-tell-a-world-that-keeps-evolving.md)):
- The world of one era is given brains drawn by coin from two eras.
- Every mutation is off, so no brain changes and every copy keeps its era.
- The world plays 100 iterations, and the later era's share of the agents
  is its score.

Every pair is played twice, once on each era's world. Each fixed era also
plays against itself four times. A score that differs from 0.5 by less than
those self-contests do is noise.

<!-- runs E28 -->
> [!info] The trials behind this chapter
> **240 trials**, contests: the world of one era, its agents given brains drawn by coin from two eras, every mutation off, and the later era's share of the agents counted at the end; each plays 100 iterations, from a world another run kept in its archive.
>
> - **against 2,500** — 76 trials: worlds `B2-10000-4af4a3-s005`, `B2-10000-4af4a3-s006`; eras 2,500 to 50,000, every 2,500; coins 17.
> - **against 10,000** — 72 trials: worlds `B2-10000-4af4a3-s005`, `B2-10000-4af4a3-s006`; eras 5,000 to 50,000, every 2,500; coins 17.
> - **against 25,000** — 68 trials: worlds `B2-10000-4af4a3-s005`, `B2-10000-4af4a3-s006`; eras 5,000, 7,500, 12,500, 15,000, 17,500, 20,000, 22,500, 25,000, 27,500, 30,000, 32,500, 35,000, 37,500, 40,000, 42,500, 45,000, 47,500, 50,000; coins 17.
> - **each against itself** — 24 trials: worlds `B2-10000-4af4a3-s005`, `B2-10000-4af4a3-s006`; eras 2,500, 10,000, 25,000; coins 17, 18, 19, 20.
>
> To play these trials again: `python3 gol_lab.py run E28`, or ▶ in the Book tab of a computer running `gol_server.py`. Each writes its settings and engine beside its result, in `GraphOfLifeRuns/<trial>/meta.json` and `result.json`.
<!-- /runs -->

The trials are played in the lab, which shows them beside this chapter and
plays them when ▶ is pressed: 240 contests of about a minute each.

<!-- turns -->
---

← [Chapter 94 · What a brain can compute](94-what-a-brain-can-compute.md) · [Contents](../README.md) · [Appendix A · What a simulation costs](A-costs.md) →
<!-- /turns -->
