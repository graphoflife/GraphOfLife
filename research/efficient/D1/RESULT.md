# D1. The rules as physics: one sheet (RESULT)

World: run `B1-10000-085a32-s4801` (s4003, 10,000 tokens), iterations 12,000 to 12,099 = 100 game frames
(each checked against its reproduction frame, index 2t, and the game frame, index 2t+1; for link cutting also
the game frame of the iteration before). Nothing was played; only recorded frames and `GraphOfLife.py` were read.
Code: `d1_check.py`, `d1_extra.py`; numbers: `d1_result.json`, `d1_extra.json`.

## The rules this world plays
`contest = lottery`, `allocation = strongest_first`, `redistribution = parents`, `link_search = off`,
`link_consent = mutual` (unused: only `link_search` reads it), `link_token = false`,
`link_upkeep = paid_or_used`, `prune_after = blotto`, `inactive_window = phase`, `keep_home = 0`,
`allow_gifting = true`, `allow_revolutions = false`, `pool = off`, `incumbency = 1`, `birth_links = 0`,
`mend_on_death = false`, `mutate_on_replication = true` (mutation probability 0.2 per copy).

## The sheet (each law: sentence, code, check)
"Holds on all N" means every case in 100 iterations. Code is `GraphOfLife.py` unless said otherwise.

1. **Tokens are conserved.** The sum of all tokens is 10,000 after every phase; every token is staked once in
   the game (`blotto_phase` 2099-2300, `stake_in_rounds` gol_rounds.py 197-262; the cleanup hands the dead's
   tokens out, `_cleanup_and_redistribute` 2428-2574). Holds on all 100 iterations (frames of both phases;
   sum of stakes = tokens at the start of the game; each agent's record equals its tokens).
   Also: after a cleanup nobody holds 0 tokens (min is 1), holds on all 100.
2. **A token moves at most one link per game.** Agent u can stake only on `[u] + neighbours(u)`
   (`targets_of`, gol_rounds.py 209; `place`, 123). Holds on all 417,267 agent records (target list equals
   self + neighbours exactly) and all 691,351 stakes (each is on the agent or across an existing link).
   Moves outside the game: a parent pays a child's endowment (the child need not be linked to it; `_spawn_child`
   1811), gifts and upkeep cross a link that exists (`_choose_gifts` 1863 and the pay loop; `_pay_upkeep` 1927).
   Not frame-checked (the reproduction frame does not record who held what before).
3. **A node keeps every token staked on it, whoever wins; the winner only changes whose brain it carries.**
   `new_tokens[v] = incoming_totals[v]` (line 2240). A survivor's tokens in the frame = stakes on it + estate
   income, and the estate part is never negative: holds on all 100 iterations. Winners exist exactly for the
   nodes with some stake (4,892 of 5,322 in the example frame): holds on all 100. The winner's brain is
   implanted: the node shows the winner's genotype id, or a new mutant id whose parent id is the winner's; a
   node whose owner wins keeps its own brain id exactly: holds on all 295,037 surviving nodes.
4. **Only neighbours can stake on a node.** Same code and checks as law 2 (holds on all 691,351 stakes).
   The plan's exception does not exist here (`link_search = off`; no stake ever came from farther). **Corrected
   on the "core": it is only a safe core within one game.** A node whose neighbours all carry its own genotype
   id (a "core"; about 374 per iteration) is never attacked by another genotype in that game (0 of about 37,400 core-node cases), but
   in the next iteration 61.8% of them (of 20,597 follow-ups) have a neighbour of another genotype id: in 17,173
   cases a newborn linked to it (children link to their parent's neighbours, `_spawn_child` 1811-1850, and are
   mutants 20% of the time), in 4,837 cases a neighbour node was won and implanted with a mutant, in 1,348 the
   node itself changed. So the "stranger" is nearly always a mutant child of kin. Kin can also still take a node
   from kin (lottery).
5. **The contest is a lottery, not "largest wins".** `_lottery` 2359-2371: the winner is drawn from the stakers
   with probability stake / total stake (stakes weighed by `incumbency` = 1, no pools). Checked: the winner is
   one of the stakers on all 377,820 contested-or-uncontested nodes; the recorded `amount` is the largest
   single stake on all 377,820. Chance follows share: over 197,847 (node, staker) pairs on nodes with two or
   more stakers, observed wins vs expected in 10 share bins have chi-square 1.5 on 10 degrees of freedom
   (observed 197,847 wins is exactly expected by construction; the bins are the test). Grouping stakers by
   genotype (kin add up): chi-square 13.9 on 10 df (p about 0.18; the largest bin, share above 0.9, is
   +2.7 sigma, small). The largest single stake loses on average 290.5 of 1,978 contested nodes per
   iteration (14.7%). **"Many small kin stakes do not defend" is therefore false in this world: they add up by
   chance in proportion** (clones' shares add; verified by the kin-group test).
6. **A failed attack feeds the defender.** True and a special case of law 3. Per iteration, on average 513
   surviving nodes were defended by their owner against at least one other staker, which left 859 attackers'
   tokens on them; 1,536 surviving nodes were won by a neighbour (min 659, max 2,363).
7. **Upkeep and the cull.**
   a. A node with no stake on it ends with 0 tokens and is removed (`starved`, 2461): holds on all 100, and the
      cleanup's `starved` count equals the number of unstaked nodes (394 per iteration on average).
   b. **Corrected: a link is cut after the game only if no stake crossed it in the game AND it was not paid for
      this iteration AND it was not made this iteration** (`_stale_edges` 1227-1249, `_add_edge` 1110 stamps new
      links as paid; `_pay_upkeep` 1927). The set of cut links predicted this way equals the recorded
      `pruned_edges` exactly on all 100 iterations (avg 1,570 cut of 10,727 links per iteration). Phase 1 cuts
      nothing (`prune_after = blotto`): holds on all 100. Gifts do not save a link (they are marked in the earlier phase).
   c. Only the largest connected piece survives (`_cleanup_and_redistribute` 2517-2521): after removing the
      starved and the cut links, the survivors equal the largest component on all 100 iterations; the rest are
      "orphaned" (**828 per iteration on average, up to 3,342, more than twice the starved**), and the
      surviving links are exactly the uncut links between survivors (all 100).
8. **What an attacker sees: corrected.** Under `strongest_first` an agent places a token at a time and each
   look reads, besides its own tokens and six quantiles of neighbour tokens and degrees, the at-risk flags,
   the messages and noise, **the board: 24 numbers per candidate** (tokens staked so far, the lead, my stake,
   its own stake, what is still to be placed around it, rivals and so on; `gol_config.BOARD`,
   `Board.read` gol_rounds.py 160-190). The stakes already placed in earlier rounds are seen; stakes still to come
   are not. The richest agents place first (round 1 has only those with most tokens left). Read from code
   (cannot be frame-checked); the plan's "not the stakes about to arrive" is right but incomplete.
9. **Estates (`redistribution = parents`).** All tokens of the nodes removed in a cleanup (starved plus
   orphaned) are pooled and shared among the parents of children born this iteration that are still alive,
   one chance per living child (`_share_out` 2576-2653, 2630-2650); if nobody is an heir, every survivor has an equal chance.
   Not "per head": **per child**. Checked in the game phase: all tokens of the dead (average 1,960 per
   iteration) were shared (equals `cleanup.redistributed`, all 100); every recipient is such a parent (all
   100); the draws follow the weights (chi-square per df = 0.99, range 0.89-1.10). The reproduction phase
   (average 753 per iteration) runs the same code; it was only checked for conservation, not per recipient.

## Derivations with real numbers (100 iterations; averages of per-iteration values)
The world is strongly swinging: nodes at the start of the game range 1,918-6,379, at the end 1,234-4,766.

- **Tokens per agent:** 10,000 / N = 2.63 on average at the start of the game (1.57-5.21) and 3.75 at the end (2.10-8.10).
  Share holding 1 / 2 / 3+ tokens: start of game **43% / 23% / 34%**; end of game 25% / 24% / 51%
  (what an agent stakes is what it holds at the start of the game).
- **Estates:** per iteration 753 (reproduction) + 1,960 (game) = 2,713 tokens, **27% of the world's tokens change
  hands through the estate channel each iteration**. In the game phase: 960 heirs on average (range 382-1,768),
  so 2.04 tokens per heir and 0.66 per survivor; 564 nodes actually receive something. Only parents get it, so
  "income per head" is "income per child born and surviving".
- **Ways to stake** (a token goes to the agent itself or one of its k neighbours; k is 4.96 / 5.16 / 5.34 on average for agents
  with 1 / 2 / 3 tokens, median 5). Distinct allocations (how many tokens on which node) = C(t+k, k):
  1 token, k = 5: 6; 2 tokens: 21 (mean over real agents 25.7); 3 tokens: 56 (mean 100.1); the number of
  token-by-token choice sequences is (k+1)^t = 6 / 36 / 216. 66% of the agents that stake have 1-2 tokens, but
  even 1 token is a real choice among about 6 targets, and under a lottery one token is a real chance.
  **Corrected: "with 1 or 2 tokens there is almost nothing to decide" is not true here**; what is poor is the
  size of a stake, not the number of options. Own node gets 41% of all staked tokens on average (24%-55%).
- **Defence law, as a lottery law:** an owner keeps node v with probability (own stake)/(all stakes), and its
  kin's stakes count with it. "The largest single stake" gives no safety (14.7% of contested nodes go to
  a non-largest staker). Safe by construction only if every staker is kin or no one stakes there.
  With W tokens and N nodes a group can put about (W - N)/B extra on each of B border nodes, but
  each node holds at least one token already (law 7a), so W - N is the free part.
- **Surface law: fails.** Components of one genotype (end of game, 37,725 with at least 2 nodes): the share of
  nodes on the border is 92% for 2 nodes, 87% for 3-5, 82% for 6-11, 79% for 12-29, 79% for 30-99 and 83% for
  100+ nodes (61 territories, the largest 459). Log-log slope of B vs N is 0.98 (a surface would be 0.67 in 3-D). Territories
  are thin and ragged, not blobs; defence per border node does not grow with size.
- **The delay.** A token takes one iteration per link. Mean diameter of a territory (5 nodes or more) is 4.9 links (max 41),
  whole world 30-80 links across. But agents are short-lived: age at the end of the game has median 1
  iteration, 75% <= 2, 90% <= 4, 99% <= 9. Help from farther than about 2-4 links arrives after the node it
  was meant for is gone; only the 1-2 neighbourhood counts.
- **Income per head vs a raid:** see A2. Estates pay about 0.45 tokens per agent-iteration in the game phase alone; strangers' tokens
  staked on a won node (prey) 0.20 per agent-iteration by lineage (A2).
- **Poverty:** with a mean of 2.6 tokens per agent and 43% holding 1, bodies are already poor. See the correction under "Ways to stake".

## Verdict on the plan's list
Holds: 1, 2, 3, 6, 7a, 9 (with "parents" correction). Corrected: 4 (core is safe only inside a game, mutant
kin arrive within one iteration), 5 (lottery, kin stakes add by chance), 7b (links need no stake AND no payment AND not new),
7c (cull kills twice as many as starvation), 8 (the board is also seen), the "income per head" and "poverty"
derivations.

## Caveats
One world and 100 iterations in one boom-and-bust stretch. "Kin" is the same genotype id, which a mutation changes.
Laws about reproduction (law 2's gifts, law 9 in phase 1) are cited from code, not frame-checked.
