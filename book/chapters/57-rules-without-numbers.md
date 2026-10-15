# Rules without numbers

> [!question] Questions of this chapter
> - Which numbers in B2's rules were chosen by hand, and what does each do?
> - What is left of a world when all of them are taken away at once?
> - Which numbers can be replaced by something the world decides for itself,
>   and which are a matter of definition?

## Why ask

The author of the simulation set a rule for everything that follows: no
factor chosen by hand may sit in the rules, except the total of tokens and the
parameters of mutation. Every other number should come out of the world. The
one and a half by which an owner's stake counts was named as the example of
what is not wanted.

B2, the world that cannot break, has seven such numbers. Each was chosen to
stop one way a world was seen to fail, and each was tested in
[Chapters 48](48-a-world-that-cannot-break.md) and
[49](49-who-dies-and-why.md):

| number | in B2 | what it stops | shown in |
|---|---|---|---|
| `incumbency` | 1.5 | an owner's stake counts one and a half times: defending is cheaper than attacking | Chapter 49 |
| `birth_links` | 2 | a newborn is joined to its parent and two of the parent's neighbours: no world grows as a tree | Chapter 48 |
| `prune_needs` | 2 | an unused link lapses only if its ends share two neighbours: pruning never cuts | Chapter 48 |
| `valence` | 8 | a brain may choose at most 8 links: no hubs | Chapter 48 |
| `keep_home` | 0.3 | 30% of an agent's tokens stay at home: no agent stakes itself to death, and no world of one-token zombies | Chapter 49 |
| `max_age` | 100 | every agent dies at 100: no hoarding queen, and room for the young | Chapter 49 |
| `min_child` | 10 | a child starts with at least 10 tokens: children are not spent as missiles | Chapter 49 |

Two more rules of B2 have no number in them: a death is mended by joining the
dead agent's neighbours in a ring, and a brain changes only when it is
copied. They stay. The brain itself has sizes too, one hidden layer of 16 and
messages of 8 numbers. Those are the subject of Chapter 53 and are left
alone here.

Experiments 14 and 15 take six of B2's numbers away **one at a time**, all
but the owner's one and a half, to show what each prevents. This chapter does the opposite. It takes all seven away at once
and asks what a world does with none of them. Then it asks which of them can
be replaced by a rule with no number in it.

## B3: B2 without its numbers

The pilots of this chapter (batch N in `research/oee/pilots/LOG.md`) run B1's
game with every number at its original value. An owner's stake counts once.
A newborn has only the links its brain chooses. Any connection that carries
no tokens lapses. No agent has a cap on its links, none keeps anything at
home by rule, none dies of age, and a child may have a single token. On top
of that they have what B2 has without a number: brains with one hidden layer
of 16 that change only when copied, deaths mended in a ring, and a memory of
the last game's stakes. Revolutions are off. Three variants, two seeds each,
at 10,000 tokens:

- **B3a**: with alliances ([Chapter 56](56-alliances.md));
- **B3b**: with alliances, and a newborn also joined to every neighbour its
  parent was allied with in the last game (`birth_allies`), which builds
  space from the alliances, with no number;
- **B3c**: without alliances.

<!-- turns -->
---

← [Chapter 56 · Alliances](56-alliances.md) · [Contents](../README.md) · [Chapter 59 · Worlds of 100,000 tokens](59-worlds-of-100000-tokens.md) →
<!-- /turns -->
