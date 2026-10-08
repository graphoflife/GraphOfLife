# Reproduction statistics

Every reproduction phase records every birth: the parent, the tokens it held,
the tokens it gave, the candidates the child was joined to, and the
connections the parent handed over ([Chapter 3](../chapters/03-one-iteration.md)).
The statistics of a phase-1 row summarise them.

## The fields

| field | definition |
|---|---|
| `births` | the number of children born |
| `meanInvestedShare` | the mean, over the parents, of tokens given ÷ tokens held |
| `reproTokenShare` | all tokens given to children ÷ all tokens in the world |
| `meanChildLinks` | the mean number of candidates a child was joined to |
| `handovers` | the number of connections parents handed to children |
| `gifts`, `giftTokens`, `giftShare` | gifts of tokens — absent, since gifting is off in every run of this book |

## Two shares that answer different questions

`meanInvestedShare` is **per parent**: how much of itself a typical parent
gives. `reproTokenShare` is **per world**: how much of all the tokens went
into children. A world in which a few poor agents each give everything they
have has a high per-parent share and a tiny world share.

**Example.** Three parents holding 2, 4 and 100 tokens give 1, 2 and 10:
`meanInvestedShare` = (½ + ½ + ⅒)/3 ≈ 0.37, and in a world of 10,000 tokens
`reproTokenShare` = 13/10,000 = 0.0013.

## Links of a child

A child's links are chosen among its parent's candidates, **the parent
included**, so `meanChildLinks` counts the link to the parent too, if there is
one. A child with no link at all is not part of the network and dies in the
cleanup that ends the phase ([Births and the four ways to die](births-and-deaths.md)).

`handovers` counts connections that moved from parent to child: the parent
loses them, the child gains them. They never add a connection to the world.

<!-- turns -->
[Contents](../README.md)
<!-- /turns -->
