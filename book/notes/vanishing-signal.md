# How a signal fades through layers

A brain's inputs reach its outputs only by passing through every layer in
between. Each layer can let a change through larger or smaller. When every
layer lets it through smaller, a deep network ends up nearly blind. Its
outputs barely move, whatever its inputs do. This is called the **vanishing
signal** (in training by gradients, the *vanishing gradient*: Hochreiter 1991;
Glorot and Bengio 2010).

## The gain of one layer

Take one hidden layer of Graph of Life's brain ([Chapter 4](../chapters/04-the-brain.md)):

$$
a = \sigma(W x + b), \qquad \sigma(z) = \frac{1}{1 + e^{-z}} .
$$

Move the input *x* by a small amount δ*x*. Each sum *z* moves by *W* δ*x*,
and each output by the slope of the sigmoid at that sum times that:

$$
\delta a_i = \sigma'(z_i) \, (W \delta x)_i , \qquad \sigma'(z) = \sigma(z)\,\bigl(1 - \sigma(z)\bigr) \le \tfrac14 .
$$

The slope of the sigmoid is at most ¼, at *z* = 0, and falls off on both
sides: it is 0.10 at *z* = ±2 and 0.018 at *z* = ±4.

With weights drawn as a founder's are, normal with standard deviation
1/√*k* for a fan-in of *k*, the product *W* δ*x* is about as large as δ*x*
itself. That is what the scale 1/√*k* is chosen for. So one sigmoid layer
passes a change on at most about **a quarter** as large. This ratio is the
layer's **gain**:

$$
g = \frac{\lVert \delta a \rVert / \sqrt{\text{outputs}}}{\lVert \delta x \rVert / \sqrt{\text{inputs}}} ,
$$

the root-mean-square change out over the root-mean-square change in.

## Layers multiply

Gains multiply through the layers. Five sigmoid layers, each passing about a
quarter, pass

$$
\left(\tfrac14\right)^5 \approx 0.001 .
$$

A change of 1 in every input moves the outputs by about a thousandth. What
the outputs are then is set almost entirely by the brain's biases and by the
average activity of its layers, not by what it sees. The brain still computes
something. It just computes nearly the same thing for every input.

## Why this matters for evolution

Selection can only favour a brain that reacts to its situation if its
reactions differ enough to matter. Two candidates whose scores differ by a
thousandth of a token's worth are, for every decision of
[Chapter 3](../chapters/03-one-iteration.md), the same candidate. A mutation
that makes a brain attend to a neighbour's wealth has almost no effect to be
selected for. Only mutations that move what the brain does *whatever it sees*
have any.

## The usual remedies

Deep learning met this problem when networks first grew deep, and found
three remedies:

- **Fewer layers.** Each layer removed multiplies the gain by about four.
- **A different squashing.** tanh has slope 1 at 0, four times the sigmoid's.
  A *rectifier* (ReLU), max(0, *z*), has slope 1 wherever it is active (Nair
  and Hinton 2010).
- **Larger starting weights**, scaled to the squashing so that each layer's
  gain starts near 1 (Glorot and Bengio 2010): weights of standard deviation
  √(2/*k*) for rectifiers (He et al. 2015), and for sigmoids about four times
  what suits tanh, since the sigmoid's slope is a quarter of tanh's.

[Chapter 18](../chapters/18-what-the-brains-are-like.md) measures the gain of
Graph of Life's brains. Chapter 43 (planned, see [the contents](../README.md))
is to try the remedies.

## References

- Glorot, X. and Bengio, Y. (2010). Understanding the difficulty of training
  deep feedforward neural networks. *Proceedings of AISTATS*, 249–256.
- He, K., Zhang, X., Ren, S. and Sun, J. (2015). Delving deep into
  rectifiers. *Proceedings of ICCV*, 1026–1034.
- Hochreiter, S. (1991). *Untersuchungen zu dynamischen neuronalen Netzen*.
  Diploma thesis, Technische Universität München.
- Nair, V. and Hinton, G. E. (2010). Rectified linear units improve
  restricted Boltzmann machines. *Proceedings of ICML*, 807–814.
