# Boundary Conditions #

There are various conditions one can choose from to deal with boundaries when
taking a fft. 
``` @docs

originalSize
effectiveSize
```

## Periodic ##

The simplest to implement, this just assumes that the two edges are adjacent
`boundary=Periodic()`. 

## Pad{N} ##

The typical choice in convolutional neural networks, this assumes that the
signal is zero beyond the values given. Currently, it is up to the user to
figure out what the appropriate padding is; it is at worst the size of the
space domain support of your filter. Suppose you have a 1D filter with 5
adjacent non-zero entries in the space domain, e.g. 
`w = [0..., 0, 5,1,-9,3,2, 0..., 0]`; then the most carry-over that is possible
is `[2, 0..., 0, 5,1,-9,3]`, so if we pad with at least 5 zeros, then no
entries will carry over. To do this, set `boundary = Pad((5,))`. You can choose
larger paddings, of course, but you pay the price of a larger effective signal
size. This can be chosen automatically by setting the padding amount to -1,
e.g. `boundary = Pad((-1,))`.

``` @docs

Pad
```

## Symmetric ##

This is the assumption underlying the DCT type II[^1]
![This is from Wikipedia https://commons.wikimedia.org/wiki/File:DCT-symmetries.svg](https://upload.wikimedia.org/wikipedia/commons/a/ae/DCT-symmetries.svg)

It assumes that the first and last value are repeated. This has two primary
advantages: it makes the boundary continuous, and the derivative is less
discontinuous than it would be with simply repeating the entries (as in the DCT
type I above). The trade-off presently is that the implementation requires
twice the space of the original image, which can be cost prohibitive.

To use this boundary condition, just set `boundary=Sym()`.


[^1]: This image is [from wikipedia](https://commons.wikimedia.org/wiki/File:DCT-symmetries.svg).
