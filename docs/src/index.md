# FourierFilterFlux Documentation #
A package to convolve with your favorite filters in the frequency domain in
1,2, or 3 dimensions, and you are looking for them to be differentiable and/or
computed using a gpu. This is most useful if your filters are reasonably large
(more than around 100 entries total). Currently used in ScatteringTransform.jl
and collatingTransform.jl
The core is the [ConvFFT type](@ref), for which there are a couple of [Built-in
constructors](@ref).
```@contents
```
## Index
```@index
```
