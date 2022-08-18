# ConvFFT type

The core type of this package. As a simple example in 1D:

```jldoctest 1DconvEx
julia> using FourierFilterFlux, Plots, FFTW, Flux, LinearAlgebra, Random

julia> Random.seed!(1234);

julia> w = zeros(128,2); w[25:30,1] = randn(6);

julia> w[55:60,2] = randn(6);

julia> ŵ = rfft(w, 1);

julia> W = ConvFFT(ŵ,nothing,(128,1,2))
ConvFFT[input=((128,), nfilters = 2, σ=identity, bc=Periodic()]

```

```@example 1DconvEx
using FourierFilterFlux, Plots, FFTW, Flux, LinearAlgebra, Random # hide
Random.seed!(1234) # hide
w = zeros(128,2); w[25:30,1] = randn(6)  # hide
w[55:60,2] = randn(6) # hide
ŵ = rfft(w, 1) # hide
W = ConvFFT(ŵ,nothing,(128,1,2))  # hide
plot(W, title="Time domain Filters", dispReal=true, apply=identity, vis=1:2)
```

So we've created two filters with small support and displayed them in the
time domain (note that this has been done with a [Plot Recipe](@ref)). Applying
this to a signal `x`, which is non-zero at only the two boundary locations:

```jldoctest 1DconvEx
julia> x = zeros(128,1,2);

julia> x[105,1,2] = 1; x[end,1,2] = -1; # positive on one side and negative on the other

julia> r = W(x)
128×2×1×2 Array{Float64, 4}:
[:, :, 1, 1] =
 0.0  0.0
 0.0  0.0
 ⋮
 0.0  0.0
 0.0  0.0

[:, :, 1, 2] =
 -0.359729     -2.08167e-17
  1.08721      -1.38778e-17
  ⋮
  1.27045e-17  -1.249e-16
 -9.26904e-17  -2.77556e-17
```

```@example 1DconvEx
x = zeros(128,1,2); # hide
x[105,1,2] = 1; x[end,1,2] = -1; # hide
r = W(x) # hide
plot(r[:,:,1,2],title="convolved with x")
```

Note that `x` has two extra dimensions; the second is the number of input
channels and the third is the number of examples. The output `r=W(x)` has three
extra dimensions, with the new dimension (the second one) varying over the
filter. Because the default setting assumes periodic boundaries, we get a
bleed-over, causing what should be exact replicas of the filters to give the
difference between adjacent values. We can change the boundary conditions to
address this, e.g. `Pad(6)`

```jldoctest 1DconvEx
julia> W = ConvFFT(ŵ,nothing,(128,1,2),boundary= FourierFilterFlux.Pad(6))
┌ Warning: You didn't hand me a set of filters constructed with the boundary in mind. I'm going to adjust them to fit, this may not be what you intended
└ @ FourierFilterFlux ~/allHail/projects/FourierFilterFlux/src/FourierFilterFlux.jl:148
ConvFFT[input=((128,), nfilters = 2, σ=identity, bc=Pad(6,)]

julia> r =W(x); size(r)
(128, 2, 1, 2)
```

```@example 1DconvEx
W = ConvFFT(ŵ, nothing,(128,1,2),boundary= FourierFilterFlux.Pad(6)) # hide
plot(r[:,:,1,2],title="convolved with x")
```

and now we have a positive copy and a negative copy of each filter. For more
details, see the [Boundary Conditions](@ref) section. If you want to start with
some random filter, rather than constructing your own, there's a simple way to
do that as well:

```jldoctest 1DconvEx
julia> ex2Dsize = (127, 352, 1, 10);

julia> filt = ConvFFT(ex2Dsize, 3, relu, trainable=true,bias=false)
ConvFFT[input=((127, 352), nfilters = 3, σ=relu, bc=Periodic()]

```

```@example 1DconvEx
using FourierFilterFlux,Flux, Plots #hide
ex2Dsize = (127, 352, 1, 10); # hide
filt = ConvFFT(ex2Dsize, 3, relu, trainable=true,bias=false) # hide
plot(heatmap(filt,dispReal=true,vis=1,colorbar=false,title="Filter 1"),
	heatmap(filt,dispReal=true,vis=2,colorbar=false,title="Filter 2"),
	heatmap(filt,dispReal=true,vis=3,colorbar=false, title="Filter 3"),
	layout=(1,3))
```

Here we've created a set of real weights represented in the Fourier domain. Now
let's see if these filters can be fit to various lowpass filters:

```jldoctest 1DconvEx
julia> using LinearAlgebra

julia> fitThis = zeros(64,352,3); fitThis[1:3,(176-3:176+3),1] .= 1;

julia> fitThis[1:15,(176-15:176+15),2] .= 1; fitThis[1:32,(176-44:176+44),3] .= 1;

julia> targetConv = ConvFFT(fitThis, nothing, (127,352,1,10))
ConvFFT[input=((127, 352), nfilters = 3, σ=identity, bc=Periodic()]

julia> sum(norm.(map(-, targetConv.weight, filt.weight)) .^ 2) / sum(norm.(targetConv.weight) .^ 2) # the relative error
1.0038243395792357

julia> loss(x,y) = norm(filt(x) - targetConv(x))
loss (generic function with 1 method)

julia> genEx(n) = [(cpu(randn(ex2Dsize)), true) for i=1:n];

julia> Flux.train!(loss, Flux.params(filt), genEx(100), ADAM()) # train for 100 samples

julia> sum(norm.(map(-, targetConv.weight, filt.weight)) .^ 2) / sum(norm.(targetConv.weight) .^ 2) # the relative error
0.8206548516626784

```

```@example 1DconvEx
using FourierFilterFlux, Flux, LinearAlgebra # hide
fitThis = zeros(64,352,3); fitThis[1:3,(176-3:176+3),1] .= 1; # hide
fitThis[1:15,(176-15:176+15),2] .= 1; fitThis[1:32,(176-44:176+44),3] .= 1; # hide
targetConv = ConvFFT(fitThis, nothing, (127,352,1,10)) # hide
loss(x,y) = norm(filt(x) - targetConv(x)) # hide
genEx(n) = [(cpu(randn(ex2Dsize)), true) for i=1:n]; # hide
Flux.train!(loss, Flux.params(filt), genEx(100), ADAM()) # hide
loss(randn(ex2Dsize), nothing) # hide
plot(heatmap(filt,vis=1), heatmap(filt,vis=2), heatmap(filt,vis=3),
	heatmap(targetConv,vis=1), heatmap(targetConv,vis=2),
	heatmap(targetConv,vis=3))
```

The top three are the fit filters, while the bottom three are the targets.
So after 100 examples, we have something that passibly resembles the desired low-pass filters, though the relative error is still around 80%.

Finally, if you would rather be doing your computations on the gpu, simply use the `gpu` function of Flux.jl or `cu` of CUDA.jl:

```@repl 1DconvEx
Wgpu = W |> gpu
x = x |> gpu;
Wgpu(x)[:,1,1,1]'
```

```@docs

ConvFFT
```

## Plot Recipe

To ease displaying the weights, we have a
[recipe](https://docs.juliaplots.org/latest/recipes/) that can be used with
either `plot` or `heatmap`. First, we will generate some preconstructed 1D and
2D filters to demonstrate with:

```@example 1DconvEx
W1 = waveletLayer((256,1,5))
W2 = ConvFFT((256,256),12,boundary=Sym());
```

The general form of the plot is

```
plot(cv::ConvFFT{N}; vis=1, dispReal=false, apply=abs, restrict=(Colon(), vis)) where {N}

heatmap(cv::ConvFFT{N}; vis=1, dispReal=false, apply=abs, restrict=(Colon(), vis)) where {N}
```

The arguments are

- `vis=1`: a structure identifying which filters you want to display. The
  default value is just the first one (displaying them one at a time makes more
  sense in the 2D case). It is worth noting that because these are created
  using a rfft, this is only the postive frequencies in the first dimension
  (vertical in the 2D shearlet plots and horizontal in the 1D plot).

```@example 1DconvEx
plot(heatmap(W2,title="first filter"), heatmap(W2,vis=4,title="Fourth filter"))
```

```@example 1DconvEx
plot(W1,vis=:,title="All of the Morlet Wavelets")
```

- `dispReal::Bool=false`: takes care of doing the correct type of fft to get
  the space domain filters centered; usually if this is used you also want to
  change
- `apply::Function=abs`: from `abs` to the identity; the default of `abs` makes
  the most sense in the frequency domain. You probably also want to change
- `restrict=nothing`: to show the plot only around the support of the filter:

```@example 1DconvEx
plot(heatmap(W2, dispReal=true,title= "first wavelet space domain"),
     heatmap(W2, dispReal=true, restrict= (255:345,260:350), apply=identity,
	         title= "first wavelet space domain \nzoomed without abs"))
```
