```@setup 1DconvEx
using Plots; gr()
Plots.reset_defaults()
```
# ConvFFT type #
The core type of this package. As a simple example in 1D:
```@example 1DconvEx
using FourierFilterFlux, Plots, FFTW, Flux, LinearAlgebra
w = zeros(128,2); w[25:30,1] = randn(6)
w[55:60,2] = randn(6)
ŵ = rfft(w, 1)
W = ConvFFT(ŵ,nothing,(128,1,2))
plot(W, title="Time domain Filters", dispReal=true,
     apply=identity, vis=1:2);
savefig("randFilters.svg")#hide
```
![](randFilters.svg)

So we've created two filters with small support and displayed them in the
time domain (note that this has been done with a [Plot Recipe](@ref)). Applying
this to a signal `x`, which is non-zero at only the two boundary locations:
``` @repl 1DconvEx
x = zeros(128,1,2);
x[1,1,2] = 1; x[end,1,2] = -1; # positive on one side and negative on the other
r = W(x); size(r)
plot(r[:,:,1,2],title="convolved with x"); #hide
savefig("convolveWx.svg") #hide
```
![](convolveWx.svg)

Note that `x` has two extra dimensions; the second is the number of input
channels and the third is the number of examples. The output `r=W(x)` has three
extra dimensions, with the new dimension (the second one) varying over the
filter. Because the default setting assumes periodic boundaries, we get a
bleed-over, causing what should be exact replicas of the filters to give the
difference between adjacent values. We can change the boundary conditions to
address this, e.g. `Pad(6)`
``` @example 1DconvEx
W = ConvFFT(ŵ,nothing,(128,1,2),boundary= Pad(6))
r =W(x); size(r)
plot(r[:,:,1,2],title="convolved with x"); #hide
savefig("convolveWxPad.svg") #hide
```
![](convolveWxPad.svg)

and now we have a positive copy and a negative copy of each filter. For more
details, see the [Boundary Conditions](@ref) section. If you want to start with
some random filter, rather than constructing your own, there's a simple way to
do that as well:
``` @repl 1DconvEx
using FourierFilterFlux,Flux #hide
ex2Dsize = (127, 352, 1, 10);
filt = ConvFFT(ex2Dsize, 3, relu, trainable=true,bias=false)
plot(heatmap(filt,dispReal=true,vis=1,colorbar=false,title="Filter 1"),
	heatmap(filt,dispReal=true,vis=2,colorbar=false,title="Filter 2"),
	heatmap(filt,dispReal=true,vis=3,colorbar=false, title="Filter 3"),
	layout=(1,3));
savefig("2Dfilts.svg") #hide
```
![](2Dfilts.svg)

Here we've created a set of real weights represented in the Fourier domain. Now 
let's see if these filters can be fit to various lowpass filters:
``` @example 1DconvEx
using FourierFilterFlux,Flux #hide
fitThis = zeros(64,352,3); fitThis[1:3,(176-3:176+3),1] .= 1;
fitThis[1:15,(176-15:176+15),2] .= 1; fitThis[1:32,(176-44:176+44),3] .= 1;
targetConv = ConvFFT(fitThis, nothing, (127,352,1,10)) 
loss(x,y) = norm(filt(x) - targetConv(x))
genEx(n) = [(cpu(randn(ex2Dsize)), true) for i=1:n];
Flux.train!(loss, params(filt), genEx(100), ADAM())
plot(heatmap(filt,vis=1), heatmap(filt,vis=2), heatmap(filt,vis=3),
	heatmap(targetConv,vis=1), heatmap(targetConv,vis=2),
	heatmap(targetConv,vis=3));
savefig("fitting2Dfilts.svg") #hide
```
![](fitting2Dfilts.svg)

The top three are the fit filters, while the bottom three are the targets. So
after 100 examples, we have something that passibly resembles the desired
low-pass filters, though not yet matching the norm. Finally, if you would
rather be doing your computations on the gpu, simply use the `gpu` function of
Flux.jl or `cu` of CUDA.jl:

``` @repl 1DconvEx
Wgpu = W |> gpu
x = x |> gpu;
Wgpu(x)[:,1,1,1]'
```

```@docs

ConvFFT
```

## Plot Recipe ##
To ease displaying the weights, we have a
[recipe](https://docs.juliaplots.org/latest/recipes/) that can be used with
either `plot` or `heatmap`. First, we will generate some preconstructed 1D and
2D filters to demonstrate with:
``` @example 1DconvEx
W1 = waveletLayer((256,1,5))
W2 = shearingLayer((256,256));
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
``` @example 1DconvEx
plot(heatmap(W2,title="first shearlet"), heatmap(W2,vis=4,title="Fourth shearlet"));
savefig("demoVis.svg") #hide
plot(W1,vis=:,title="All of the Morlet Wavelets");
savefig("demoVis1D.svg") #hide
```
  ![](demoVis.svg)
  ![](demoVis1D.svg)
  
  
- `dispReal::Bool=false`: takes care of doing the correct type of fft to get
  the space domain filters centered; usually if this is used you also want to
  change
- `apply::Function=abs`: from `abs` to the identity; the default of `abs` makes
  the most sense in the frequency domain. You probably also want to change
- `restrict=nothing`: to show the plot only around the support of the filter:
``` @example 1DconvEx
plot(heatmap(W2, dispReal=true,title= "first wavelet space domain"),
     heatmap(W2, dispReal=true, restrict= (255:345,260:350), apply=identity,
	         title= "first wavelet space domain \nzoomed without abs"));
savefig("zooming.svg") #hide
```
  ![](zooming.svg)

