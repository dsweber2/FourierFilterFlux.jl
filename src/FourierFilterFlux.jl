module FourierFilterFlux
using Reexport 
# @reexport using CUDA
using CUDA
using Zygote, Flux, Shearlab, LinearAlgebra
using AbstractFFTs, FFTW #TODO: check the license on FFTW and such
using ContinuousWavelets
using Flux
using Adapt
using RecipesBase
using Base: tail

import Adapt: adapt
export pad, poolSize, originalDomain, params!, formatJLD, getBatchSize
export Periodic, Pad, ConvBoundary, Sym, analytic, outType, nFrames
# layer types and constructors
export ConvFFT, waveletLayer, shearingLayer, averagingLayer
# inits
export positive_glorot_uniform, iden_perturbed_gaussian, 
    uniform_perturbed_gaussian
# Analytic types
export TransformTypes, AnalyticWavelet, RealWaveletRealSignal, 
    RealWaveletComplexSignal, NonAnalyticMatching
# utils
export effectiveSize, originalSize

include("boundaries.jl")

"""
    # preconstructed filters w
    ConvFFT(w::AbstractArray{T,N}, b, originalSize, σ = identity; plan=true, 
    boundary = Periodic(), dType=Float32, trainable=true, OT=Float32, An=nothing) 
        where {T,N}
    # randomly constructed filters
    ConvFFT(k::NTuple{N,Integer}, nOutputChannels = 5,
            σ=identity; nConvDims=2, init = Flux.glorot_normal,
            plan=true, bias = true,
            dType=Float32, OT=Float32, boundary=Periodic(), 
            trainable=true, An=nothing) where N

Similar to `Conv` from Flux.jl, but does pointwise multiplication in the
Fourier domain, with boundary conditions `boundary`, and applies the
nonlinearity `σ`. It is worth noting that the bias is added in the Fourier
domain; if you don't want a bias, set `b=nothing`. For the first method, the
weights `w` should have dimension 1 greater than the size of the convolution,
while for the second, `k` gives the size of the input (including channels and
batch size), for which appropriate weights are generated in the fourier domain
according to the distribution `init`.
# Shared Arguments
- `σ=identity`: a function to apply.
- `dType::DataType=Float32`: the data type being input. By default, it assumes 
                            both the filter and the signal are real, which 
                            allows us to use an rfft and half the (complex)
                            coefficients we would otherwise need.
- `OT::DataType=Float32`: the output datatype, usually determined by the (space
                          domain) type of the filters. Currently assumes that 
                          if this is complex, then the filters are analytic 
                          (so only defined for positive frequencies).
- `plan::Bool=true`: use precomputed fft plan(s), as this is a significant cost.
                     Set this to `false` if you have variable batch/channel sizes.
- `trainable::Bool=true`: The entries are trainable as Flux objects, and so are 
                          returned in a `params` call if this is `true`.
- `boundary::ConvBoundary`: determines how the edges are treated. See e.g. `Pad`.
- `An::Union{Nothing,NTuple{<:Integer}}=nothing`: only used if the output type is 
                    complex, so the filters are complex. In that case, it is a 
                    list of the filters which are actually real. This is 
                    included to allow for averaging wavelets which are real even
                    for analytic wavelets. For both `waveletLayer` and 
                    `shearingLayer`, this is the last filter, so if there are
                    18 total wavelets, `An=(18,)`.
# First constructor only

- `w::AbstractArray{T,N}`: the weights. should be `D+1`, with the last
                           dimension being the total number of filters
` `b::Union{Nothing,AbstractArray}`: if `nothing`, no bias is added. Otherwise it should be (input channels)×(output channels) 
# Second constructor only
- `k::NTuple{N,Integer}`: the dimensions of the input. In the 1D case it should
                          be three entries e.g. `(132,3,100)` which is
                          (signal)×(channels)×(examples). In the 2D case it
                          should be four entries e.g. `(132,132,3,100)`, which
                          is (x)×(y)×(channels)×(examples).
- `nOutputChannels::Integer=5`: the number of filters to use.

- `bias::Bool=true`: determines whether or not to create a bias.
- `init::function=Flux.glorot_normal`: The way to initialize both the bias (if
                    defined) and the weights.  Any function that results in a
                    matrix is allowed, though I would suggest something from
                    Flux, or one of the ones defined in this package.
- `nConvDims::Integer`: the number of dimensions that we will be doing the
                    convolution over, as `k` is somewhat ambiguous.
"""
struct ConvFFT{D, OT, F, A, V, PD, P, T, An}
    σ::F
    weight::A
    bias::V
    bc::PD
    fftPlan::P
    analytic::An
end

function ConvFFT(w::AbstractArray{T,N}, b, originalSize, σ =
                 identity; plan=true, boundary = Periodic(),
                 dType=Float32, trainable=true, OT=Float32, 
                 An=nothing) where {T,N}
    @assert length(originalSize) >= N-1
    if dType <: Complex
        OT = dType
    end

    if length(originalSize) == N-1
        exSz = (originalSize..., 1) # default number of channels is 1
    else
        exSz = originalSize
    end
    netSize,boundary = effectiveSize(exSz[1:N-1],boundary)
    nullEx = Adapt.adapt(typeof(w), zeros(dType, netSize..., exSz[N:end]...))
    convDims = (1:(N-1)...,)

    # Check that they applied the boundary condition, and if not do it ourselves
    if dType <: Complex && size(nullEx,1) != size(w,1) 
        wtmp = ifftshift(ifft(w, convDims),convDims)
        wtmp = applyBC(wtmp, boundary, N-1)
        w = fft(fftshift(wtmp, convDims), convDims)
        @warn("You didn't hand me a set of filters constructed with the boundary in mind. I'm going to adjust them to fit, this may not be what you intended")
    elseif dType <: Real && size(nullEx,1) >>1+1 != size(w,1)
        wtmp = ifftshift(irfft(w, exSz[1], convDims),convDims)
        wtmp,_ = applyBC(wtmp, boundary, N-1)
        w = rfft(fftshift(wtmp,(convDims)),convDims)
        @warn("You didn't hand me a set of filters constructed with the boundary in mind. I'm going to adjust them to fit, this may not be what you intended")
    end

    if  plan && dType <: Real && OT <:Real
        fftPlan = plan_rfft(real.(nullEx), convDims)
    elseif plan && dType <: Real # output is complex, wavelets analytic
        null2 = Adapt.adapt(typeof(w), zeros(dType, netSize..., exSz[N:end]...)) .+
            0im
        fftPlan = (plan_rfft(real.(nullEx), convDims), plan_fft!(null2, convDims))
    elseif plan
        fftPlan = plan_fft!(nullEx, convDims)
    else
        fftPlan = nothing
    end
    if typeof(An) <: Nothing
        An = map(x->NonAnalyticMatching(), (1:size(w)[end]...,))
    end

    return ConvFFT{N-1, OT, typeof(σ), typeof(w), typeof(b), 
                   typeof(boundary), typeof(fftPlan), 
                   trainable, typeof(An)}(σ, w, b, boundary, fftPlan, An)
end


function ConvFFT(k::NTuple{N,Integer}, nOutputChannels = 5,
                 σ=identity; nConvDims=2, init = Flux.glorot_normal,
                 plan=true, bias = true,
                 dType=Float32, OT=Float32, boundary=Periodic(), 
                 trainable=true, An=nothing) where N

    effSize, boundary = effectiveSize(k[1:nConvDims], boundary)
    if dType <: Real && OT <: Real
        effSize = (effSize[1] >>1 + 1, effSize[2:end]...)
    end
    w = init(effSize..., nOutputChannels) .+
        im .* init(effSize..., nOutputChannels)

    if bias == true
        b = init(k[(nConvDims+1):end-1]..., nOutputChannels) 
    else
        b = nothing
    end

    ConvFFT(w, b, k, σ, plan = plan, boundary = boundary, dType = dType,
            trainable=trainable, OT=OT, An=An)
end

function Base.show(io::IO, l::ConvFFT)
    # stored as a brief description
    if typeof(l.fftPlan)<:Tuple 
        sz = l.fftPlan[2]
    else
        sz = l.fftPlan.sz
    end
    es = originalSize(sz[1:ndims(l.weight)-1], l.bc)
    print(io, "ConvFFT[input=($(es), " *
          "nfilters = $(size(l.weight)[end]), " *
          "σ=$(l.σ), " * 
          "bc=$(l.bc)]")
end

function Base.show(io::IO, l::ConvFFT{D, OT, A, B, C, PD, P}) where {D, OT, A, B, C, PD, P<:Tuple}
    if typeof(l.fftPlan[1])<:Tuple
        sz = l.fftPlan[1][2]
    else
        sz = l.fftPlan[1].sz
    end
    es = originalSize(sz[1:ndims(l.weight)-1], l.bc)
    print(io, "ConvFFT[input=($(es), " *
          "nfilters = $(size(l.weight)[end]), " *
          "σ=$(l.σ), " * 
          "bc=$(l.bc)]")
end

analytic(p::ConvFFT) = p.analytic!=nothing

outType(p::ConvFFT{D, OT}) where {D, OT} = OT
nFrames(p::ConvFFT) = size(p.weight)[end]

include("transforms.jl")
include("Utils.jl")
include("convFFTConstructors.jl")
end # module
