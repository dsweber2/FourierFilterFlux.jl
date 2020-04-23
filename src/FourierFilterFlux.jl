module FourierFilterFlux

using CuArrays
using Zygote, Flux, Shearlab, LinearAlgebra
using AbstractFFTs
using Wavelets
using Flux
using Adapt
using RecipesBase
using Base: tail

import Adapt: adapt
export pad, poolSize, RationPool, batchOff, originalDomain, params!, adapt
# layer types
export ConvFFT, waveletLayer, shearingLayer, averagingLayer
# inits
export positive_glorot_uniform, iden_perturbed_gaussian, uniform_perturbed_gaussian
export buildRecord, expandRecord!, addCurrent!

include("boundaries.jl")

"""
    ConvFFT(w::AbstractArray{T,N}, b, originalSize, σ =
                 identity; plan=true, padBy = nothing,
                 dType=Float32, trainable=true) where {T,N}

    ConvFFT(k::NTuple{N,Integer}, nOutputChannels = 5,
                 σ=identity; nConvDims=2, init = Flux.glorot_normal,
                 useGpu=false, plan=true,
                 dType=Float32, padBy=nothing, trainable=true) where N

Similar to `Conv`, but does pointwise multiplication in the Fourier domain, and
thus a circular convolution. By default, it assumes both the filter and the
signal are real, which allows us to use an rfft and half the (complex)
coefficients we would otherwise need. By default, it assumes a fixed batch size
(the last entry of k) so that it can create a single fftPlan and reuse it,
rather than replanning every time. By default, it is picked up by params, but
this can be modified if desired.

D gives the dimension, T gives the output entry type (e.g. if the transform is
real to real, this is <: real, whereas if its analytic this is <:complex)
"""
struct ConvFFT{D, F, A, V, PD, P, T}
    σ::F
    weight::A
    bias::V
    BoundaryCondition::PD
    fftPlan::P
end

function ConvFFT(w::AbstractArray{T,N}, b, originalSize, σ =
                 identity; plan=true, boundary = Periodic(),
                 dType=Float32, trainable=true) where {T,N}
    @assert length(originalSize) >= N-1
    if length(originalSize) == N
        exSz = (originalSize..., 1) # default number of channels is 1
    else
        exSz = originalSize
    end
    if padBy !=nothing
        netSize = exSz[1:N-1] .+ (2 .* padBy)
    end
    nullEx = Adapt.adapt(typeof(w), zeros(dType, netSize..., exSz[N:end]...))

    convDims = (1:(N-1)...,)

    if  plan && dType <: Real
        fftPlan = plan_rfft(nullEx, convDims)
    elseif plan
        fftPlan = plan_fft!(nullEx, convDims)
    else
        fftPlan = nothing
    end

    return ConvFFT{N-1, typeof(σ), typeof(w), typeof(b), 
                   typeof(boundary),
                   typeof(fftPlan), trainable}(σ, w, b, boundary, fftPlan)
end

function ConvFFT(k::NTuple{N,Integer}, nOutputChannels = 5,
                 σ=identity; nConvDims=2, init = Flux.glorot_normal,
                 useGpu=false, plan=true,
                 dType=Float32, boundary=nothing, trainable=true) where N

    if boundary<:Pad
        boundary = Pad{nConvDims}(([k[i]>>2+1 for i=1:nConvDims]..., ))
        effSize = k[1:nConvDims] .+ 2 .* padBy
        effSize = (effSize[1]>>1+1, effSize[2:end]...)
    end
    w = init(effSize..., nOutputChannels)
    b = init(k[(nConvDims+1):end-1]..., nOutputChannels)
    if useGpu
        w = cu(w)
        b = cu(b)
    end
    ConvFFT(w, b, k, σ, plan = plan, boundary = boundary, dType = dType,
            trainable=trainable)
end

function Base.show(io::IO, l::ConvFFT)
    sz = l.fftPlan.sz
    if typeof(l.boundary) <:Pad
        effectiveSize = ([sz[1]-2*p for p in l.boundary.padBy]...,)
    else
        effectiveSize = sz
    end
    print(io, "ConvFFT[input=($(effectiveSize), " *
          "nfilters = $(size(l.weight)[end]), " *
          "σ=$(l.σ)]")
end

import Flux.params!
function params!(p::Params, x::ConvFFT{A, B, C, D, E, F, false}, seen =
                 IdSet()) where {A,B,C,D,E,F}
    return
end

function params!(p::Params, x::ConvFFT{A, B, C, D, E, F, true}, seen =
    IdSet()) where {A,B,C,D,E,F}
    params!(p, x.weight, seen)
    params!(p, x.bias, seen)
end

include("transforms.jl")

end # module
