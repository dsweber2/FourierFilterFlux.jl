module FourierFilterFlux
# TODO: drop useGpu, since we can just pipe through gpu now
using CuArrays
using Zygote, Flux, Shearlab, LinearAlgebra
using AbstractFFTs, FFTW #TODO: check the license on FFTW and such
using Wavelets
using Flux
using Adapt
using RecipesBase
using Base: tail

import Adapt: adapt
export pad, poolSize, originalDomain, params!, adapt, cu, formatJLD, getBatchSize
export Periodic, Pad, ConvBoundary, Sym, analytic, outType, nFrames
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

D gives the dimension, T gives an indication whether it is trainable or not,
and OT gives the output entry type (e.g. if the transform is
real to real, this is <: real, whereas if its analytic this is <:complex)
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
    if length(originalSize) == N-1
        exSz = (originalSize..., 1) # default number of channels is 1
    else
        exSz = originalSize
    end
    netSize,boundary = effectiveSize(exSz[1:N-1],boundary)
    nullEx = Adapt.adapt(typeof(w), zeros(dType, netSize..., exSz[N:end]...))

    convDims = (1:(N-1)...,)

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

    return ConvFFT{N-1, OT, typeof(σ), typeof(w), typeof(b), 
                   typeof(boundary), typeof(fftPlan), 
                   trainable, typeof(An)}(σ, w, b, boundary, fftPlan, An)
end


function ConvFFT(k::NTuple{N,Integer}, nOutputChannels = 5,
                 σ=identity; nConvDims=2, init = Flux.glorot_normal,
                 useGpu=false, plan=true,
                 dType=Float32, OT=Float32, boundary=Periodic(), 
                 trainable=true, An=nothing) where N

    effSize, boundary = effectiveSize(k[1:nConvDims], boundary)
    if dType <: Real
        effSize = (effSize[1] >>1 + 1, effSize[2:end]...)
    end
    w = init(effSize..., nOutputChannels)
    b = init(k[(nConvDims+1):end-1]..., nOutputChannels)
    if useGpu
        w = cu(w)
        b = cu(b)
    end
    ConvFFT(w, b, k, σ, plan = plan, boundary = boundary, dType = dType,
            trainable=trainable, OT=OT)
end

function Base.show(io::IO, l::ConvFFT{<:Any, <:Real})
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

function Base.show(io::IO, l::ConvFFT{<:Any, <:Complex})
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
