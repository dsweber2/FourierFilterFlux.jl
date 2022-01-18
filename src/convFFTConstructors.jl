"""
    waveletLayer(inputSize::Union{Int,NTuple{N, T}};
                 dType = Float32, σ = identity, trainable = false,
                 plan = true, init = Flux.glorot_normal, bias=false,
                 convBoundary=Sym(), cw = Morlet(), averagingLayer = false,
                 varargs...) where {N,T}

Create a ConvFFT layer that uses wavelets from [ContinuousWavelets.jl](https://github.com/dsweber2/ContinuousWavelets.jl). By default it isn't trainable. varargs are any of the settings that can be passed on to creating a `CFW` type.
# New Arguments
- `convBoundary::ConvBoundary=Sym()`: the type of symmetry to use in computing
                                      the transform. Note that convBoundary and
                                      boundary are different, with boundary
                                      needing to be set using types from
                                      ContinuousWavelets and convBoundary needs
                                      to be set using the FourierFilterFlux
                                      boundary types.
- `cw::ContWaveClass=Morlet()`: the type of wavelet to use, e.g. `dog2`,
                                `Morlet()`. See ContinuousWavelets.jl for more.

- `averagingLayer::Bool=false`: the same idea as for shearingLayer, if this is
                                true, only return the results from the
                                averaging filter.
arguments from wavelet constructor
Q=8, boundary::T=SymBoundary(),
        averagingType::A = Father(),
        averagingLength::Int = 4, frameBound=1, p::N=Inf,
        β=4
"""
function waveletLayer(inputSize::Union{T,NTuple{N,T}}; dType = Float32, σ = identity, trainable = false, plan = true, init = Flux.glorot_normal, bias = false, convBoundary = Sym(), cw = Morlet(), averagingLayer = false, varargs...) where {N,T<:Int}
    waveletType = wavelet(cw; varargs...)
    wavelets, _ = computeWavelets(inputSize[1], waveletType; T = T)

    if averagingLayer
        wavelets = wavelets[:, 1:1]
    elseif typeof(waveletType.averagingType) <: Union{ContinuousWavelets.Dirac,Father}
        wavelets = cat(wavelets[:, 2:end], wavelets[:, 1], dims = 2)
    end

    if dType <: Real
        wavelets = Complex{dType}.(wavelets) # correct the element type
    else
        wavelets = dType.(wavelets)
    end

    if typeof(cw) <: Dog || typeof(cw) <: ContOrtho
        OT = dType
        An = nothing
    else
        if typeof(waveletType.averagingType) <: NoAve
            An = (-1,)
        else
            An = (size(wavelets, 2),)
        end
        if dType <: Real
            OT = Complex{dType}
            averagingStyle = RealWaveletRealSignal
        else
            OT = dType
            averagingStyle = RealWaveletComplexSignal
        end
        An = map(ii -> ((ii in An) ? averagingStyle() :
                        AnalyticWavelet()), (1:size(wavelets, 2)[end]...,))
    end
    if bias
        bias = dType.(init(inputSize[2:end-1]..., size(wavelets, 2)))
    else
        bias = nothing
    end

    return ConvFFT(wavelets, bias, inputSize, σ, plan = plan, dType = dType,
        trainable = trainable, boundary = convBoundary, OT = OT, An = An)
end
