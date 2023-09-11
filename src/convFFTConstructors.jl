@doc """
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

- `averagingLayer::Bool=false`: if true, use just the averaging filter, and drop all other filters.

# Arguments from ContinuousWavelets.jl

+ `scalingFactor`, `s`, or `Q::Real=8.0`: the number of wavelets between the octaves ``2^J`` and ``2^{J+1}`` (defaults to 8, which is most appropriate for music and other audio). Valid range is ``(0,\\infty)``.
+ `β::Real=4.0`: As using exactly `Q` wavelets per octave leads to excessively many low-frequency wavelets, `β` varies the number of wavelets per octave, with larger values of `β` corresponding to fewer low frequency wavelets(see [Wavelet Frequency Spacing](https://dsweber2.github.io/ContinuousWavelets.jl/dev/spacing/#Wavelet-Frequency-Spacing) for details).
  Valid range is ``(1,\\infty)``, though around `β=6` the spacing is approximately linear *in frequency*, rather than log-frequency, and begins to become concave after that.
+ `boundary::WaveletBoundary=SymBoundary()`: The default boundary condition is `SymBoundary()`, implemented by appending a flipped version of the vector at the end to eliminate edge discontinuities. See [Boundary Conditions](https://dsweber2.github.io/ContinuousWavelets.jl/dev/bound/#Boundary-Conditions) for the other possibilities.
+ `averagingType::Average=Father()`: determines whether or not to include the averaging function, and if so, what kind of averaging. The options are
  - `Father`: use the averaging function that corresponds to the mother Wavelet.
  - `Dirac`: use the sinc function with the appropriate width.
  - `NoAve`: don't average. this has one fewer filters than the other `averagingTypes`
+ `averagingLength::Int=4`:  the number of wavelet octaves that are covered by the averaging,
+ `frameBound::Real=1`: gives the total norm of the whole collection, corresponding to the upper frame bound; if you don't want to impose a particular bound, set `frameBound<0`.
+ `normalization` or `p::Real=Inf`: the p-norm preserved as the scale changes, so if we're scaling by ``s``, `normalization` has value `p`, and the mother wavelet is ``\\psi``, then the resulting wavelet is ``s^{1/p}\\psi(^{t}/_{s})``.
  The default scaling, `Inf` gives all the same maximum value in the frequency domain.
  Valid range is ``(0,\\infty]``, though ``p<1`` isn't actually preserving a norm.
"""
function waveletLayer(inputSize::Union{T,NTuple{N,T}};
    dType = Float32,
    σ = identity,
    trainable = false,
    plan = true,
    init = Flux.glorot_normal,
    bias = false,
    convBoundary = Sym(),
    cw = Morlet(),
    averagingLayer = false,
    varargs...) where {N,T<:Int}
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
