############################ 2D methods ###############################


"""
shearingLayer(inputSize::Union{Int,NTuple{N, T}}; 
                       scale = -1, shearLevel = scale, 
                       dType = Float32, σ = abs, trainable = false,
                       plan = true, boundary=Pad(-1,-1), 
                       averagingLayer = false) where {N,T}

create a ConvFFT layer that uses shearlets. By default it isn't trainable

if averagingLayer is true, create a ConvFFT layer that uses just the single averaging shearlet. By default it isn't trainable.
"""
function shearingLayer(inputSize::Union{Int,NTuple{N, T}}; 
                       scale = -1, shearLevel = scale, 
                       dType = Float32, σ = abs, trainable = false,
                       plan = true, boundary=Pad(-1,-1), averagingLayer = false) where {N,T}
    scale = defaultShearletScale(inputSize, scale)
    if shearLevel < 0
        shearLevel = scale
    end
    effectiveShears = ceil.(Int, (1:shearLevel)/2)
    shears = Shearlab.getshearletsystem2D(inputSize[1:2]..., scale,
                                          effectiveShears, 
                                          typeBecomes = dType, 
                                          padded=typeof(boundary)<:Pad)
    if averagingLayer
        shearlets = shears.shearlets[:,:, end:end]
    else
        shearlets = shears.shearlets
    end
    if dType<:Real
        shearlets = Complex{dType}.(shearlets)# correct the element type
    else
        shearlets = dType.(shearlets)
    end
    nShears = shears.nShearlets
    if typeof(boundary) <:Pad
        boundary = Pad{2}(shears.padBy)
    end
    bias = nothing

    return ConvFFT(shearlets, bias, inputSize, σ, plan=plan, 
                   boundary = boundary, dType = dType,
                   trainable=trainable, OT=dType)
end

function defaultShearletScale(inputSize, scale)
    if minimum(inputSize[1:2])>=24 && scale >0
        return scale
    elseif minimum(inputSize[1:2])>=24
        return 4
    elseif minimum(inputSize[1:2])>=12
        return 2
    else
        error("can't meaningfully make shearlets at size $(inputSize[1:2])")
    end
end



#######################################################################
#######################################################################
############################ 1D methods ###############################
#######################################################################
#######################################################################

"""
    waveletLayer(inputSize::Union{Int,NTuple{N, T}}; 
                 dType = Float32, σ = identity, trainable = false,
                 plan = true, init = Flux.glorot_normal, bias=false,
                 convBoundary=Sym(), cw = Morlet(), averagingLayer = false,
                 varargs...) where {N,T}

create a ConvFFT layer that uses wavelets from Wavelets.jl. By default it isn't trainable. varargs are any of the settings that can be passed on to creating a `CFW` type.

Note that convBoundary and boundary are different, with boundary needing to be set using types from ContinuousWavelets and convBoundary needs to be set using the FourierFilterFlux boundary types.

"""
function waveletLayer(inputSize::Union{Int,NTuple{N, T}}; 
                      dType = Float32, σ = identity, trainable = false,
                      plan = true, init = Flux.glorot_normal, bias=false,
                      convBoundary=Sym(), cw = Morlet(), averagingLayer = false,
                      varargs...) where {N,T} 
    waveletType = wavelet(cw; varargs...)
    wavelets,ω = computeWavelets(inputSize[1], waveletType; T=T)

    if averagingLayer
        wavelets = wavelets[:, 1:1]
    elseif typeof(waveletType.averagingType) <: Union{ContinuousWavelets.Dirac, Father}
        wavelets = cat(wavelets[:, 2:end], wavelets[:,1], dims=2)
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
            An = (size(wavelets,2),)
        end
        if dType <: Real
            OT = Complex{dType}
            averagingStyle = RealWaveletRealSignal
        else
            OT = dType
            averagingStyle = RealWaveletComplexSignal
        end
        An = map(ii->((ii in An) ? averagingStyle() :
                      AnalyticWavelet()), (1:size(wavelets,2)[end]...,)) 
    end
    if bias
        bias = dType.(init(inputSize[2:end-1]..., size(wavelets,2)))
    else
        bias = nothing
    end

    return ConvFFT(wavelets, bias, inputSize, σ, plan=plan, dType = dType,
                   trainable=trainable, boundary = convBoundary, OT=OT, An=An)
end
