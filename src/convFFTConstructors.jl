############################ 2D methods ###############################


"""
    shearingLayer(n::Integer, m::Integer, channelsIn::Integer; scale=4,
                  shearLevels = scale, batchSize = 10, useGpu = true, dType
                  = Float32, σ = abs, trainable = false, plan = true)

create a ConvFFT layer that uses shearlets. By default it isn't trainable

if averagingLayer is true, create a ConvFFT layer that uses just the single averaging shearlet. By default it isn't trainable.
"""
function shearingLayer(inputSize::Union{Int,NTuple{N, T}}; 
                       scale = -1, shearLevel = scale, useGpu = true,
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
    nShears = shears.nShearlets
    if typeof(boundary) <:Pad
        boundary = Pad{2}(shears.padBy)
    end
    bias = nothing

    if useGpu
        shearlets = cu(shearlets)
    end
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
    waveletLayer(n::Integer, m::Integer, channelsIn::Integer; scale=4,
                  shearLevels = scale, batchSize = 10, useGpu = true, dType
                  = Float32, σ = abs, trainable = false, plan = true)

create a ConvFFT layer that uses wavelets from Wavelets.jl. By default it isn't trainable
"""
function waveletLayer(inputSize::Union{Int,NTuple{N, T}}; useGpu = true, 
                      dType = Float32, σ = abs, trainable = false,
                      plan = true, init = Flux.glorot_normal, bias=false,
                      padded=true, padBy= ceil(Int,inputSize[1]/10), cw =
                      WT.Morlet(), varargs...) where {N,T} 
    waveletType = wavelet(cw; varargs...)
    if !padded
        padBy = nothing
        effSize = inputSize[1] >> 1 + 1
    else
        effSize = (2*padBy + inputSize[1]) >> 1
    end
    wavelets,ω = computeWavelets(effSize, waveletType; T=T)
    if bias
        bias = init(inputSize[2:end-1]..., size(wavelets,2))
    else
        bias = nothing
    end
    if useGpu
        wavelets = cu(wavelets)
        bias = cu(bias)
    end


    return ConvFFT(wavelets, bias, inputSize, σ, plan=plan, dType = dType,
                   trainable=trainable, padBy = padBy)
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
