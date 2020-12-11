using Zygote:@show,hook
function dem(x, msg)
    # @show (norm(x), typeof(x), msg)
    return x
end

# todo version that doesn't have an fft built in
function (shears::ConvFFT{D,OT,A,B,C,PD,P})(x) where {D,OT,A,B,C,PD,P <: Tuple}
    if typeof(shears.weight) <: CuArray && !(typeof(x) <: Flux.CuArray ||
                                            typeof(x) <: CuArray)
        error("don't try to apply a CuArray to a non-CuArray")
    end
    xbc, usedInds = applyBC(x, shears.bc, ndims(shears.weight) - 1)

    Forward = shears.fftPlan[1]
    x̂ = Forward * xbc

    nextLayer = hook(x -> dem(x, "should be after internalConvFFT"), internalConvFFT(x̂,
                                shears.weight, usedInds, shears.fftPlan[2],
                                shears.bias, shears.analytic))
    tmp = hook(x -> dem(x, "nonlin"), shears.σ.(nextLayer))
    return hook(x -> dem(x, "entering ConvFFT"), tmp)
end

function (shears::ConvFFT)(x)
    if typeof(shears.weight) <: CuArray && !(typeof(x) <: CuArray)
        error("don't try to apply a CuArray to a non-CuArray")
    end
    xbc, usedInds = applyBC(x, shears.bc, ndims(shears.weight) - 1)

    F = shears.fftPlan
    x̂ = F * xbc

    nextLayer = internalConvFFT(x̂, shears.weight, usedInds, F,
                                shears.bias, shears.analytic)
    return shears.σ.(nextLayer)
end

abstract type TransformTypes end # If the input/wavelets are either real or
# analytic, there are efficiency gains to be had

struct AnalyticWavelet <: TransformTypes end
struct RealWaveletRealSignal <: TransformTypes end
struct RealWaveletComplexSignal <: TransformTypes end
struct NonAnalyticMatching <: TransformTypes end

function internalConvFFT(x̂, shears::AbstractArray{<:Number,N}, usedInds,
                         fftPlan, bias, isAnalytic) where N
    axShear = axes(shears)
    axx = axes(x̂)[N:end - 1]
    łλ(ii, bias) = applyWeight(x̂, shears[axShear[1:end - 1]..., ii], usedInds,
                       fftPlan, bias[ii], isAnalytic[ii])
    łλ(ii, bias::Nothing) = applyWeight(x̂, shears[axShear[1:end - 1]..., ii],
                                     usedInds, fftPlan, bias, isAnalytic[ii])
    mapped = map(ii -> łλ(ii, bias), 1:size(shears)[end])
    return permutedims(cat(mapped..., dims=1), ((2:N)..., 1, (N + 1):ndims(mapped[1])...))
end

# no bias, not analytic and both match (either both real or both complex)
function applyWeight(x̂, shear, usedInds, fftPlan, bias::Nothing, An::NonAnalyticMatching)
    tmp = fftPlan \ (shear .* x̂) # filter
    tmp = tmp[usedInds..., axes(tmp)[length(usedInds) + 1:end]...] # get rid of the padding
    tmp = reshape(tmp, (1, size(tmp)...)) # add a dummy dimension to join over
    return tmp
end

# no bias, analytic wavelet (so complex valued, but only the positive half of x̂
# matters)
function applyWeight(x̂, shear, usedInds, fftPlan, bias::Nothing, An::AnalyticWavelet)
    outer = axes(x̂)[ndims(shear) + 1:end]
    isSourceOdd = mod(size(fftPlan, 1) + 1, 2)
    accessedAxes = axes(shear)
    tmp = eltype(x̂).(shear .* x̂[accessedAxes..., outer...]) # filter
    wave = cat(tmp, adapt(tmp, zeros(eltype(tmp), size(shear, 1) - 1 - isSourceOdd,
                                     size(tmp)[2:end]...)), dims=1) # symmetrize
    tmp = fftPlan \ wave       # back to time domain
    tmp = tmp[usedInds..., axes(tmp)[length(usedInds) + 1:end]...] # get rid of the padding
    tmp = reshape(tmp, (1, size(tmp)...)) # add a dummy dimension to join over
    return tmp
end

# no bias, not analytic, still symmetric (i.e. real, this is for the averaging
# wavelet where the other wavelets are analytic/complex)
function applyWeight(x̂, shear, usedInds, fftPlan, bias::Nothing, An::RealWaveletRealSignal)
    outer = axes(x̂)[ndims(shear) + 1:end]
    isSourceOdd = mod(size(fftPlan, 1) + 1, 2)
    tmp = shear .* x̂ # filter
    wave = cat(tmp, reverse(conj.(tmp[2:end - isSourceOdd, outer...]), dims=1), dims=1) # symmetrize
    tmp = fftPlan \ wave        # back to time domain
    tmp = tmp[usedInds..., axes(tmp)[length(usedInds) + 1:end]...] # get rid of the padding
    tmp = reshape(tmp, (1, size(tmp)...)) # add a dummy dimension to join over
    return tmp
end

# no bias, signal asymmetric/complex, but the wavelet is real (averaging function)
function applyWeight(x̂, shear, usedInds, fftPlan, bias::Nothing, An::RealWaveletComplexSignal)
    outer = axes(x̂)[ndims(shear) + 1:end]
    isSourceOdd = mod(size(fftPlan, 1) + 1, 2)
    tmp = cat(shear, reverse(conj.(shear[2:end - isSourceOdd]),
                             dims=1), dims=1) # construct the full wavelet
    tmp = tmp .* x̂ # filter
    tmp = fftPlan \ tmp        # back to time domain
    tmp = tmp[usedInds..., axes(tmp)[length(usedInds) + 1:end]...] # get rid of the padding
    tmp = reshape(tmp, (1, size(tmp)...)) # add a dummy dimension to join over
    return tmp
end
# biased (and one of the others, doesn't matter which)
function applyWeight(x̂, shear, usedInds, fftPlan, bias, An)
    return applyWeight(x̂, shear, usedInds, fftPlan, nothing, An) .+
        bias
end
