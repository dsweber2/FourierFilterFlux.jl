using Zygote:@show,hook
function dem(x,msg)
    #@show (norm(x), typeof(x), msg)
    return x
end

# todo version that doesn't have an fft built in
function (shears::ConvFFT{D, OT, A, B, C, PD, P})(x) where {D, OT, A, B, C, PD, P<:Tuple}
    if typeof(shears.weight)<: CuArray && !(typeof(x) <: Flux.CuArray)
        error("don't try to apply a CuArray to a non-CuArray")
    end
    xbc, usedInds = applyBC(x, shears.bc, ndims(shears.weight)-1)

    Forward = shears.fftPlan[1]
    x̂ = Forward * CUDA.CuArray(xbc)

    nextLayer = hook(x->dem(x,"should be after internalConvFFT"), internalConvFFT(x̂,
                                shears.weight, usedInds, shears.fftPlan[2], 
                                shears.bias, shears.analytic))
    tmp = hook(x->dem(x,"nonlin"), shears.σ.(nextLayer))
    return hook(x->dem(x, "entering ConvFFT"), tmp)
end
function (shears::ConvFFT)(x)
    if typeof(shears.weight)<: CuArray && !(typeof(x) <: CuArray)
        error("don't try to apply a CuArray to a non-CuArray")
    end
    xbc, usedInds = applyBC(x, shears.bc, ndims(shears.weight)-1)

    F = shears.fftPlan
    x̂ = F * xbc

    nextLayer = internalConvFFT(x̂, shears.weight, usedInds, F,
                                shears.bias, shears.analytic)
    return shears.σ.(nextLayer)
end


function internalConvFFT(x̂, shears::AbstractArray{<:Number, N}, usedInds,
                         fftPlan, bias, An) where N
    axShear = axes(shears)
    axx = axes(x̂)[N:end-1]
    if typeof(An) <: Tuple
        isAnalytic = map(ii->((ii in An) ? 1 : true), (1:size(shears)[end]...,))
    else
        isAnalytic = map(x->nothing, (1:size(shears)[end]...,))
    end
    x̂ = hook(x->dem(x,"noop"), x̂)
    łλ(ii,bias)= argWrapper(x̂, shears[axShear[1:end-1]..., ii], usedInds, 
                       fftPlan, bias[ii], isAnalytic[ii])
    łλ(ii,bias::Nothing)= argWrapper(x̂, shears[axShear[1:end-1]..., ii],
                                     usedInds, fftPlan, bias, isAnalytic[ii])
    mapped = hook(x->dem(x, "map"), map(ii->łλ(ii, bias), 1:size(shears)[end]))
    cats = hook(x->dem(x,"cat"), cat(mapped...,dims=1))
    dogs = hook(x->dem(x,"permute"), permutedims(cats, ((2:N)..., 1, (N+1):ndims(mapped[1])...)))
    return hook(x->dem(x,"entering internalConvFFT"), dogs)
end

# no bias, not analytic and real valued output
function applyWeight(x̂, shear, usedInds, fftPlan, bias::Nothing, An::Nothing)
    tmp = fftPlan \ (shear .* x̂) # filter
    tmp = tmp[usedInds..., axes(tmp)[length(usedInds)+1:end]...] # get rid of the padding
    tmp = reshape(tmp, (1, size(tmp)...)) # add a dummy dimension to join over
    return tmp
end

# no bias, analytic (so complex valued)
function applyWeight(x̂, shear, usedInds, fftPlan, bias::Nothing, An::Bool)
    outer = axes(x̂)[ndims(shear)+1:end]
    isSourceOdd = mod(size(fftPlan,1)+1,2)
    tmp = eltype(x̂).(shear .* x̂) # filter
    wave = cat(tmp, adapt(tmp, zeros(eltype(tmp), size(shear,1)-1-isSourceOdd,
                                     size(tmp)[2:end]...)), dims=1) # symmetrize
    tmp = fftPlan \ wave       # back to time domain
    tmp = tmp[usedInds..., axes(tmp)[length(usedInds)+1:end]...] # get rid of the padding
    tmp = reshape(tmp, (1, size(tmp)...)) # add a dummy dimension to join over
    return tmp
end

# no bias, not analytic, complex valued, but still symmetric
function applyWeight(x̂, shear, usedInds, fftPlan, bias::Nothing, An::Integer)
    outer = axes(x̂)[ndims(shear)+1:end]
    isSourceOdd = mod(size(fftPlan,1)+1,2)
    tmp = shear .* x̂ # filter
    wave = cat(tmp, reverse(conj.(tmp[2:end-isSourceOdd, outer...]), dims=1), dims=1) # symmetrize
    tmp = fftPlan \ wave        # back to time domain
    tmp = tmp[usedInds..., axes(tmp)[length(usedInds)+1:end]...] # get rid of the padding
    tmp = reshape(tmp, (1, size(tmp)...)) # add a dummy dimension to join over
    return tmp
end

# biased (and one of the others, doesn't matter which)
function applyWeight(x̂, shear, usedInds, fftPlan, bias, An)
    return applyWeight(x̂, shear, usedInds, fftPlan, nothing, An) .+
        bias
end


"""
this is purely because the pullback of applyWeight doesn't pass usedInds or the
bias backwards, even though it does x̂ and shear correctly
"""
argWrapper(x̂, shear, usedInds, fftPlan, bias, An) = applyWeight(x̂, shear, usedInds, fftPlan, bias, An)
Zygote.@adjoint function argWrapper(x̂, shear, usedInds, fftPlan, bias, An)
    # get what Zygote thinks it should be
    y, _back = Zygote.pullback(applyWeight, x̂, shear, usedInds, fftPlan, bias, An)
    function back(Δ)
        #println("in argwrapper, $(typeof(Δ))")
        ∂ = _back(Δ)
        return ∂[1], ∂[2], usedInds, ∂[4], bias, An
    end
    return y, back
end
