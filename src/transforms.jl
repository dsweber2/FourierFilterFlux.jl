# todo version that doesn't have an fft built in
function (shears::ConvFFT{D, OT, A, B, C, PD, P})(x) where {D, OT, A, B, C, PD, P<:Tuple}
    if typeof(shears.weight)<: CuArray && !(typeof(x) <: CuArray)
        error("don't try to apply a CuArray to a non-CuArray")
    end
    xbc, usedInds = applyBC(x, shears.bc, ndims(shears.weight)-1)

    Forward = shears.fftPlan[1]
    x̂ = hook(x->dem(x,"switching to fourier"), Forward * xbc)

    nextLayer = hook(x->dem(x,"internalConvFFT, $(typeof(x))"),
                     internalConvFFT(x̂,
                                     shears.weight, usedInds, shears.fftPlan[2], 
                                     shears.bias, shears.analytic))
    return shears.σ.(nextLayer)
end
using Zygote:@show,hook
function dem(x,msg)
    @show (norm(x), msg)
    return x
end
function (shears::ConvFFT)(x)
    if typeof(shears.weight)<: CuArray && !(typeof(x) <: CuArray)
        error("don't try to apply a CuArray to a non-CuArray")
    end
    xbc, usedInds = hook(applyBC(x, shears.bc, ndims(shears.weight)-1))

    F = shears.fftPlan
    x̂ = hook(x->dem(x,"switching to fourier"), F * xbc)

    nextLayer = hook(x->dem(x,"internalConvFFT, $(typeof(x))"), internalConvFFT(x̂, shears.weight, usedInds, F,
                                shears.bias, shears.analytic))
    return shears.σ.(nextLayer)
end


function internalConvFFT(x̂, shears::AbstractArray{<:Number, N}, usedInds,
                         fftPlan, bias, An) where N
    axShear = axes(shears)
    axx = axes(x̂)[N:end-1]
    if typeof(An) <: Tuple
        isAnalytic = [((ii in An) ? 1 : true) for ii=1:size(shears)[end]]
    else
        isAnalytic = [nothing for ii=1:size(shears)[end]]
    end
    x̂ = hook(x->dem(x,"noop"), x̂)
    λ(ii)= argWrapper(x̂, shears[axShear[1:end-1]..., ii], usedInds, 
                       fftPlan, (N, ii), bias, 
                       isAnalytic[ii])
    mapped = hook(x->dem(x,"map"), map(λ, 1:size(shears)[end]))
    cats = hook(x->dem(x,"cat"), cat(mapped...,dims=1))
    dogs = hook(x->dem(x,"permute"), permutedims(cats, ((2:N)..., 1, (N+1):ndims(mapped[1])...)))
    return dogs
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
    tmp = hook(x->dem(x,"hook plan divides"), fftPlan \ wave)       # back to time domain
    tmp = tmp[usedInds..., axes(tmp)[length(usedInds)+1:end]...] # get rid of the padding
    tmp = reshape(tmp, (1, size(tmp)...)) # add a dummy dimension to join over
    return tmp
end

# no bias, not analytic, complex valued, but still symmetric
function applyWeight(x̂, shear, usedInds, fftPlan, bias::Nothing, An::Integer)
    outer = axes(x̂)[ndims(shear)+1:end]
    isSourceOdd = mod(size(fftPlan,1)+1,2)
    tmp = hook(x->dem(x,"filtering"), shear .* x̂) # filter
    wave = cat(tmp, reverse(conj.(tmp[2:end-isSourceOdd, outer...]), dims=1), dims=1) # symmetrize
    tmp = hook(x->dem(x,"hook plan divides"), fftPlan \ wave)        # back to time domain
    tmp = hook(x->dem(x,"drop padding"), tmp[usedInds..., axes(tmp)[length(usedInds)+1:end]...]) # get rid of the padding
    tmp = hook(x->dem(x,"reshape"), reshape(tmp, (1, size(tmp)...))) # add a dummy dimension to join over
    return hook(x->dem(x,"into apply weight"), tmp)
end

# biased (and one of the others, doesn't matter which)
function applyWeight(x̂, shear, usedInds, fftPlan, bias, An)
    return applyWeight(x̂, shear, usedInds, fftPlan, nothing, An) .+
        bias[axes(x̂)[N:end-1]..., ii]
end


"""
this is purely to make sure the pullback has all necessary info
"""
argWrapper(x̂, shear, usedInds, fftPlan, indices, bias, An) = applyWeight(x̂, shear, usedInds, fftPlan, bias, An)
Zygote.@adjoint function argWrapper(x̂, shear, usedInds, fftPlan, indices,
                             bias, An)
    # get what Zygote thinks it should be
    y, _back = Zygote.pullback(applyWeight, x̂, shear, usedInds, fftPlan, bias, An)
    function back(Δ)
        println("arg wrapper start $(norm(Δ)) $(typeof(Δ)), $(norm(Δ))")
        ∂ = _back(Δ)
        println("argWrapper $(norm(∂[1]))")
        return ∂[1], ∂[2], usedInds, ∂[4], indices, bias, An
    end
    return y, back
end
