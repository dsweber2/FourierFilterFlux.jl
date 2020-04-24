# todo version that doesn't have an fft built in
function (shears::ConvFFT{D, OT, A, B, C, PD, P})(x) where {D, OT, A, B, C, PD, P<:Tuple}
    if typeof(shears.weight)<: CuArray && !(typeof(x) <: CuArray)
        error("don't try to apply a CuArray to a non-CuArray")
    end
    xbc, usedInds = applyBC(x, shears.bc, ndims(shears.weight)-1)

    Forward = shears.fftPlan[1]
    x̂ = Forward * xbc    

    nextLayer = internalConvFFT(x̂, shears.weight, usedInds, shears.fftPlan[2],
                                shears.bias, shears.analytic)
    return shears.σ.(nextLayer)
end

function (shears::ConvFFT)(x)
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
    λ(ii)= applyWeight(x̂, shears[axShear[1:end-1]..., ii], usedInds, 
                       fftPlan, (N, ii), bias, 
                       typeof(An)<:Tuple ? ((ii in An) ? 1 : true) : nothing)
    mapped = map(λ, 1:size(shears)[end])
    return permutedims(cat(mapped...,dims=1),
                       ((2:N)..., 1, (N+1):ndims(mapped[1])...))
end

# no bias, not analytic and real valued output
function applyWeight(x̂, shear, usedInds, fftPlan, indices, bias::Nothing, An::Nothing)
    (N,ii)=indices
    tmp = fftPlan \ (shear .* x̂) # filter
    tmp = tmp[usedInds..., axes(tmp)[length(usedInds)+1:end]...] # get rid of the padding
    tmp = reshape(tmp, (1, size(tmp)...)) # add a dummy dimension to join over
    return tmp
end

# no bias, analytic (so complex valued)
function applyWeight(x̂, shear, usedInds, fftPlan, indices, bias::Nothing, An::Bool)
    (N,ii)=indices
    outer = axes(x̂)[ndims(shear)+1:end]
    isSourceOdd = mod(size(fftPlan,1)+1,2)
    tmp = shear .* x̂ # filter
    wave = cat(tmp, adapt(tmp, zeros(size(shear,1)-1-isSourceOdd, size(tmp)[2:end]...)), dims=1) # symmetrize
    tmp = fftPlan \ wave       # back to time domain
    tmp = tmp[usedInds..., axes(tmp)[length(usedInds)+1:end]...] # get rid of the padding
    tmp = reshape(tmp, (1, size(tmp)...)) # add a dummy dimension to join over
    return tmp
end

# no bias, not analytic, complex valued, but still symmetric
function applyWeight(x̂, shear, usedInds, fftPlan, indices, bias::Nothing, An::Integer)
    (N,ii)=indices
    outer = axes(x̂)[ndims(shear)+1:end]
    isSourceOdd = mod(size(fftPlan,1)+1,2)
    tmp = shear .* x̂ # filter
    wave = cat(tmp, reverse(conj.(tmp[2:end-isSourceOdd, outer...]), dims=1), dims=1) # symmetrize
    tmp = fftPlan \ wave        # back to time domain
    tmp = tmp[usedInds..., axes(tmp)[length(usedInds)+1:end]...] # get rid of the padding
    tmp = reshape(tmp, (1, size(tmp)...)) # add a dummy dimension to join over
    return tmp
    (N,ii)=indices
end

# biased (and one of the others, doesn't matter which)
function applyWeight(x̂, shear, usedInds, fftPlan, indices, bias, An)
    (N, ii) = indices
    return applyWeight(x̂, shear, usedInds, fftPlan, indices, nothing) .+ bias[axes(x̂)[N:end-1]..., ii]
end
