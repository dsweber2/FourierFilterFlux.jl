# todo version that doesn't have an fft built in
function (shears::ConvFFT)(x)
    if maximum(shears.padBy) != 0
        x̂ = pad(x, shears.padBy)
    else
        x̂ = x
    end
    F = shears.fftPlan
    x̂ = F * x̂
    # actual location of the signal minus the padding
    usedInds = ([shears.padBy[ii] .+ (1:size(x, ii)) for ii=1:ndims(shears.weight)-1]..., )
    nextLayer = internalConvFFT(x̂, shears.weight, usedInds, F,
                                shears.bias)
    return shears.σ.(nextLayer)
end


function internalConvFFT(x̂, shears::AbstractArray{<:Number, N}, usedInds,
                         fftPlan, bias) where N
    axShear = axes(shears)
    axx = axes(x̂)[N:end-1]
    λ(ii)= applyWeight(x̂, shears[axShear[1:end-1]..., ii], usedInds, fftPlan,
                       (N, ii), bias)
    mapped = map(λ, 1:size(shears)[end])
    return permutedims(cat(mapped...,dims=1),
                       ((2:N)..., 1, (N+1):ndims(mapped[1])...))
end

# no bias
function applyWeight(x̂, shear, usedInds, fftPlan, wef, bias::Nothing)
    (N,ii)=wef
    #tmp = fftshift(fftPlan \ (shear .* x̂), fftPlan.region[2:end]) # filter
    #tmp = ifftshift(fftPlan \ (shear .* x̂), fftPlan.region) # filter
    tmp = fftPlan \ (shear .* x̂) # filter
    tmp = tmp[usedInds..., axes(tmp)[length(usedInds)+1:end]...] # get rid of the padding
    tmp = reshape(tmp, (1, size(tmp)...)) # add a dummy dimension to join over
    return tmp
end

# biased
function applyWeight(x̂, shear, usedInds, fftPlan, wef, bias)
    (N, ii) = wef
    return applyWeight(x̂, shear, usedInds, fftPlan, wef, nothing) .+ bias[axes(x̂)[N:end-1]..., ii]
end
