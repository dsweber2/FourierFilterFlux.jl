# todo version that doesn't have an fft built in
function (shears::ConvFFT{D, OT, A, B, C, PD, P})(x) where {D, OT, A, B, C, PD, P<:Tuple}
    if typeof(shears.weight)<: CuArray && !(typeof(x) <: CuArray)
        error("don't try to apply a CuArray to a non-CuArray")
    end
    xbc, usedInds = applyBC(x, shears.bc, ndims(shears.weight)-1)

    Forward = shears.fftPlan[1]
    x̂ = Forward * xbc    
    println("here")
    nextLayer = internalConvFFT(x̂, shears.weight, usedInds, shears.fftPlan[2],
                                shears.bias, shears.analytic)
    println("only application left")
    return shears.σ.(nextLayer)
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
    println("only application left")
    return shears.σ.(nextLayer)
end


function internalConvFFT(x̂, shears::AbstractArray{<:Number, N}, usedInds,
                         fftPlan, bias, An) where N
    axShear = axes(shears)
    axx = axes(x̂)[N:end-1]
    #println("internalConvFFT $An, $([typeof(An)<:Tuple ? ((ii in An) ? 1 : true) : nothing for ii=1:size(shears)[end]]))")
    if typeof(An) <: Tuple
        isAnalytic = [((ii in An) ? 1 : true) for ii=1:size(shears)[end]]
    else
        isAnalytic = [nothing for ii=1:size(shears)[end]]
    end
    #gatheredIters = zip(eachslice(shears,dims=ndims(shears)), isAnalytic)
    #λ(iters) = applyWeight(x̂,iters[1],usedInds, fftPlan, (N,N), bias, iters[2])
    #mapped = map(λ, gatheredIters)
    # mapped = [argWrapper(x̂, shears[axShear[1:end-1]..., ii], usedInds,
    #                       fftPlan, (N,ii), bias, isAnalytic[ii]) for ii in
    #           1:length(isAnalytic)]
    # index alone
    λ(ii)= argWrapper(x̂, shears[axShear[1:end-1]..., ii], usedInds, 
                       fftPlan, (N, ii), bias, 
                       isAnalytic[ii])
    mapped = map(λ, 1:size(shears)[end])
    println("made it here?")
    cats = cat(mapped...,dims=1)
    println("is it the cats?")
    dogs = permutedims(cats, ((2:N)..., 1, (N+1):ndims(mapped[1])...))
    println("no, is dogs")
    return dogs
end

# no bias, not analytic and real valued output
function applyWeight(x̂, shear, usedInds, fftPlan, bias::Nothing, An::Nothing)
    println("no bias, not analytic and real valued output, An=$An")
    tmp = fftPlan \ (shear .* x̂) # filter
    tmp = tmp[usedInds..., axes(tmp)[length(usedInds)+1:end]...] # get rid of the padding
    tmp = reshape(tmp, (1, size(tmp)...)) # add a dummy dimension to join over
    return tmp
end

# no bias, analytic (so complex valued)
function applyWeight(x̂, shear, usedInds, fftPlan, bias::Nothing, An::Bool)
    println("no bias, analytic (so complex valued), An=$An")
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
function applyWeight(x̂, shear, usedInds, fftPlan, bias::Nothing, An::Integer)
    println("no bias, not analytic, complex valued, but still symmetric, An=$An")
    outer = axes(x̂)[ndims(shear)+1:end]
    isSourceOdd = mod(size(fftPlan,1)+1,2)
    tmp = shear .* x̂ # filter
    #wave = cat(tmp, adapt(tmp, zeros(size(shear,1)-1-isSourceOdd,
    #size(tmp)[2:end]...)), dims=1) # symmetrize (WRONG)
    wave = cat(tmp, reverse(conj.(tmp[2:end-isSourceOdd, outer...]), dims=1), dims=1) # symmetrize
    tmp = fftPlan \ wave        # back to time domain
    tmp = tmp[usedInds..., axes(tmp)[length(usedInds)+1:end]...] # get rid of the padding
    tmp = reshape(tmp, (1, size(tmp)...)) # add a dummy dimension to join over
    println("can finish")
    return tmp
end

# biased (and one of the others, doesn't matter which)
function applyWeight(x̂, shear, usedInds, fftPlan, bias, An)
    println("biased (and one of the others, doesn't matter which)")
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
        @info " about to run ∂(applyWeight)" indices[2]
        ∂ = _back(y)
        @info "" length(Δ) Δ
        @info "actually returning" ∂[1] ∂[2] usedInds ∂[4] indices bias An
        #@info "actually returning" randn(Complex{Float64},101,1,2)
        #randn(Complex{Float64},101) usedInds nothing indices bias An # this
        #established that it is something about applyWeight that's the issue
        return ∂[1], ∂[2], usedInds, ∂[4], indices, bias, An
        #(randn(Complex{Float64},101,1,2), randn(Complex{Float64},101),
        #usedInds, nothing, indices, bias, An)
    end
    return y, back
end


# function applyWeight(x̂, shear, usedInds, fftPlan, indices, bias::Nothing, An)
#     if typeof(An) <: Nothing
#         println("no bias, not analytic and real valued output, An=$An, ii=$ii")
#         tmp = fftPlan \ (shear .* x̂) # filter
#         tmp = tmp[usedInds..., axes(tmp)[length(usedInds)+1:end]...] # get rid of the padding
#         tmp = reshape(tmp, (1, size(tmp)...)) # add a dummy dimension to join over
#         return tmp
#     elseif typeof(An) <: Bool
#         println("no bias, analytic (so complex valued), An=$An, ii=$ii")
#         outer = axes(x̂)[ndims(shear)+1:end]
#         isSourceOdd = mod(size(fftPlan,1)+1,2)
#         tmp = shear .* x̂ # filter
#         wave = cat(tmp, adapt(tmp, zeros(size(shear,1)-1-isSourceOdd, size(tmp)[2:end]...)), dims=1) # symmetrize
#         tmp = fftPlan \ wave       # back to time domain
#         tmp = tmp[usedInds..., axes(tmp)[length(usedInds)+1:end]...] # get rid of the padding
#         tmp = reshape(tmp, (1, size(tmp)...)) # add a dummy dimension to join over
#         return tmp
#     elseif typeof(An) <: Integer
#         println("no bias, not analytic, complex valued, but still symmetric, An=$An, ii=$ii")
#         outer = axes(x̂)[ndims(shear)+1:end]
#         isSourceOdd = mod(size(fftPlan,1)+1,2)
#         tmp = shear .* x̂ # filter
#         wave = cat(tmp, reverse(conj.(tmp[2:end-isSourceOdd, outer...]), dims=1), dims=1) # symmetrize
#         tmp = fftPlan \ wave        # back to time domain
#         tmp = tmp[usedInds..., axes(tmp)[length(usedInds)+1:end]...] # get rid of the padding
#         tmp = reshape(tmp, (1, size(tmp)...)) # add a dummy dimension to join over
#         println("can finish")
#         return tmp
#     end        
# end
