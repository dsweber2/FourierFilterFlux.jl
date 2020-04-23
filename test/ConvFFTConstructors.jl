# TODO: add some checks for different boundary conditions
# TODO: add checks for analytic wavelets
# ConvFFT constructor tests
@testset "ConvFFT constructors" begin
    @testset "Utils" begin
        explicit = [1 0 0; 0 1 0; 0 0 1; zeros(2,3)]
        @test maximum(abs.(iden_perturbed_gaussian(5,3) - explicit) ) < 1
    end
    @testset "2D constructors" begin
        # normal size
        originalSize = (21,11,1,10)
        x = randn(Float32, originalSize)
        weightMatrix = randn(Float32, (21+10)>>1+1, 11+10, 1)
        weightMatrix = reshape([I zeros(16,5)], (16,21,1))
        padding = (5,5)
        shears = ConvFFT(weightMatrix, nothing, originalSize, abs,
                         plan=true, boundary = Pad(padding))
        @test size(shears.fftPlan)== originalSize .+ (10, 10, 0, 0)
        @test shears.σ == abs
        @test shears.bias == nothing
        @test shears.bc.padBy == (5,5)

        shears = ConvFFT(weightMatrix, nothing, originalSize, abs,
                         plan=true, boundary = Pad(padding), trainable=true)
        @test params(shears).order[1] == shears.weight
        @test length(params(shears).order)==1

        shears = ConvFFT(weightMatrix, nothing, originalSize, abs, 
                         boundary = Pad(padding),trainable=false)
        @test isempty(params(shears))

        x = randn(21,11,1,10)
        ∇ = gradient((x)->shears(x)[1,1,1,1,3], x)
        @test minimum(∇[1][:,:,:, [1:2..., 4:10...]] .≈ 0)

        # check that the identity ConvFFT is, in fact, an identity
        weightMatrix = ones(Float32, (21+10)>>1+1, 11+10, 1)
        weightMatrix = cat(weightMatrix, weightMatrix, dims=3)
        padding = (5,5)
        shears = ConvFFT(weightMatrix, nothing, originalSize, identity,
                         plan=true, boundary = Pad(padding))
        x = randn(Float32,21,11,1,10)
        @test shears(x)[:,:,2,:,:] ≈ x

        # check that global multiplication in the Fourier domain is just multiplication
        weightMatrix = 2 .* ones(Float32, (21+10)>>1+1, 11+10, 1)
        padding = (5,5)
        shears = ConvFFT(weightMatrix, nothing, originalSize, identity,
                         plan=true, boundary = Pad(padding))
        x = randn(21,11,1,10)
        @test shears(x)[:,:,1,:,:] ≈ 2 .* x    

        # internal methods tests
        x̂ = pad(x, shears.bc.padBy);
        x̂ = shears.fftPlan *ifftshift(x̂,(1,2))
        usedInds1 = shears.bc.padBy[1] .+ (1:size(x, 1))
        usedInds2 = shears.bc.padBy[2] .+ (1:size(x, 2))
        usedInds = (usedInds1, usedInds2)
        nextLayer = FourierFilterFlux.internalConvFFT(x̂, shears.weight, usedInds,
                                                       shears.fftPlan, nothing)
        ∇ = gradient((x̂)->FourierFilterFlux.internalConvFFT(x̂,
                                                             shears.weight,
                                                             usedInds,
                                                             shears.fftPlan, nothing)[1,1,1,1,1],
                     x̂)
        @test minimum(abs.(diag(∇[1][:, :, 1,1])).≈ 2f0/31/21)

        ax = axes(x̂)[3:end-1]
        tmp = FourierFilterFlux.applyWeight(x̂, shears.weight, usedInds,
                                             shears.fftPlan, (ax, 1), nothing)
        ∇ = gradient((x̂) -> FourierFilterFlux.applyWeight(x̂, shears.weight[:,:,1], usedInds,
                                                           shears.fftPlan, (ax, 1),
                                                           nothing)[1,1,1,1,1], x̂) 
        @test minimum(abs.(diag(∇[1][:, :, 1,1])).≈ 2f0/31/21)

        ∇ = gradient((x̂) -> (shears.fftPlan \ (x̂ .* shears.weight[:,:,1]))[1,1,1,1], x̂)
        @test minimum(abs.(diag(∇[1][:, :, 1,1])).≈ 1f0/31*2/21)
        sheared = shears(x)
        @test size(sheared) == (21,11,1,1,10)
        
        # convert to a gpu version
        gpuVer = shears |>gpu
        typeof(gpuVer.weight)
        # TODO: this isn't implemented quite yet
        #@test typeof(gpuVer.weight) <: CuArray
        #@test typeof(gpuVer.fftPlan) <: CuArrays.CUFFT.rCuFFTPlan

        # extra channel dimension
        originalSize = (20,10,16, 1,10)
        shears = ConvFFT(randn(Float32, 16, 20, 3), nothing, originalSize, abs,
                         plan=true, boundary = Pad((5,5)))
        @test size(shears.fftPlan)== originalSize .+ (10, 10, 0, 0, 0)
        @test shears.σ == abs
        @test shears.bias == nothing
        @test shears.bc.padBy == (5,5)
        wSize = originalSize[1:2] .+ (10, 10)
        wSize = (wSize[1]>>1+1, wSize[2], 3)
        @test size(shears.weight) == wSize

        # random initialization
        originalSize = (20, 10, 16, 1, 10)
        shears = ConvFFT(originalSize, 3, abs,
                         plan=true, boundary = Pad((5,5)))
        @test size(shears.fftPlan)== originalSize .+ (10, 10, 0, 0, 0)
        @test shears.σ == abs
        @test size(shears.bias) == (originalSize[3:4]..., 3)
        @test shears.bc.padBy == (5,5)
        wSize = originalSize[1:2] .+ (10, 10)
        wSize = (wSize[1]>>1+1, wSize[2], 3)
        @test size(shears.weight) == wSize
    end


    @testset "1D constructors" begin
        # normal size
        originalSize = (21,1,10)
        x = randn(Float32, originalSize)
        weightMatrix = randn(Float32, (21+10)>>1 + 1, 1)
        padding = (5,)
        shears = ConvFFT(weightMatrix, nothing, originalSize, abs,
                         plan=true, boundary = Pad(padding))
        @test size(shears.fftPlan)== originalSize .+ (10, 0, 0)
        @test shears.σ == abs
        @test shears.bias == nothing
        @test shears.bc.padBy == (5,)

        shears = ConvFFT(weightMatrix, nothing, originalSize, abs,
                         plan=true, boundary = Pad(padding), trainable=true)
        @test params(shears).order[1] == shears.weight

        shears = ConvFFT(weightMatrix, nothing, originalSize, abs,
                         plan=true, boundary = Pad(padding), trainable=false)
        @test isempty(params(shears))

        x = randn(21,1,10)
        ∇ = gradient((x)->shears(x)[1,1,1,3], x)
        @test minimum(∇[1][:,:, [1:2..., 4:10...]] .≈ 0)

        # check that the identity ConvFFT is, in fact, an identity
        weightMatrix = ones(Float32, (21+10)>>1+1, 1)
        weightMatrix = cat(weightMatrix, weightMatrix, dims=2)
        padding = 5
        shears = ConvFFT(weightMatrix, nothing, originalSize, identity,
                         plan=true, boundary = Pad(padding))
        x = randn(Float32,21,1,10)
        @test shears(x)[:,1,:,:] ≈ x

        # check that global multiplication in the Fourier domain is just multiplication
        weightMatrix = 2 .* ones(Float32, (21+10)>>1+1, 1)
        padding = 5
        shears = ConvFFT(weightMatrix, nothing, originalSize, identity,
                         plan=true, boundary = Pad(padding))
        x = randn(21,1,10)
        @test shears(x)[:,1,:,:] ≈ 2 .* x    

        # internal methods tests
        x̂ = pad(x, shears.bc.padBy); 
        x̂ = shears.fftPlan *ifftshift(x̂,(1,2))
        usedInds = (shears.bc.padBy[1] .+ (1:size(x, 1)),)
        nextLayer = FourierFilterFlux.internalConvFFT(x̂, shears.weight, usedInds,
                                                       shears.fftPlan, nothing)
        ∇ = gradient((x̂)->FourierFilterFlux.internalConvFFT(x̂,
                                                             shears.weight,
                                                             usedInds,
                                                             shears.fftPlan, nothing)[1,1,1,1,1],
                     x̂)
        @test minimum(abs.(∇[1][:, 1,1]).≈ 2f0/31)
        ∇ = gradient((x̂) -> (shears.fftPlan \ (x̂ .* shears.weight[:,:,1]))[1,1,1,1], x̂)
        @test minimum(abs.(∇[1][:, :, 1,1]).≈ 1f0/31*2)
        sheared = shears(x)
        @test size(sheared) == (21,1,1,10)

        # convert to a gpu version
        gpuVer = shears |>gpu
        # TODO: this isn't implemented quite yet
        #@test typeof(gpuVer.weight) <: CuArrays.CuArray
        #@test typeof(gpuVer.fftPlan) <: CuArrays.CUFFT.rCuFFTPlan

        # extra channel dimension
        originalSize = (20,16, 1,10)
        shears = ConvFFT(randn(Float32, 16, 3), nothing, originalSize, abs,
                         plan=true, boundary = Pad(5))
        @test size(shears.fftPlan)== originalSize .+ (10, 0, 0, 0)
        @test shears.σ == abs
        @test shears.bias == nothing
        @test shears.bc.padBy == (5,)
        wSize = ((originalSize[1] + (10)) >>1 +1, 3)
        @test size(shears.weight) == wSize

        # random initialization
        originalSize = (20, 16, 1, 10)
        shears = ConvFFT(originalSize, 3, abs,
                         plan=true, boundary = Pad(5), nConvDims=1)
        @test size(shears.fftPlan)== originalSize .+ (10, 0, 0, 0)
        @test shears.σ == abs
        @test size(shears.bias) == (originalSize[2:end-1]..., 3)
        @test shears.bc.padBy == (5,)
        wSize = ((originalSize[1] + 10)>>1 + 1, 3)
        @test size(shears.weight) == wSize
    end
end
