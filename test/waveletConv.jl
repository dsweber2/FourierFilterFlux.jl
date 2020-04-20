shearLevel = 1; scale =2; inputSize = (200,1,2); useGpu = false; σm =abs; dType=Float32
# standard input size
@testset "shearing constructors tiny" begin
    inputSizes = [(200,1,2), (225, 4, 5, 3)]
    scalesShears = [(1,1),(2,1), (2,2),(4,1), (4,2), (4,3), (4,4)]#, (8,1), (8,2), (8,3), (8,4), (8,6), (8,8)]
    shearLevels = [1,2,4,5]
    useGpus = [true, false]
    dType = Float32
    σs =[identity, abs, relu]
    for inputSize in inputSizes, (scale, shearLevel) in scalesShears, useGpu in useGpus, σm in σs
        shears = waveletLayer(inputSize, useGpu = useGpu, dType = dType, σ=σm, decreasing=3.0)
        @test size(shears.fftPlan) == inputSize .+ (2 .* shears.padBy, fill(0, length(inputSize)-1)...,)
        @test shears.σ == σm
        @test shears.bias == nothing
        @test ndims(shears.weight)==2
        if useGpu
            init = cu(randn(dType, inputSize));
        else
            init = randn(dType, inputSize);
        end
        plot(shears.weight)
        fullResult = shears(init);
        @test size(fullResult) == (inputSize[1]...,size(shears.weight,2),
                                   inputSize[2:end]...)
        if σm!=relu
            @test minimum(abs.(fullResult)) > 0
        end

        # compare with the result from Shearlab itself
        w = wavelet(WT.Morlet(), boundary=WT.ZPBoundary())
        transf =cwt(init, w)
        @test σm.(transf) ≈ cpu(fullResult)

        
        # just the averaging filter
        shears = averagingLayer(inputSize, scale=scale, shearLevel=shearLevel, useGpu =
                             useGpu, dType = dType, σ=σm)
        @test size(shears.fftPlan)== inputSize .+ (2 .* shears.padBy...,
                                                fill(0, length(inputSize)-2)...)
        @test shears.σ == σm
        @test shears.bias == nothing
        @test ndims(shears.weight)==3
        @test size(shears.weight, 3) ==1
        singleRes = shears(init)
        ax = axes(fullResult)
        @test fullResult[:,:,end:end, ax[4:end]...] ≈ singleRes
    end
end

@testset "shearing constructors large" begin
    # realistic size example
    inputSizes = [(400,400,1,2)]
    scalesShears = [(1,1), (2,2), (4,4), (8,1), (8,4), (8,8)]
    useGpus = [true,false]
    σs = [identity, abs, relu]
    dType = Float32
    for inputSize in inputSizes, (scale, shearLevel) in scalesShears, useGpu in useGpus, σm in σs
        shears = shearingLayer(inputSize, scale=scale, shearLevel=shearLevel, useGpu =
                            useGpu, dType = dType, σ=σm) 
        @test size(shears.fftPlan)== inputSize .+ (2 .* shears.padBy...,
                                                fill(0, length(inputSize)-2)...)
        @test shears.σ == σm
        @test shears.bias == nothing
        @test ndims(shears.weight)==3

        if useGpu
            init = cu(randn(dType, inputSize));
        else
            init = randn(dType, inputSize);
        end

        fullResult = shears(init);

        effectiveShears = ceil.(Int, (1:shearLevel)/2)
        cpuShears = Shearlab.getshearletsystem2D(inputSize[1:2]..., scale,
                                                 effectiveShears, 
                                                 typeBecomes=Float32,
                                                 padded=true)
        res = Shearlab.sheardec2D(init, cpuShears)
        @test res ≈ σm.(cpu(fullResult))


        # just the averaging filter
        shears = averagingLayer(inputSize, scale=scale, shearLevel=shearLevel, useGpu =
                             useGpu, dType = dType, σ=σm)
        @test size(shears.fftPlan)== inputSize .+ (2 .* shears.padBy...,
                                                fill(0, length(inputSize)-2)...)
        @test shears.σ == σm
        @test shears.bias == nothing
        @test ndims(shears.weight)==3
        @test size(shears.weight, 3) ==1
    end
end
