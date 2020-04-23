scale =4; shearLevel = 3; inputSize = (25,25,1,2); useGpu = false; σm =relu; dType=Float32
# standard input size
@testset "shearing constructors tiny" begin
    inputSizes = [(25,25,1,2), (25, 25, 4, 5, 3)]
    scalesShears = [(1,1),(2,1), (2,2),(4,1), (4,2), (4,3), (4,4)]#, (8,1), (8,2), (8,3), (8,4), (8,6), (8,8)]
    shearLevels = [1,2,4,5]
    useGpus = [true, false]
    dType = Float32
    σs =[identity, abs, relu]
    for inputSize in inputSizes, (scale, shearLevel) in scalesShears, useGpu in useGpus, σm in σs

        shears = shearingLayer(inputSize, scale=scale, shearLevel=shearLevel,
                               useGpu = useGpu, dType = dType, σ=σm)

        @test size(shears.fftPlan) == inputSize .+ (2 .* shears.bc.padBy...,
                                                   fill(0, length(inputSize)-2)...)
        @test shears.σ == σm
        @test shears.bias == nothing
        @test ndims(shears.weight)==3
        if useGpu
            init = gpu(randn(dType, inputSize));
        else
            init = randn(dType, inputSize);
        end

        fullResult = shears(init);
        @test size(fullResult) == (inputSize[1:2]...,size(shears.weight,3),
                                   inputSize[3:end]...)
        if σm!=relu
            @test minimum(abs.(fullResult)) > 0
        end

        # compare with the result from Shearlab itself
        effectiveShears = ceil.(Int, (1:shearLevel)/2)
        cpuShears = Shearlab.getshearletsystem2D(inputSize[1:2]..., scale,
                                                 effectiveShears, 
                                                 typeBecomes=Float32,
                                                 padded=true)
        res = Shearlab.sheardec2D(init, cpuShears)

        @test σm.(res) ≈ cpu(fullResult)
        # just the averaging filter
        shears = shearingLayer(inputSize, scale=scale, shearLevel=shearLevel, useGpu =
                             useGpu, dType = dType, σ=σm, averagingLayer=true)
        @test size(shears.weight, 3) ==1
        singleRes = shears(init)
        ax = axes(fullResult)
        @test fullResult[:,:,end:end, ax[4:end]...] ≈ singleRes
    end
end

inputSize = (400,400,1,2); scale=1; shearLevel=1; useGpu=true; σm=abs
@testset "shearing constructors large" begin
    # realistic size example
    inputSizes = [(400,400,1,2)]
    scalesShears = [(1,1), (2,2), (4,4), (8,1), (8,4), (8,8)]
    useGpus = [true,false]
    σs = [identity, abs, relu]
    dType = Float32
    for inputSize in inputSizes, (scale, shearLevel) in scalesShears, useGpu in useGpus, σm in σs
        shears = shearingLayer(inputSize, scale=scale, shearLevel=shearLevel,
                               useGpu = useGpu, dType = dType, σ=σm) 
        @test size(shears.fftPlan)== inputSize .+ (2 .* shears.bc.padBy...,
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
        #println("$inputSize, $scale, $shearLevel, $useGpu, $σm")
        #typeof(res)
        @test σm.(res) ≈ cpu(fullResult)
        # if !(typeof(tesRes)<:Test.Pass)
        #     println("PROBLEM:    $inputSize, $scale, $shearLevel, $useGpu, $σm")
        # else
        #     println("fine: $inputSize, $scale, $shearLevel, $useGpu, $σm")
        # end
        # just the averaging filter
        shears = shearingLayer(inputSize, scale=scale, shearLevel=shearLevel,
                               useGpu = useGpu, dType = dType, σ=σm,
                               averagingLayer=true)       
        @test size(shears.fftPlan)== inputSize .+ (2 .* shears.bc.padBy...,
                                                fill(0, length(inputSize)-2)...)
        @test shears.σ == σm
        @test shears.bias == nothing
        @test ndims(shears.weight)==3
        @test size(shears.weight, 3) ==1
    end
end
