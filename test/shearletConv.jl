scale =2; shearLevel = 2; inputSize = (25,25,4,5,3); useGpu = false; σm =abs; dType=Float32
cpuShears = 3; singleRes = 3; shears = 3
# standard input size
@testset "shearing constructors tiny" begin
    inputSizes = [(25,25,1,2), (25, 25, 4, 5, 3)]
    scalesShears = [(1,1),(2,1), (2,2),(4,1), (4,2), (4,3), (4,4)]#, (8,1), (8,2), (8,3), (8,4), (8,6), (8,8)]
    shearLevels = [1,2,4,5]
    if CUDA.functional()
        useGpus = [true, false]
    else
        useGpus = [false]
    end
    dType = Float32
    σs =[identity, abs, relu]
    for inputSize in inputSizes, (scale, shearLevel) in scalesShears, useGpu in useGpus, σm in σs
        @testset "inputSize=$inputSize, (scale, shearLevel) =$((scale,shearLevel)),useGpu=$useGpu, σm=$(σm)" begin
            shears=3
            with_logger(ConsoleLogger(stderr,Logging.Error)) do
                #global shears
                shears = shearingLayer(inputSize, scale=scale, shearLevel=shearLevel,
                                       dType = dType, σ=σm)
            end
            @test size(shears.fftPlan) ==
                inputSize .+ (2 .* shears.bc.padBy...,
                              fill(0, length(inputSize)-2)...)
            @test shears.σ == σm
            @test shears.bias == nothing
            @test ndims(shears.weight)==3
            @test outType(shears)<:Real
            if useGpu
                init = gpu(randn(dType, inputSize));
                shears = shears |> gpu
            else
                init = randn(dType, inputSize);
            end

            fullResult = shears(init);
            @test size(fullResult) == (inputSize[1:2]...,size(shears.weight,3),
                                       inputSize[3:end]...)
            if σm!=relu
                # are there only a very small number of zeros?
                @test count(abs.(fullResult) .== 0) < .001 *prod(size(fullResult))
            end
            # compare with the result from Shearlab itself
            effectiveShears = ceil.(Int, (1:shearLevel)/2)
            cpuShears=3; res = 2
            with_logger(ConsoleLogger(stderr,Logging.Error)) do
                #global cpuShears, res
                cpuShears = Shearlab.getshearletsystem2D(inputSize[1:2]...,
                                                         scale,
                                                         effectiveShears,
                                                         typeBecomes=Float32,
                                                         padded=true)
                res = Shearlab.sheardec2D(init, cpuShears)
            end
            @test σm.(res) ≈ cpu(fullResult)
            # just the averaging filter
            with_logger(ConsoleLogger(stderr,Logging.Error)) do
                #global shears
                shears = shearingLayer(inputSize, scale=scale, shearLevel=shearLevel,
                                       dType = dType, σ=σm, averagingLayer=true)
            end
            @test size(shears.weight, 3) ==1
            if useGpu
                shears = shears |> gpu
            end
            singleRes = 3
            with_logger(ConsoleLogger(stderr,Logging.Error)) do
                #global singleRes
                singleRes = shears(init)
            end
            ax = axes(fullResult)
            @test fullResult[:,:,end:end, ax[4:end]...] ≈ singleRes
        end
    end
end
