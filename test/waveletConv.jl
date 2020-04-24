cw = WT.Morlet(); β = 4.0; averagingLength = 2; normalization = Inf; scale = 8
inputSize = (305,2)
function f(inputSize, cw, β, normalization, scale)
    x = randn(Float32,inputSize)
    W = waveletLayer(size(x),cw=cw,decreasing=β, averagingLength=averagingLength,
                     normalization=normalization, s = scale)
    waves = wavelet(cw; s=scale, averagingLength=averagingLength,
                    normalization=normalization, decreasing=β)
    if (typeof(cw) <: Union{WT.Morlet, WT.Paul})
        @test analytic(W)
    end
    xCu = cu(x)
    wFFF = W(xCu);
    wWave = cwt(x, waves);
    @test sizeof(wFFF)===sizeof(wWave)
    tesRes = @test norm((cpu(wFFF)-wWave))./norm(x) ≈ 0 atol=1f-7 # just fft ifft is on this order
    if !(typeof(tesRes) <: Test.Pass)
        @info "values are" inputSize cw β averagingLength normalization scale 
    end
end
@testset "Wavelets.jl construction and application" begin
    CWs = [WT.Morlet(), WT.Morlet(4π), WT.dog1, WT.paul16]
    inputSizes = ((305,2),(256,1,4))
    scales = [1,8,12]
    averagingLengths =(0, 2, 4)
    normalizations=[1.0, Inf]
    βs = [1.0,3.5]
    
    for inputSize in inputSizes, cw in CWs, β in βs, normalization in
        normalizations, scale in scales 
        f(inputSize, cw, β, normalization, scale)
    end
end

# standard input size
@testset "Wavelets.jl constructors tiny" begin
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
