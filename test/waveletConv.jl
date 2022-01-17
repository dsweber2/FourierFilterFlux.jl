CWs = [Morlet(), Morlet(4π), dog1, paul16];
inputSizes = ((305, 2), (256, 1, 4), 129);
scales = [1, 8, 12];
averagingLengths = (-2, 0, 2);
normalizations = [1.0, Inf];
βs = [1.0, 3.5];
σs = [identity, abs, relu];
function compareWithWavelet(inputSize, cw, β, normalization, scale, averagingLength)
    x = randn(Float32, inputSize)
    W = 3
    waves = 4
    wWave = 4
    # for the purpose of testing, we don't need the wall of warnings
    with_logger(ConsoleLogger(stderr, Logging.Error)) do
        #global W, waves
        W = waveletLayer(size(x), cw = cw, decreasing = β,
            averagingLength = averagingLength,
            normalization = normalization, s = scale)

        waves = wavelet(cw; s = scale, averagingLength = averagingLength,
            normalization = normalization, decreasing = β)
    end

    if (typeof(cw) <: Union{Morlet,Paul})
        @test analytic(W)
    end
    xCu = x
    wFFF = W(xCu)

    wWave = 5
    with_logger(ConsoleLogger(stderr, Logging.Error)) do
        wWave = ContinuousWavelets.cwt(x, waves)
    end
    testRes = @test size(wFFF)[1:2] == size(wWave)[1:2] # not sure why it was
    # triple equal...
    if !(typeof(testRes) <: Test.Pass)
        @info "values are" inputSize cw β averagingLength normalization scale
    end
    reord = [2:size(wWave, 2)..., 1]
    testRes = @test norm((cpu(wFFF) - wWave[:, reord, axes(wFFF)[3:end]...])) ./ norm(x) ≈ 0 atol = 2.0f-7 # just fft ifft is on this order
    if !(typeof(testRes) <: Test.Pass)
        @info "values are" inputSize cw β averagingLength normalization scale
    end
end
@testset "Wavelets.jl construction and application" begin
    CWs = [Morlet(), Morlet(4π), dog1, paul16]
    inputSizes = ((305, 2), (256, 1, 4), 129)
    scales = [1, 8, 12]
    averagingLengths = (0, 2, 4)
    normalizations = [1.0, Inf]
    βs = [1.0, 3.5]
    σs = [identity, abs, relu]

    for inputSize in inputSizes, cw in CWs, β in βs, normalization in
                                                     normalizations, scale in scales, averagingLength in averagingLengths
        @testset "inputSize=$inputSize, cw =$(cw), β=$β" begin
            compareWithWavelet(inputSize, cw, β, normalization, scale, averagingLength)
        end
    end
end
