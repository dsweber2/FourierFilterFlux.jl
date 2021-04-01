using FourierFilterFlux, ContinuousWavelets
using Flux, FFTW, CUDA, Shearlab, Wavelets, Zygote
using Logging, Test, LinearAlgebra
@testset "FourierFilterFlux.jl" begin
    include("boundaryTests.jl")
    include("CUDATests.jl")
    include("ConvFFTConstructors.jl")
    include("ConvFFTtransform.jl")
    include("shearletConv.jl")
    include("waveletConv.jl")
end
