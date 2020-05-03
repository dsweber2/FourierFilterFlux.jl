#using Revise
using FourierFilterFlux
using Flux, FFTW, CuArrays, Shearlab, Wavelets
using Logging, Test, LinearAlgebra

@testset "FourierFilterFlux.jl" begin
    include("boundaryTests.jl")
    include("ConvFFTConstructors.jl")
    include("ConvFFTtransform.jl")
    include("shearletConv.jl")
    include("waveletConv.jl")
end
