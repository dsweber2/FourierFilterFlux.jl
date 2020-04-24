#using Revise
using FourierFilterFlux
using Test, Flux, LinearAlgebra, FFTW, CuArrays, Shearlab, Wavelets

@testset "FourierFilterFlux.jl" begin
    include("boundaryTests.jl")
    include("ConvFFTConstructors.jl")
    include("ConvFFTtransform.jl")
    include("shearletConv.jl")
end
