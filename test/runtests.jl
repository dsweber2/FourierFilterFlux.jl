using FourierFilterFlux
using Test, Flux

@testset "FourierFilterFlux.jl" begin
    include("boundaryTests.jl")
    include("ConvFFTConstructors.jl")
    include("ConvFFTtransform.jl")
    include("shearletConv.jl")
end
