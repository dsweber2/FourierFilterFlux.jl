# test padding
@testset "padding dimension" begin
    @testset "padding 1D" begin
        # no meta dims
        input = ones(Float32, 10)
        padded = pad(input, (5,))
        padded = pad(input, 5)
        @test size(padded) == (20,)
        @test padded[1]== 0.0
        @test padded[6]== 1.0
        @test eltype(padded) == eltype(input)

        # one meta dim, different type
        input = ones(Float64, 10,2)
        padded = pad(input, 5)
        @test size(padded) == (20,2)
        @test minimum(padded[1,:].== 0.0)
        @test minimum(padded[6,:].== 1.0)
        @test eltype(padded) == eltype(input)

        # two meta dims, different type
        input = ones(ComplexF64, 10,5,3)
        padded = pad(input, 5)
        @test size(padded) == (20,5,3)
        @test minimum(padded[1,:,:].== 0.0)
        @test minimum(padded[6,:,:].== 1.0)
        @test eltype(padded) == eltype(input)

        # test the gradient is just an indicator with the right offset
        i = 6;
        grad = gradient(x->pad(x, 5)[i,1,1],randn(5,5,2))[1]
        @test size(grad)==(5,5,2)
        @test grad[1,1,1] ==1
    end

    @testset "padding 2D" begin
        # no meta dims
        input = ones(Float32, 10,10)
        padded = pad(input, (5,5))
        @test size(padded) == (20,20)
        @test padded[1,1]== 0.0
        @test padded[6,6]== 1.0
        @test eltype(padded) == eltype(input)

        # one meta dim, different type
        input = ones(Float64, 10,10,2)
        padded = pad(input, (5,5))
        @test size(padded) == (20,20,2)
        @test minimum(padded[1,1,:].== 0.0)
        @test minimum(padded[6,6,:].== 1.0)
        @test eltype(padded) == eltype(input)

        # two meta dims, different type
        input = ones(ComplexF64, 10,10,5,3)
        padded = pad(input, (5,5))
        @test size(padded) == (20,20,5,3)
        @test minimum(padded[1,1,:,:].== 0.0)
        @test minimum(padded[6,6,:,:].== 1.0)
        @test eltype(padded) == eltype(input)

        # test the gradient is just an indicator with the right offset
        i = 6; j=7
        grad = gradient(x -> vcat(x, randn(5,5,5,2))[i,1,1,1], randn(5,5,5,2))[1]
        grad = gradient(x -> pad(x, (5,6))[i,j,1,1], randn(5,5,5,2))[1]
        @test size(grad)==(5,5,5,2)
        @test grad[1,1,1,1] ==1
    end
end
