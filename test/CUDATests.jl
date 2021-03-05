if CUDA.functional()
    @testset "CUDA methods" begin
        w = ConvFFT((100,), nConvDims=1)
        @test cu(w.fftPlan) isa CUFFT.rCuFFTPlan # does cu work on the fft plans when applied directly?
        cw = cu(w)
        @test cw.weight isa CuArray # does cu work on the weights?
        @test cw.fftPlan isa CUFFT.rCuFFTPlan # does cu work on the fftPlan?
        x = randn(100)
        cx = cu(x)
        @test cw(cx) isa CuArray
        @test cw(cx) ≈ cu(w(x)) # CUDA and cpu version get the same result approximately
        ∇cu = gradient(t -> cw(t)[1], cx)[1]
        ∇ = gradient(t -> w(t)[1], x)[1]
        @test ∇ ≈ cpu(∇cu)
        w1 = waveletLayer((100, 1, 1))
        w1(x)
        cw1 = cu(w1)
        @test cw1(cx) ≈ cu(w1(x))
        ∇cu = gradient(t -> abs(cw1(t)[1]), cx)[1]
        ∇ = gradient(t -> abs(w1(t)[1]), x)[1]
        @test ∇ ≈ cpu(∇cu)
    end
end
