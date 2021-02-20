if CUDA.functional()
    @testset "CUDA methods" begin
        w = ConvFFT((100,), nConvDims=1)
        @test cu(w.fftPlan) isa CUFFT.rCuFFTPlan # does cu work on the fft plans when applied directly?
        cw = cu(w)
        @test cw.weight isa CuArray # does cu work on the weights?
        @test cw.fftPlan isa CUFFT.rCuFFTPlan # does cu work on the fftPlan?
        cu(randn(4))
        typeof(cu(w.fftPlan))
        cu(w)
        x = randn(100)
        @test cw(cu(x)) isa CuArray
        @test cw(cu(x)) ≈ cu(w(x)) # CUDA and cpu version get the same result approximately
    end
end
