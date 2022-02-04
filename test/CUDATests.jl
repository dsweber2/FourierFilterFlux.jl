if CUDA.functional()
    @testset "CUDA methods" begin
        w = ConvFFT((100,), nConvDims = 1)
        @test cu(w.fftPlan) isa CUFFT.rCuFFTPlan # does cu work on the fft plans when applied directly?
        cw = cu(w)
        @test cw.weight isa NTuple{N,CuArray} where {N} # does cu work on the weights?
        @test cw.fftPlan isa CUFFT.rCuFFTPlan # does cu work on the fftPlan?
        cw1 = gpu(w)
        @test cw1.weight isa NTuple{N,CuArray} where {N} # does gpu work on the weights?
        @test cw1.fftPlan isa CUFFT.rCuFFTPlan # does gpu work on the fftPlan?
        w1 = cpu(cw)
        @test w1.weight isa NTuple{N,Array} where {N} # does cpu work on the weights?
        @test w1.fftPlan isa FFTW.rFFTWPlan # does cpu work on the fftPlan?
        x = randn(100)
        cx = cu(x)
        @test cw(cx) isa CuArray
        @test cw(cx) ≈ cu(w(x)) # CUDA and cpu version get the same result approximately
        cw(cx)
        ∇cu = gradient(t -> sum(cw(t)), cx)[1]
        ∇ = gradient(t -> sum(w(t)), x)[1]
        @test ∇ ≈ cpu(∇cu)
        w1 = waveletLayer((100, 1, 1))
        cw1 = cu(w1)
        @test cw1(cx) ≈ cu(w1(x))

        CUDA.@allowscalar ∇cu = gradient(t -> abs(cw1(t)[1]), cx)[1]
        CUDA.@allowscalar ∇ = gradient(t -> abs(w1(t)[1]), x)[1]
        @test ∇ ≈ cpu(∇cu)
    end
end
