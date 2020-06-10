@testset "ConvFFT transform" begin
    @testset "ConvFFT 2D" begin
        originalSize = (10,10,1,2)
        tmp = zeros(originalSize);
        init = cu(zeros(originalSize)); init[5,5,1,2] = Float32(1)
        shears = ConvFFT(originalSize,useGpu=true)
        res = shears(init);
        @test size(res) == (10,10,5,1,2)
        # TODO: for the other boundary conditions. This is just periodic
        function minimalTransform(shears, init)
            equivalent = cu(zeros(10, 10, 5, 1, 2));
            for i=1:5
                equivalent[:,:,i,:,:] = irfft(rfft(init,(1,2)) .*
                                              shears.weight[:,:,i], 10, (1,2)) .+ shears.bias[:,i]
            end
            return equivalent
        end
        #@info "" minimum(res), minimum(minimalTransform(shears, init))
        @test minimalTransform(shears, init) ≈ res

        shears = ConvFFT(originalSize, 5, abs, useGpu=true)
        res = shears(init);
        @test abs.(minimalTransform(shears, init)) ≈ res
    end
    @testset "ConvFFT 1D" begin
        originalSize = (10,1,2)
        tmp = zeros(originalSize); tmp
        init = cu(zeros(originalSize)); init[5,1,2] = Float32(1)
        shears = ConvFFT(originalSize, nConvDims=1, useGpu=true,
                         boundary=Pad(-1))
        res = shears(init);
        @test size(res) == (10,5,1,2)
        # TODO: this is only padded
        function minimalTransform(shears, init)
            equivalent = cu(zeros(16, 5, 1, 2));
            for i=1:5
                equivalent[:,i,:,:] = irfft(rfft(pad(init,
                                                     shears.bc.padBy),
                                                 (1,)) .*
                                            shears.weight[:,i], 16, (1,)) .+ shears.bias[:,i]
            end
            return equivalent[4:13, :, :, :]
        end
        @test minimalTransform(shears, init) ≈ res

        shears = ConvFFT(originalSize, 5, abs, useGpu=true, nConvDims=1,
                         boundary=Pad(-1))
        res = shears(init);
        @test abs.(minimalTransform(shears, init)) ≈ res
    end
end
