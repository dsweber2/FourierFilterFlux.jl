import NNlib.relu
relu(x::C) where C <: Complex = real(x) > 0 ? x : C(0)

# ways to convert between gpu and cpu
import Flux.functor
function functor(cft::ConvFFT{D,OT,F,A,V,PD,P,T,An}) where {D,OT,F,A,V,PD,P,T,An}
    return (cft.weight, cft.bias,
            cft.fftPlan), y -> ConvFFT{D,OT,F,typeof(y[1]),typeof(y[2]),PD,typeof(y[3]),T,An}(cft.σ, y[1],y[2],
                                                              cft.bc, y[3],
                                                              cft.analytic)
end
# import Flux.gpu

# import CUDA.cu
# TODO this is somewhat kludgy, not sure why cu was converting these back
function CUDA.cu(P::FFTW.rFFTWPlan)
    return plan_rfft(cu(zeros(real(eltype(P)), P.sz)), P.region)
end
CUDA.cu(P::CUFFT.rCuFFTPlan) = P


function CUDA.cu(P::FFTW.cFFTWPlan)
    return plan_fft(cu(zeros(eltype(P), P.sz)), P.region)
end
CUDA.cu(P::CUFFT.cCuFFTPlan) = P


function Flux.trainable(CFT::ConvFFT{A,B,C,D,E,F,G,true}) where {A,B,C,D,E,F,G}
    (CFT.weight, CFT.bias)
end
function Flux.trainable(CFT::ConvFFT{A,B,C,D,E,F,G,false}) where {A,B,C,D,E,F,G}
    tuple()
end


"""
jld doesn't like the pointers required by FFTW or CuArray for fft plans, so
this creates a version which can be saved via jld. The jank format I'm using is
a tuple listing the typeo of the input (eg CuArray{Float32,3}), the input size,
and fft region.

"""
function formatJLD(cft::ConvFFT{D,OT,F,A,V,PD,P,T,An}) where {D,OT,F,A,V,PD,P,T,An}
    newPlan = formatJLD(cft.fftPlan)
    newWeight = cft.weight |> cpu
    newBias = cft.bias |> cpu
    ConvFFT{D,OT,F,typeof(newWeight),typeof(newBias),PD,typeof(newPlan),T,An}(cft.σ, newWeight, newBias, cft.bc,
                                        newPlan, cft.analytic)
end

function formatJLD(pl::Tuple)
    return ([formatJLD(x) for x in pl]...,)
end
function formatJLD(pl::AbstractFFTs.Plan)
    ArrayType = (typeof(pl) <: CUFFT.CuFFTPlan) ? CuArray : Array
    return (ArrayType{eltype(pl),ndims(pl)}, size(pl), pl.region)
end
formatJLD(p) = p
"""
    weights = originalDomain()
given a ConvFFT, get the weights as represented in the time domain. optionally, apply a function σ to each pointwise afterward
"""
function originalDomain(cv; σ=identity)
    σ.(irfft(cpu(cv.weight), size(cv.fftPlan, 1), (1, 2)))
end

function getBatchSize(c::C) where {C <: ConvFFT}
    if typeof(c.fftPlan) <: Tuple
        return c.fftPlan[2][end]
    else
        return c.fftPlan.sz[end]
    end
end

function getBatchSize(c::ConvFFT{D,OT,A,B,C,PD,P}) where {D,OT,A,B,C,PD,P <: Tuple}
    if typeof(c.fftPlan[1]) <: Tuple
        return c.fftPlan[1][2][end]
    else
        return c.fftPlan[1].sz[end]
    end
end

function adapt(Atype, x::T) where T <: CUDA.CUFFT.CuFFTPlan
    transformSize = x.osz
    dataSize = x.sz
    if dataSize != transformSize
        # this is an rfft, since the dimension isn't preserved
        newX = plan_rfft(zeros(dataSize), x.region)

    else
        newX = plan_fft(zeros(dataSize), x.region)
    end
    return newX
end
function fromRestrictLocs(restrict, z, i)
    if typeof(restrict[i]) <: Colon
        return 1:size(z, i)
    else
        return restrict[i]
    end
end
@recipe function f(x,y,cv::ConvFFT{2}; vis=1, dispReal=false,
                   apply=abs, restrict=(Colon(), Colon()))
    restrict = (restrict..., vis)
    w = cv.weight
    z = dispReal ?
        apply.(irfft(cpu(w), size(cv.fftPlan, 1),
                     (1, 2)))[restrict...] :
                         apply.(ifftshift(cpu(w), 2))[restrict...]
    (x, y, z)
end
@recipe function f(cv::ConvFFT{2}; vis=1, dispReal=false,
                   apply=abs, restrict=(Colon(), Colon()))
    restrict = (restrict..., vis)
    w = cv.weight
    z = dispReal ?
                apply.(ifftshift(irfft(cpu(w), size(cv.fftPlan, 1),
                                  (1, 2)), (1, 2)))[restrict...] :
                apply.(cpu(w))[restrict...]
    xSz = fromRestrictLocs(restrict, z, 2)
    x = dispReal ? xSz :
               range(xSz[1] - size(w, 2) / 2, stop=xSz[2] + size(w, 2) / 2,
                     length=length(xSz))
    y = fromRestrictLocs(restrict, z, 1)
    (x, y, z)
end

@recipe function f(x, cv::ConvFFT{1}; vis=1, dispReal=false,
                   apply=abs, restrict=(Colon(), vis))
    w = cv.weight
    if typeof(cv.fftPlan) <: Tuple
        origSize = size(cv.fftPlan[1], 1)
    else
        origSize = size(cv.fftPlan, 1)
    end

    z = dispReal ?
                apply.(irfft(cpu(w), origSize,
                             (1,)))[restrict...] :
                apply.(cpu(w))[restrict...]
    (x, z)
end
@recipe function f(cv::ConvFFT{1}; vis=1, dispReal=false,
                   apply=abs, restrict=(Colon(), vis))
    w = cv.weight
    if typeof(cv.fftPlan) <: Tuple
        origSize = size(cv.fftPlan[1], 1)
    else
        origSize = size(cv.fftPlan, 1)
    end

    z = dispReal ?
                apply.(irfft(cpu(w), origSize,
                             (1,)))[restrict...] :
                apply.(cpu(w))[restrict...]

    x = 1:size(z, 1)
    (x, z)
end


"""
    positive_glorot_uniform(dims...)
same idea as a glorot_uniform, but limited to strictly positive entries.
"""
positive_glorot_uniform(dims...) =
    (rand(Float32, dims...) .* sqrt(2.0f0 / sum(Flux.nfan(dims...))))

"""
    uniform_perturbed_gaussian(dims...)
If there are ``n`` total entries in the matrix, each entry is gaussian distributed with a mean of ``¹/ₙ`` and a standard deviation of ``\\frac{1}{10·n}``
"""
function uniform_perturbed_gaussian(dims...)
    netSize = prod(dims)
    A = 1 ./ netSize  .+ randn(dims) ./ netSize / 10;
    A = Float32.(A ./ norm(A))
end
"""
    iden_perturbed_gaussian(dims...)
an identity along the diagonal with Gaussian deviations of standard deviation ``\\frac 1 {100}`` everywhere
"""
function iden_perturbed_gaussian(dims...) # only works for the 2d case
    m = minimum(dims)
    netSize = prod(dims)
    if m == dims[2]
        return [I; zeros(Float32, dims[1] - m, dims[2])] .+ randn(Float32, dims) ./ 100
    else
        return [I zeros(Float32, dims[1], dims[2] - m)] .+ randn(Float32, dims) ./ 100
    end
end
# doubly stochastic matrix (Probably more work than it's worth)
