"""
    weights = originalDomain()
given a ConvFFT, get the weights as represented in the time domain. optionally, apply a function σ to each pointwise afterward
"""
function originalDomain(cv; σ=identity)
    σ.(irfft(cpu(cv.weight), size(cv.fftPlan,1), (1,2)))
end


function adapt(Atype, x::T) where T<:CuArrays.CUFFT.CuFFTPlan
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

# A plotting method to show all the filters
@recipe function f(cv::ConvFFT; vis=1, dispReal=false, apply=abs,restrict=(:,:))
    if dispReal
        apply.(irfft(cpu(cv.weight[:,:,vis]), size(cv.fftPlan,1), (1,2)))[restrict...]
    else
        apply.(cpu(cv.weight[:,:,vis]))[restrict...]
    end
end
