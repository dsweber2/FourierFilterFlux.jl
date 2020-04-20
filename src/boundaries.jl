# various types of boundary conditions
abstract type ConvBoundary end
# no adjustments made
struct Periodic <: ConvBoundary end
# padded with zeros
struct Pad{N} <: ConvBoundary 
    padBy::NTuple{N,Int}
end
# as in the DCT2 TODO: implement this in a way that is as space efficient as a DCT2, instead of doubling things
struct Symmetric <: ConvBoundary end

# size methods. given the size from the fft, return the size of the input
function effectiveSize(sz, boundary::Pad{N}) where N
    return ([sz[1]-2*p for p in l.boundary.padBy]..., )
end

function effectiveSize(sz, boundary::Periodic) where N
    return ([sz[1]-2*p for p in l.boundary.padBy]..., )
end



# padding methods
for (TYPE, CONVERT) in (
    (AbstractArray, identity), 
    (CuArray, cu)
)
    @eval begin
        # 1D padding
        function pad(x::$TYPE{T, N}, padBy::Union{<:Integer, NTuple{1, <:Integer}}) where {T, N}
            szx = size(x)
            padded = cat($CONVERT(zeros(T, padBy[1], szx[2:end]...)),
                             x,
                             $CONVERT(zeros(T, padBy[1], szx[2:end]...)), dims=(1,))
            return padded
        end

        # 2D padding
        function pad(x::$TYPE{T, N}, padBy::NTuple{2,<:Integer}) where {T, N}
            szx = size(x)
            firstRow = cat($CONVERT(zeros(T, padBy...,  szx[3:end]...)),
                           $CONVERT(zeros(T, padBy[1], szx[2:end]...)),
                           $CONVERT(zeros(T, padBy...,  szx[3:end]...)), dims=2)
            secondRow = cat($CONVERT(zeros(T, szx[1] , padBy[2], szx[3:end]...)),
                            x,
                            $CONVERT(zeros(T, szx[1] , padBy[2], szx[3:end]...)),dims=(2,))
            thirdRow = cat($CONVERT(zeros(T, padBy...,  szx[3:end]...)),
                           $CONVERT(zeros(T, padBy[1], szx[2:end]...)),
                           $CONVERT(zeros(T, padBy...,  szx[3:end]...)), dims=(2,)) 
            return cat(firstRow, secondRow, thirdRow, dims=(1,))
        end
    end
end

Zygote.@adjoint function pad(x, padBy::Union{<:Integer,NTuple{1, <:Integer}})
    return pad(x,padBy), function(Δ)
        axΔ = axes(Δ)
        return (Δ[(1+padBy[1]):(end-padBy[1]), axΔ[2:end]...],
                nothing)
    end
end
