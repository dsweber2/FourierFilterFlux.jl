# various types of boundary conditions
abstract type ConvBoundary end
# no adjustments made
struct Periodic <: ConvBoundary end
# padded with zeros
struct Pad{N} <: ConvBoundary
    padBy::NTuple{N,Int}
end
"""
    Pad(x::Vararg{<:Integer, N}) where N = Pad{N}(x)

N gives the number of dimensions of convolution, while `x` gives the specific amount to pad in each dimension (done on both sides). If the values in `x` are negative, then the support of the filters will be determined automataically
"""
Pad(x::Vararg{<:Integer,N}) where {N} = Pad{N}(x)
import Base.ndims
ndims(p::Pad{N}) where {N} = N

function Base.show(io::IO, p::Pad{N}) where {N}
    print(io, "Pad$(p.padBy)")
end

# as in the DCT2 TODO: implement this in a way that is as space efficient as a DCT2, instead of doubling things
struct Sym <: ConvBoundary end

"""
    originalSize(sz, boundary::ConvBoundary)
Given the size `sz` from the fft, return the size of the input.
"""
function originalSize(sz, boundary::Pad{N}) where {N}
    return sz .- 2 .* boundary.padBy# ([sz[ii]-2*p for p in boundary.padBy]..., )
end

function originalSize(sz, boundary::Periodic) where {N}
    return sz# ([sz[1] for p in boundary.padBy]..., )
end

# TODO: when Sym gets implemented using a CFT, this should shrink
function originalSize(sz, boundary::Sym) where {N}
    return Int.(sz ./ 2) # ([Int(sz[ii]/2) for p in boundary.padBy]..., )
end

"""
    effectiveSize(sz, boundary::ConvBoundary)
Given the input size, figure out the resulting size
"""
function effectiveSize(sz, boundary::Pad{N}) where {N}
    if boundary.padBy[1] > 0
        return sz .+ 2 .* boundary.padBy, boundary
    elseif boundary.padBy[1] == 0
        return sz, Periodic
    elseif boundary.padBy[1] < 0
        newBoundary = Pad(([sz[i] >> 2 + 1 for i = 1:N]...,))
        return effectiveSize(sz, newBoundary)[1], newBoundary
    end
end

function effectiveSize(sz, boundary::Periodic) where {N}
    return sz, boundary
end

# TODO: when Sym gets implemented using a CFT, this should shrink
function effectiveSize(sz, boundary::Sym) where {N}
    return 2 .* sz, boundary
end

# apply the boundary condition appropriately to the input matrix. nd is the
# number of dimensions that we're working with
function applyBC(x, bc::Pad, nd)
    usedInds = ([bc.padBy[ii] .+ (1:size(x, ii)) for ii = 1:ndims(bc)]...,)
    return (pad(x, bc.padBy), usedInds)
end

function applyBC(x, bc::Periodic, nd)
    return (x, axes(x)[1:nd])
end

function applyBC(x, bc::Sym, nd)
    flipThisDim = cat(x, reverse(x, dims = nd), dims = nd)
    if nd == 1
        return flipThisDim, axes(x)[1:nd]
    else
        return applyBC(flipThisDim, bc, nd - 1)[1], axes(x)[1:nd]
    end
end


# padding methods
for (TYPE, CONVERT) in ((AbstractArray, identity),
    (CuArray, cu))
    @eval begin
        # 1D padding
        function pad(x::$TYPE{T,N}, padBy::Union{<:Integer,NTuple{1,<:Integer}}) where {T,N}
            szx = size(x)
            padded = cat($CONVERT(zeros(T, padBy[1], szx[2:end]...)),
                x,
                $CONVERT(zeros(T, padBy[1], szx[2:end]...)), dims = (1,))
            return padded
        end

        # 2D padding
        function pad(x::$TYPE{T,N}, padBy::NTuple{2,<:Integer}) where {T,N}
            szx = size(x)
            firstRow = cat($CONVERT(zeros(T, padBy..., szx[3:end]...)),
                $CONVERT(zeros(T, padBy[1], szx[2:end]...)),
                $CONVERT(zeros(T, padBy..., szx[3:end]...)), dims = 2)
            secondRow = cat($CONVERT(zeros(T, szx[1], padBy[2], szx[3:end]...)),
                x,
                $CONVERT(zeros(T, szx[1], padBy[2], szx[3:end]...)), dims = (2,))
            thirdRow = cat($CONVERT(zeros(T, padBy..., szx[3:end]...)),
                $CONVERT(zeros(T, padBy[1], szx[2:end]...)),
                $CONVERT(zeros(T, padBy..., szx[3:end]...)), dims = (2,))
            return cat(firstRow, secondRow, thirdRow, dims = (1,))
        end
    end
end

Zygote.@adjoint function pad(x, padBy::Union{<:Integer,NTuple{1,<:Integer}})
    return pad(x, padBy),
    function (Δ)
        axΔ = axes(Δ)
        return (Δ[(1+padBy[1]):(end-padBy[1]), axΔ[2:end]...],
            nothing)
    end
end


Zygote.@adjoint function applyBC(x, bc::Periodic, nd)
    return applyBC(x, bc, nd), function (Δ)
        xbc, usedInds = Δ
        return xbc, nothing, nothing
    end
end

Zygote.@adjoint function applyBC(x, bc::Sym, nd)
    return applyBC(x, bc, nd),
    function (Δ)
        xbc, usedInds = Δ
        ax = axes(x)
        aΔ = axes(xbc)
        return xbc[ax[1:nd]..., aΔ[nd+1:end]...], nothing, nothing
    end
end
