Flux.@functor ConvFFT

function Flux.trainable(CFT::ConvFFT{A,B,C,D,E,F,G,true}) where {A,B,C,D,E,F,G}
    (CFT.weight, CFT.bias)
end
function Flux.trainable(::ConvFFT{A,B,C,D,E,F,G,false}) where {A,B,C,D,E,F,G}
    tuple()
end
