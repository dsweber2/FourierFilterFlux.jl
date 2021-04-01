using Documenter, FourierFilterFlux
ENV["GKSwstype"] = "nul"
using Pkg
Pkg.build("GR")
makedocs(sitename="FourierFilterFlux.jl",
         pages=[
             "Install" => "installation.md",
             "ConvFFT" => [
                 "Core Type" => "coreType.md",
                 "Initialization" => "init.md",
                 "Boundary Conditions" => "bound.md"
             ],
             "Built-in Constructors" => "constructors.md"
         ])

deploydocs(
    repo="github.com/dsweber2/FourierFilterFlux.jl.git",
)
