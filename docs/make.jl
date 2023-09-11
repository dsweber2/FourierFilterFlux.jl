using Documenter, FourierFilterFlux
ENV["PLOTS_TEST"] = "true"
ENV["GKSwstype"] = "100"
ENV["LINES"] = "9"
ENV["COLUMNS"] = "60"
makedocs(sitename = "FourierFilterFlux.jl",
    authors = "David Weber",
    clean = true,
    format = Documenter.HTML(),
    pages = [
        "Install" => "installation.md",
        "ConvFFT" => [
            "Core Type" => "coreType.md",
            "Initialization" => "init.md",
            "Boundary Conditions" => "bound.md"
        ],
        "Built-in Constructors" => "constructors.md"
    ])
deploydocs(repo = "github.com/dsweber2/FourierFilterFlux.jl.git")
