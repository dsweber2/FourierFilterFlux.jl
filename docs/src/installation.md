# Installation

At the moment this package is unregistered and has some unregistered
dependencies, so you will have to do it the hard way. Either press `]` and run
```
(@v9.9) pkg> add https://github.com/dsweber2/ContinuousWavelets.jl.git
(@v9.9) pkg> add https://github.com/dsweber2/Shearlab.jl.git
(@v9.9) pkg> add https://github.com/dsweber2/FourierFilterFlux.jl.git
```
Alternatively, run
```
using Pkg
Pkg.add("")
Pkg.add("https://github.com/dsweber2/ContinuousWavelets.jl.git")
Pkg.add("https://github.com/dsweber2/Shearlab.jl.git")
Pkg.add("https://github.com/dsweber2/FourierFilterFlux.jl.git")
```