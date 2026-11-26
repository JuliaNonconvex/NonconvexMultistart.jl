<p align="center">
  <img src="assets/logo.png" width="400" alt="NonconvexMultistart.jl logo">
</p>

[![Documentation (stable)](https://img.shields.io/badge/docs-stable-blue.svg)](https://julianonconvex.github.io/NonconvexMultistart.jl/stable)
[![Documentation (dev)](https://img.shields.io/badge/docs-dev-blue.svg)](https://julianonconvex.github.io/NonconvexMultistart.jl/dev)
[![Build Status](https://github.com/JuliaNonconvex/NonconvexMultistart.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/JuliaNonconvex/NonconvexMultistart.jl/actions)
[![Coverage](https://codecov.io/gh/JuliaNonconvex/NonconvexMultistart.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaNonconvex/NonconvexMultistart.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-Blue-4473c9.svg)](https://github.com/invenia/BlueStyle)
[![JET](https://img.shields.io/badge/%F0%9F%9B%A9%EF%B8%8F_tested_with-JET.jl-233f9a)](https://github.com/aviatesk/JET.jl)

# NonconvexMultistart.jl

Multi-start and hyper-parameter optimization

Hyperopt meta-algorithm wrapping any Nonconvex optimizer with random, LHS, CLHS, Gaussian process and Hyperband samplers. Also provides deflation-based exploration of multiple distinct local minimizers ([Tarek and Huang, 2022](https://arxiv.org/abs/2201.11926)).

NonconvexMultistart.jl is part of the [JuliaNonconvex](https://github.com/JuliaNonconvex) ecosystem built on [NonconvexCore.jl](https://github.com/JuliaNonconvex/NonconvexCore.jl) under the [Nonconvex.jl](https://github.com/JuliaNonconvex/Nonconvex.jl) umbrella.

## Quick start

```julia
using NonconvexMultistart

alg = HyperoptAlg(IpoptAlg())
options = HyperoptOptions(sub_options = IpoptOptions(), sampler = RandomSampler())
result = optimize(model, alg, x0, options = options)
```

To find multiple distinct local minimizers with deflation:

```julia
alg = DeflatedAlg(IpoptAlg())
options = DeflationOptions(sub_options = IpoptOptions(), iters = 10)
result = optimize(model, alg, x0, options = options)
```

or combined with a multistart search:

```julia
alg = DeflationMultistartAlg(HyperoptAlg(IpoptAlg()))
options = DeflationMultistartOptions(
    multi_options = HyperoptOptions(sub_options = IpoptOptions(), iters = 10),
    iters = 4,
)
result = optimize(model, alg, x0, options = options)
```

## Documentation

The full documentation is available at [https://julianonconvex.github.io/NonconvexMultistart.jl/stable](https://julianonconvex.github.io/NonconvexMultistart.jl/stable) (stable) and [https://julianonconvex.github.io/NonconvexMultistart.jl/dev](https://julianonconvex.github.io/NonconvexMultistart.jl/dev) (development).

## Installation

```julia
using Pkg
Pkg.add("NonconvexMultistart")
```
