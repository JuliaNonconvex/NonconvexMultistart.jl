# Multi-start and hyper-parameter optimization in pure Julia

## Description

[Hyperopt.jl](https://github.com/baggepinnen/Hyperopt.jl) is a Julia library that implements a number of hyperparameter optimization algorithms which can be used to optimize the starting point of the optimization. `NonconvexHyperopt.jl` allows the use of the algorithms in `Hyperopt.jl` as meta-algorithms using the `HyperoptAlg` struct.

## Quick start

Given a model `model` and an initial solution `x0`, the following can be used to optimize the model using Hyperopt.
```julia
using NonconvexMultistart

alg = HyperoptAlg(IpoptAlg())
options = HyperoptOptions(sub_options = IpoptOptions(), sampler = RandomSampler())
result = optimize(model, alg, x0, options = options)
```
Hyperopt is an optional dependency of Nonconvex so you need to import it in order to use it. `HyperoptAlg` can wrap any other algorithm in Nonconvex, e.g. `IpoptAlg()`. When the algorithm is a `HyperoptAlg`, the `options` keyword argument must of type `HyperoptOptions`. For more on the options available see below.

## Construct an instance

To construct an instance of the Hyperopt + Ipopt algorithm, use:
```julia
alg = HyperoptAlg(IpoptAlg())
```
`HyperoptAlg` can wrap any other algorithm in Nonconvex, e.g. `NLoptAlg(:LD_MMA)` or `AugLag()`.

## Options

The options keyword argument to the `optimize` function shown above must be an instance of the `HyperoptOptions` struct when the algorihm is a `HyperoptAlg`. To specify options, use keyword arguments in the constructor of `HyperoptOptions`. The `sampler` keyword argument determines the sampling algorithm used to propose new starting points in the multi-start procedure. The `sub_options` keyword argument can be used to pass in the options for the sub-optimizer. There are 2 different ways to pass the sub-options depending on the sampler type.

The `sampler` argument can be of type:
1. `RandomSampler`
2. `LHSampler`
3. `CLHSampler`
4. `Hyperband`

When optimizing the starting point, the upper and lower bounds on the initial solution must be finite, or finite bounds must be passed in to the `options` constructor. All the options that can be passed to the `HyperoptOptions` constructor are listed below:

- `sub_options`: the options of the sub-optimizer, or a function from the number of iterations to the sub-optimizer options when the sampler is a `Hyperband`.
- `lb` and `ub`: the lower and upper bounds of the starting point search. When `nothing` (the default), the bounds of the model variables are used.
- `iters`: the number of candidate starting points that are evaluated, `40` by default.
- `searchspace_size`: the number of potential starting points that are generated. It defaults to `iters`. `LHSampler` and `CLHSampler` require `searchspace_size == iters`.
- `sampler`: the starting point sampler, `RandomSampler()` by default.
- `ctol`: the infeasibility tolerance for accepting a solution as feasible, `1e-5` by default.
- `keep_all`: when `true` (the default), all the sub-problem results are kept in the `results` field of the returned `HyperoptResult`; when `false`, only the best result is returned.

### Results

When multiple starting points are searched, the result of `optimize` is a `HyperoptResult` with the following fields:
- `minimum`: the optimal value among all searches.
- `minimizer`: the optimal solution found.
- `results`: all the search results (when `keep_all = true`).
- `optimal_ind`: the index of the optimal solution in `results`.

For example:
```julia
r = optimize(model, alg, x0, options = options)
r.minimum
r.minimizer
r.results
r.optimal_ind
```

### Sampler choice

#### RandomSampler, LHSampler and CLHSampler

All the sampler constructors are functions defined in NonconvexMultistart wrapping the Hyperopt alternatives to define defaults. All the arguments that can be passed to the sampler constructor can be found in the [Hyperopt documentation](https://github.com/baggepinnen/Hyperopt.jl#details). Example:
```julia
options = HyperoptOptions(sub_options = IpoptOptions(), sampler = LHSampler())
```

> `GPSampler` was removed in NonconvexMultistart 0.2.0 because Hyperopt removed
> `GPSampler` upstream (Hyperopt >= 0.5). Calling `GPSampler()` raises an error.
> Use `RandomSampler`, `LHSampler` or `CLHSampler` instead.

#### Hyperband

The [Hyperband algorithm](https://github.com/baggepinnen/Hyperopt.jl#hyperband) in Hyperopt requires a different way to pass in the sub-options. The Hyperband algorithm tries to optimize the allocation of resources. The `sub_options` argument must be a function with input as the "resources" and output as the sub-solver options. The `Hyperband` constructor accepts 3 arguments:
1. The maximum resources `R`
2. `η` which roughly determines the proportion of trials discarded between each round of successive halving
3. `inner` which specifies an inner sampler of type `RandomSampler`, `LHSampler` or `CLHSampler`.

Example:
```julia
options = HyperoptOptions(
    sub_options = max_iter -> IpoptOptions(max_iter = max_iter), 
    sampler = Hyperband(R=100, η=3, inner=RandomSampler()),
)
```
