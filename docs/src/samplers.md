# Sampler choice



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
