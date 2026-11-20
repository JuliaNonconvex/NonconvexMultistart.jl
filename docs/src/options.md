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
