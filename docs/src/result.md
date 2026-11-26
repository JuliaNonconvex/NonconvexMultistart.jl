# Result

## Results

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

When the algorithm is a [`DeflatedAlg`](@ref) or a
[`DeflationMultistartAlg`](@ref), the result is a `DeflationResult`
instead; see [Deflation](deflation.md) for its fields.
