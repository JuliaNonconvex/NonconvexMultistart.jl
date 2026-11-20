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
