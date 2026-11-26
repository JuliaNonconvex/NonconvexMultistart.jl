# Deflation

Deflation is a technique for exploring multiple distinct local minimizers of
a non-convex problem. Instead of restarting the optimization from random
starting points and hoping for different solutions, deflation solves the
problem repeatedly from the *same* initial point while adding a constraint
that makes all previously found solutions infeasible, so every converged
sub-problem is guaranteed to find a different solution.

`DeflatedAlg` implements the general deflation constraint of
[Tarek and Huang (2022)](https://arxiv.org/abs/2201.11926). The key idea of
the paper is that deflation can be expressed as a simple *change of the
problem formulation* — a non-convex inequality constraint — rather than a
change of the optimization algorithm, so any constraint-capable optimizer of
the Nonconvex ecosystem can be used as the sub-solver.

## Construct an instance

To construct a deflation meta-algorithm wrapping Ipopt, use:

```julia
using NonconvexMultistart, NonconvexIpopt

alg = DeflatedAlg(IpoptAlg())
options = DeflationOptions(sub_options = IpoptOptions(), iters = 10)
result = optimize(model, alg, x0, options = options)
```

`DeflatedAlg` can wrap any optimizer of the Nonconvex ecosystem that
supports nonlinear inequality constraints, e.g. `IpoptAlg()`, `MMA87()`,
`MMA02()` or `AugLagAlg()`.

## Combining deflation with multistart

Plain deflation re-solves every sub-problem from the same initial point, so
a local sub-solver may repeatedly converge to the boundary of the exclusion
region around a known solution instead of a new minimizer.
`DeflationMultistartAlg` combines deflation with an existing multistart
meta-algorithm: every deflation round runs a full multistart search of the
deflated sub-problem, so the starting point is explored by the sampler while
the deflation constraints keep the solutions distinct.

```julia
alg = DeflationMultistartAlg(HyperoptAlg(IpoptAlg()))
options = DeflationMultistartOptions(
    multi_options = HyperoptOptions(sub_options = IpoptOptions(), iters = 10),
    iters = 4,
)
result = optimize(model, alg, x0, options = options)
```

Here `options.iters` is the maximum number of distinct minimizers to find
(one deflation round per solution), while `multi_options.iters` is the
number of candidate starting points evaluated by the multistart sampler in
each round. The fixed-bound formulation (Lemma 4.2 of the paper) is used for
the deflation constraint, so no auxiliary variable is added and the
multistart sampler explores only the original variables.

## How it works

After `k` solutions `x₁, …, xₖ` have been found, the `(k+1)`-th
sub-problem minimizes the original objective over the original constraints
plus one extra deflation constraint:

```math
\sum_{j=1}^{k} \left( \max(d(x, x_j) - r, 0)^{-p} + \sigma \right) \leq y,
\qquad 0 \leq y \leq y_{\mathrm{upper}},
```

where `d(x, x_j)` is the distance between `x` and the `j`-th found
minimizer (Euclidean by default, or weighted/custom, see below), `y` is an
auxiliary variable, and `p`, `σ` and `r` are hyperparameters. Since the
deflation terms blow up to `∞` at the known solutions, no converged
sub-problem can return one of them: each solution is pushed away from all
previously found ones (Lemma 4.1 of the paper). The auxiliary variable `y`
is initialized at a feasible value; if deflation fails and the sub-solver
converges to the boundary of the exclusion region around a known solution,
the iteration is reported as unsuccessful.

`DeflationMultistartAlg` uses the alternative fixed-bound formulation
(Lemma 4.2 of the paper), with the constraint
`∑ⱼ max(d(x, xⱼ) - r, 0)⁻ᵖ + σ ≤ y_upper` and no auxiliary variable.

## Options

All the options that can be passed to the `DeflationOptions` constructor:

- `sub_options`: the options of the sub-optimizer, or a function from the
  deflation iteration index to the sub-optimizer options.
- `iters`: the maximum number of distinct minimizers to find, `10` by
  default.
- `p`: the deflation power in `d(x, x_j)⁻ᵖ`, `2.0` by default.
- `sigma`: the shift `σ` added to every deflation term, `0.0` by default.
- `radius`: the radius `r` of the hyper-ball around each known solution
  inside which all points are deflated identically, `0.0` by default. With
  `radius > 0`, every new solution is guaranteed to be at least `radius`
  away from all previously found ones, even if deflation fails. This is
  particularly useful in high dimensions, where two solutions can be
  numerically different but practically identical.
- `distance`: an optional custom distance function `d(x, x_j)`, e.g. to
  deflate in a transformed space (density-filtered designs, KL divergence
  between variational distributions, neuron values instead of weights, …).
- `weight`: an optional positive semi-definite matrix `Q` giving the
  generalized Euclidean distance `sqrt((x - x_j)' Q (x - x_j))`. A diagonal
  `Q` weighs the deviation terms per coordinate. Only one of `distance` and
  `weight` can be set.
- `y_upper`: the finite upper bound on the deflation variable `y`, `1e8` by
  default. A larger `y_upper` allows solutions closer to the known ones to
  be considered distinct.
- `ctol`: the infeasibility tolerance for accepting a sub-problem solution
  as feasible, `1e-5` by default.
- `tol`: the distance tolerance used to decide that a new solution is
  identical to a previously found one, `1e-6` by default.
- `keep_all`: when `true` (the default), all the sub-problem results are
  kept in the `results` field of the returned result; when `false`, only the
  found minimizers are returned.

`DeflationMultistartOptions` has the same deflation hyperparameters plus
`multi_options`, the options of the wrapped multistart algorithm; in that
variant `y_upper` is the fixed bound `M` of the deflation constraint.

## Result

The result of `optimize` with a `DeflatedAlg` or a
`DeflationMultistartAlg` is a `DeflationResult` with the following fields:

- `minimum`: the best objective value among the found minimizers.
- `minimizer`: the found minimizer with the best objective value.
- `minimizers`: all the distinct feasible minimizers found, in the order
  they were found.
- `results`: all the sub-problem results when `keep_all = true`, `nothing`
  otherwise.
- `converged`: `true` if all `iters` deflation iterations produced feasible
  solutions distinct from all previously found ones.
- `niters`: the number of sub-problems actually solved before stopping.

For example:

```julia
r = optimize(model, alg, x0, options = options)
r.minimum
r.minimizer
r.minimizers
r.converged
r.niters
```

## Limitations

Deflation guarantees that *if* a sub-problem converges, its solution is
different from all previously found ones; it does not guarantee that
convergence to a new minimizer will happen. In particular, a local
sub-solver started from the same initial point every time may simply
converge to the boundary of the exclusion region around a known solution,
in which case the run stops early and `converged` is `false`. Combining
deflation with multistart (`DeflationMultistartAlg`) mitigates this, since
every deflation round explores multiple starting points. Tuning `p`,
`radius`, `y_upper` and the distance measure, or starting the deflation
iterations from different initial points, can also help in practice.
