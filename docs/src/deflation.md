# Deflation

Deflation is a technique for exploring multiple distinct local minimizers of
a non-convex problem. Instead of restarting the optimization from random
starting points and hoping for different solutions, deflation solves the
problem repeatedly while adding a constraint that makes all previously found
solutions infeasible, so every converged sub-problem is guaranteed to find a
different solution.

`DeflatedAlg` and `DeflationMultistartAlg` implement the general deflation
constraint of [Tarek and Huang (2022)](https://arxiv.org/abs/2201.11926).
The key idea of the paper is that deflation can be expressed as a simple
*change of the problem formulation* — a non-convex inequality constraint —
rather than a change of the optimization algorithm, so any
constraint-capable optimizer of the Nonconvex ecosystem can be used as the
sub-solver.

## Theory

This section summarizes the deflation constraint method implemented by this
package, following Tarek and Huang (2022).

### The problem

Consider the nonlinear program

```math
\begin{aligned}
\min_{x \in \mathbb{R}^n} \; & f(x) \\
\text{s.t.} \quad & c(x) = 0, \\
& l \le x \le u,
\end{aligned}
```

where the objective $f : \mathbb{R}^n \to \mathbb{R}$ and the equality
constraints $c : \mathbb{R}^n \to \mathbb{R}^m$ ($m \le n$) are twice
continuously differentiable and $l, u \in \mathbb{R}^n$ are finite lower and
upper bounds. General nonlinear inequality constraints can be converted to
equality constraints by adding slack variables.

A local minimizer $x^*$ satisfies (under a regularity condition) the
first-order Karush–Kuhn–Tucker (KKT) conditions

```math
\nabla_x L(x^*, \lambda, z^+, z^-) = 0, \qquad c(x^*) = 0, \qquad l \le x^* \le u,
```

together with the dual-feasibility and complementarity conditions, where

```math
L(x, \lambda, z^+, z^-) = f(x) + c(x)^\top \lambda + (x - u)^\top z^+ - (x - l)^\top z^-
```

is the Lagrangian and $\lambda$, $z^+$, $z^-$ are the multipliers of the
equality and bound constraints. Local first- and second-order optimization
algorithms converge to whichever KKT point lies in the basin of attraction
of the initial guess, and naive multi-start strategies can waste many
restarts re-converging to the same solution.

### Classical deflation for nonlinear systems

Deflation originates from nonlinear root finding
([Farrell, Beentjes and Birkisson, 2016](https://arxiv.org/abs/1603.00809)).
After a root $x_1$ of a nonlinear system $F(x) = 0$ has been found, the
*deflation operator*

```math
M(x; x_1) = m(x; x_1) \, I, \qquad m(x; x_1) = \lVert x - x_1 \rVert^{-p} + \sigma,
```

for some power $p$ and shift $\sigma$ ($I$ the identity), is applied to
obtain the deflated system $M(x; x_1) F(x) = 0$. Since
$\lVert M(x; x_1) F(x) \rVert$ does not vanish as $x \to x_1$, a
Newton-type method cannot reconverge to $x_1$ and is forced to find a
different root. For several known roots, the operators multiply,
$\prod_{k=1}^{\tilde{K}} M(x; x_k) F(x) = 0$ (summation can be used instead
of multiplication for numerical stability).

### Deflating the KKT conditions is intrusive

To apply deflation to optimization, one can deflate the optimality
conditions themselves: write (most of) the KKT conditions as a nonlinear
system $F(x, \lambda) = 0$ and solve the deflated system
$M(x; x_1) F(x, \lambda) = 0$ while taking descent steps. However, this
requires a Newton-like solver on the deflated KKT system, whose Jacobian is
no longer symmetric — the top-left block is the Hessian of the Lagrangian
scaled by the deflation operator, while off-the-shelf NLP solvers such as
IPOPT assume it to be symmetric. This is why
[Papadopoulos, Farrell and Surowiec (2021)](https://arxiv.org/abs/2004.11797)
had to implement a specialized primal-dual interior-point method from
scratch. The constraint formulation below avoids this entirely: it changes
the *problem*, not the *algorithm*.

### Deflation as a constraint (Lemma 4.1)

Introduce an auxiliary variable $y \ge 0$ and add the deflation constraint
$m(x; x_1) \le y$ to the NLP:

```math
\min_{x, y} \; f(x) \quad \text{s.t.} \quad c(x) = 0, \quad m(x; x_1) \le y, \quad l \le x \le u.
```

**Lemma 4.1 (Tarek and Huang).** If $(x^*, y^*)$ is a regular KKT point of
this problem with finite $y^*$, and $m$ is bounded from below, then $x^*$ is
a regular KKT point of the original NLP and $x^* \neq x_1$.

*Sketch of proof.* Let $\eta \ge 0$ be the multiplier of the deflation
constraint $m(x; x_1) - y \le 0$. Differentiating the Lagrangian with
respect to $y$ gives $-\eta$, so stationarity forces $\eta = 0$, and the
stationarity conditions of the deflated problem reduce to those of the
original NLP. Since the constraints of the original problem are a subset of
the deflated ones, $x^*$ is feasible and the complementarity conditions
carry over. Finally, because $m(x; x_1) \to \infty$ as $x \to x_1$ while
$m(x^*) \le y^* < \infty$, it follows that $x^* \neq x_1$. The same
argument extends to inequality- and conic-constrained NLPs.

For several known solutions $x_1, \dots, x_{\tilde{K}}$, the constraint
becomes the sum

```math
\sum_{k=1}^{\tilde{K}} m(x; x_k) \le y .
```

This is exactly the constraint added by [`DeflatedAlg`](@ref) after each
found solution, with
$m(x; x_k) = \max(d(x, x_k) - r, 0)^{-p} + \sigma$ and $y$ bounded above by
`y_upper` (the finite upper bound guarantees that $y^*$ is finite).

### The fixed-bound formulation (Lemma 4.2)

Alternatively, the auxiliary variable can be replaced by a large finite
constant $M$:

```math
\min_x \; f(x) \quad \text{s.t.} \quad c(x) = 0, \quad m(x; x_1) \le M, \quad l \le x \le u.
```

**Lemma 4.2 (Tarek and Huang).** If $x^*$ is a regular KKT point at which
the deflation constraint holds at a *strict* inequality, and $m$ is bounded
from below, then $x^*$ is a regular KKT point of the original NLP and
$x^* \neq x_1$.

*Sketch of proof.* By complementarity slackness, strict feasibility of the
deflation constraint forces its multiplier $\eta = 0$; the rest of the proof
is as in Lemma 4.1.

This is the formulation used by [`DeflationMultistartAlg`](@ref) with
$M$ = `y_upper`: because no auxiliary variable is added, the multistart
sampler explores only the original variables.

### Deflation is a minimum-distance constraint

The deflation constraint is equivalent to a minimum-distance constraint
(Section 4.5 of the paper): with finite, positive $m$ and $y > \sigma$,

```math
m(x; x_1) = \lVert x - x_1 \rVert^{-p} + \sigma \le y
\qquad\Longleftrightarrow\qquad
\lVert x - x_1 \rVert^{p} \ge z, \quad z = \frac{1}{y - \sigma}.
```

Deflation therefore simply forbids a neighbourhood of every known solution.
If the solver drives $y \to \infty$ — or, in the fixed-bound formulation,
if the deflation constraint is satisfied at equality — deflation has failed
to push the solution away from the known ones.

### Distance measures

The distance $d(x, x_k)$ in the deflation function can be adapted to the
problem (Section 4.6 of the paper):

- a power $p$ of an $\ell_q$-norm distance instead of a power of the
  Euclidean norm;
- a positive semi-definite weight matrix $Q$, giving the generalized
  Euclidean distance $\sqrt{(x - x_k)^\top Q (x - x_k)}$; a diagonal $Q$
  weighs the deviation terms per coordinate (the `weight` option);
- a problem-specific distance, e.g. in a transformed space such as
  density-filtered designs in topology optimization, the Kullback–Leibler
  divergence between variational distributions, or neuron activations
  instead of weights (the `distance` option);
- a radius $r$, giving
  $m(x; x_k) = \max(\lVert x - x_k \rVert - r, 0)^{-p}$: all points within
  distance $r$ of a known solution are treated as identical, so even if
  deflation fails to find a new minimizer, the returned solution is
  guaranteed to lie outside the $r$-ball. This is useful in high
  dimensions, where two solutions can be numerically different but
  practically identical (the `radius` option).

### Failure modes and how the implementation detects them

Deflation is not a global optimization algorithm, and the following caveats
(Section 4.7 of the paper) apply. The implementation guards against the
first two:

1. **The deflation effect may be too weak.** The solver can still approach
   $x_1$ asymptotically with $y \to \infty$ (or with the constraint active
   at $M$). By the minimum-distance equivalence above, the constraint
   excludes a hyper-ball of radius $(y_{\mathrm{upper}} - \tilde{K}
   \sigma)^{-1/p}$ around each known solution. Every new solution is
   therefore validated against this exclusion radius: a solution closer
   than
   $\max(\mathrm{tol}, r + 100 \cdot (y_{\mathrm{upper}} - \tilde{K}
   \sigma)^{-1/p})$ to a known solution is rejected as a deflation failure
   and the run stops early with `converged = false`.
2. **Optimization happens in finite precision.** A failed deflation
   converges to a point with a large-but-finite constraint value rather
   than an infinite one, so the finiteness of $y$ alone is not a reliable
   distinctness test; the distance-based test above is used instead.
3. **Convergence to a new minimizer is not guaranteed.** Deflation
   guarantees distinctness *if* the sub-problem converges, but a local
   solver started from the same initial point may fail to escape to a new
   basin. Deflating one solution can also push the optimizer away from
   other nearby local minima.
4. **Hyperparameter sensitivity.** The power $p$ and the other
   hyperparameters of the distance measure must be tuned for the problem at
   hand.

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
each round. The fixed-bound formulation (Lemma 4.2 above) is used for the
deflation constraint, so no auxiliary variable is added and the multistart
sampler explores only the original variables.

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
