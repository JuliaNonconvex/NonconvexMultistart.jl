# Deflation meta-algorithms: the general deflation constraint method of
# Mohamed Tarek and Yijiang Huang, "Simplifying deflation for non-convex
# optimization with applications in Bayesian inference and topology
# optimization" (arXiv:2201.11926, 2022).
#
# Instead of deflating the KKT conditions of the NLP (which requires a
# specialized optimization algorithm), a non-convex inequality constraint is
# added to the problem formulation and any constraint-capable optimizer is
# used to solve the deflated sub-problems. Every sub-problem that converges
# is guaranteed to find a solution that is different from all previously
# found ones.
#
# [`DeflatedAlg`](@ref) re-solves the deflated sub-problems from the same
# initial point using a single sub-solver and the auxiliary-variable
# formulation of the paper (Lemma 4.1), while [`DeflationMultistartAlg`](@ref)
# solves each deflated sub-problem with an existing multistart
# meta-algorithm using the fixed-bound formulation (Lemma 4.2), so the
# starting point is explored by the multistart sampler as well.

"""
    AbstractDeflationOptions

Common supertype of [`DeflationOptions`](@ref) and
[`DeflationMultistartOptions`](@ref).
"""
abstract type AbstractDeflationOptions end

"""
    DeflatedAlg(sub_alg)

Deflation meta-algorithm for finding multiple distinct local minimizers of a
non-convex problem. It repeatedly solves the problem with the sub-solver
`sub_alg`, starting from the same initial point each time, while adding a
deflation (inequality) constraint that pushes the solver away from all
previously found solutions, as proposed in:

> Mohamed Tarek and Yijiang Huang. "Simplifying deflation for non-convex
> optimization with applications in Bayesian inference and topology
> optimization", arXiv:2201.11926 (2022).

`sub_alg` can be any optimizer of the Nonconvex ecosystem that supports
nonlinear inequality constraints. After `k` solutions `x₁, …, xₖ` have been
found, the `(k+1)`-th sub-problem adds the variable `y ∈ [0, y_upper]` and
the constraint

```math
∑_{j=1}^{k} (max(d(x, x_j) - radius, 0)^{-p} + σ) ≤ y
```

where `d(x, x_j)` is the (possibly weighted or user-defined) distance
between `x` and the `j`-th found minimizer `x_j`, and `p`, `σ` and `radius`
are set through [`DeflationOptions`](@ref). A solution of the deflated
sub-problem with a finite `y` is a KKT point of the original problem that
is distinct from `x₁, …, xₖ` (Lemma 4.1 of the paper).

```julia
alg = DeflatedAlg(IpoptAlg())
options = DeflationOptions(sub_options = IpoptOptions(), iters = 10)
```
"""
struct DeflatedAlg{S<:AbstractOptimizer} <: AbstractOptimizer
    sub_alg::S
end

"""
    DeflationMultistartAlg(multi_alg)

Deflation meta-algorithm that combines deflation with an existing multistart
meta-algorithm `multi_alg`, e.g. `HyperoptAlg(IpoptAlg())`. Instead of
re-solving the deflated sub-problems from the same initial point with a
single sub-solver, every deflation round runs a full multistart search of
the deflated sub-problem, so both the deflation constraints and the
multistart sampler work together to find new, distinct minimizers.

The deflation constraint uses the fixed-bound formulation (Lemma 4.2 of the
paper): after `k` solutions `x₁, …, xₖ` have been found, the `(k+1)`-th
sub-problem adds the constraint

```math
∑_{j=1}^{k} (max(d(x, x_j) - radius, 0)^{-p} + σ) ≤ y_upper
```

with no auxiliary variable, so the multistart sampler explores only the
original variables. `p`, `σ`, `radius`, the distance `d` and `y_upper` are
set through [`DeflationMultistartOptions`](@ref), which also carries the
options of `multi_alg`.

```julia
alg = DeflationMultistartAlg(HyperoptAlg(IpoptAlg()))
options = DeflationMultistartOptions(
    multi_options = HyperoptOptions(sub_options = IpoptOptions(), iters = 10),
    iters = 4,
)
```
"""
struct DeflationMultistartAlg{S<:AbstractOptimizer} <: AbstractOptimizer
    multi_alg::S
end

# The deflated sub-problems add a nonlinear inequality constraint to the
# model, so the meta-algorithms require and forward that capability.
function _merge_deflation_capabilities(caps)
    return caps === nothing ? nothing : merge_capabilities(caps; nonlinear_ineq=true)
end

function NonconvexCore.solver_capabilities(alg::DeflatedAlg)
    return _merge_deflation_capabilities(solver_capabilities(alg.sub_alg))
end

function NonconvexCore.solver_capabilities(alg::DeflationMultistartAlg)
    return _merge_deflation_capabilities(solver_capabilities(alg.multi_alg))
end

"""
Validates the deflation hyperparameters shared by [`DeflationOptions`](@ref)
and [`DeflationMultistartOptions`](@ref).
"""
function _validate_deflation_parameters(p, sigma, radius, bound, tol, distance, weight)
    p > 0 || throw(ArgumentError("The deflation power `p` must be positive, got $p."))
    sigma >= 0 || throw(
        ArgumentError("The deflation shift `sigma` must be non-negative, got $sigma.")
    )
    radius >= 0 || throw(
        ArgumentError("The deflation radius `radius` must be non-negative, got $radius."),
    )
    bound > 0 || throw(ArgumentError("`y_upper` must be positive, got $bound."))
    tol > 0 || throw(ArgumentError("`tol` must be positive, got $tol."))
    if distance !== nothing && weight !== nothing
        throw(ArgumentError("Only one of `distance` and `weight` can be specified."))
    end
    return nothing
end

"""
    DeflationOptions(; kwargs...)

Options for [`DeflatedAlg`](@ref). The fields are:

- `sub_options`: the options of the sub-optimizer, or a function from the
  deflation iteration index to the sub-optimizer options.
- `iters`: the maximum number of deflation sub-problems to solve, i.e. the
  maximum number of distinct minimizers to find, `10` by default.
- `p`: the deflation power in `d(x, x_j)⁻ᵖ`, `2.0` by default. Larger
  powers concentrate the deflation effect closer to the known solutions.
- `sigma`: the shift `σ` added to every deflation term, `0.0` by default.
  `σ > 0` bounds the deflation function away from zero, which can help the
  convergence of the sub-solver.
- `radius`: the radius `r` of the hyper-ball around each known solution
  inside which all points are deflated identically, `0.0` by default. With
  `radius > 0`, every new solution is guaranteed to be at least `radius`
  away from all previously found ones (in the chosen distance), even if
  deflation fails to find a new minimizer.
- `distance`: an optional custom distance function `d(x, x_j)` returning a
  non-negative number, e.g. to deflate in a transformed space. By default
  the Euclidean distance is used.
- `weight`: an optional positive semi-definite matrix `Q` giving the
  generalized Euclidean distance `sqrt((x - x_j)' Q (x - x_j))`. A diagonal
  `Q` can be used to weigh the deviations per coordinate. Only one of
  `distance` and `weight` can be set.
- `y_upper`: the finite upper bound on the deflation variable `y`, `1e8` by
  default. The deflation constraint excludes a hyper-ball of radius
  `(y_upper - K̃σ)⁻¹ᐟᵖ` around each known solution, where `K̃` is the number
  of known solutions. If deflation fails, the sub-solver converges to the
  boundary of that ball; such a point is reported as unsuccessful (see
  `tol`). A larger `y_upper` allows solutions closer to the known ones to
  be considered distinct.
- `ctol`: the infeasibility tolerance for accepting a sub-problem solution
  as feasible, `1e-5` by default.
- `tol`: the distance tolerance used to decide that a new solution is
  identical to a previously found one, `1e-6` by default. A new solution is
  accepted only if it is farther than
  `max(tol, radius + 100·(y_upper - K̃σ)⁻¹ᐟᵖ)` from every known solution in
  the configured distance; see `deflation_failed` for the exact acceptance
  criterion.
- `keep_all`: if `true` (the default), all the sub-problem results are kept
  in the `results` field of the returned `DeflationResult`.
"""
struct DeflationOptions{S1,D,W} <: AbstractDeflationOptions
    sub_options::S1
    iters::Int
    p::Float64
    sigma::Float64
    radius::Float64
    distance::D
    weight::W
    y_upper::Float64
    ctol::Float64
    tol::Float64
    keep_all::Bool
end
function DeflationOptions(;
    sub_options,
    iters=10,
    p=2.0,
    sigma=0.0,
    radius=0.0,
    distance=nothing,
    weight=nothing,
    y_upper=1e8,
    ctol=1e-5,
    tol=1e-6,
    keep_all=true,
)
    iters >= 1 || throw(ArgumentError("`iters` must be a positive integer, got $iters."))
    _validate_deflation_parameters(p, sigma, radius, y_upper, tol, distance, weight)
    return DeflationOptions(
        sub_options,
        iters,
        float(p),
        float(sigma),
        float(radius),
        distance,
        weight,
        float(y_upper),
        float(ctol),
        float(tol),
        keep_all,
    )
end

"""
    DeflationMultistartOptions(; kwargs...)

Options for [`DeflationMultistartAlg`](@ref). The fields are:

- `multi_options`: the options of the wrapped multistart meta-algorithm,
  e.g. [`HyperoptOptions`](@ref) when the algorithm is a
  `DeflationMultistartAlg(HyperoptAlg(...))`.
- `iters`: the maximum number of deflation rounds to run, i.e. the maximum
  number of distinct minimizers to find, `10` by default. Each round is one
  full multistart solve of the deflated sub-problem.
- `p`, `sigma`, `radius`, `distance` and `weight`: the deflation
  hyperparameters, see [`DeflationOptions`](@ref).
- `y_upper`: the fixed bound `M` of the deflation constraint
  `∑ⱼ m(x; xⱼ) ≤ M` (Lemma 4.2 of the paper), `1e8` by default. No
  auxiliary variable is added, so the multistart sampler explores only the
  original variables. The constraint excludes a hyper-ball of radius
  `(y_upper - K̃σ)⁻¹ᐟᵖ` around each known solution, where `K̃` is the number
  of known solutions.
- `ctol`: the infeasibility tolerance for accepting a sub-problem solution
  as feasible, `1e-5` by default.
- `tol`: the distance tolerance used to decide that a new solution is
  identical to a previously found one, `1e-6` by default. A new solution is
  accepted only if it is farther than
  `max(tol, radius + 100·(y_upper - K̃σ)⁻¹ᐟᵖ)` from every known solution in
  the configured distance.
- `keep_all`: if `true` (the default), all the sub-problem results are kept
  in the `results` field of the returned `DeflationResult`.
"""
struct DeflationMultistartOptions{M,D,W} <: AbstractDeflationOptions
    multi_options::M
    iters::Int
    p::Float64
    sigma::Float64
    radius::Float64
    distance::D
    weight::W
    y_upper::Float64
    ctol::Float64
    tol::Float64
    keep_all::Bool
end
function DeflationMultistartOptions(;
    multi_options,
    iters=10,
    p=2.0,
    sigma=0.0,
    radius=0.0,
    distance=nothing,
    weight=nothing,
    y_upper=1e8,
    ctol=1e-5,
    tol=1e-6,
    keep_all=true,
)
    iters >= 1 || throw(ArgumentError("`iters` must be a positive integer, got $iters."))
    _validate_deflation_parameters(p, sigma, radius, y_upper, tol, distance, weight)
    return DeflationMultistartOptions(
        multi_options,
        iters,
        float(p),
        float(sigma),
        float(radius),
        distance,
        weight,
        float(y_upper),
        float(ctol),
        float(tol),
        keep_all,
    )
end

"""
    DeflationResult

The result of [`DeflatedAlg`](@ref) and [`DeflationMultistartAlg`](@ref),
with the following fields:

- `minimum`: the best objective value among the found minimizers.
- `minimizer`: the found minimizer with the best objective value.
- `minimizers`: all the distinct feasible minimizers found, in the order
  they were found (in the flattened variable space).
- `results`: all the sub-problem results when `keep_all = true`, `nothing`
  otherwise.
- `converged`: `true` if all `iters` deflation iterations produced feasible
  solutions that are distinct from all previously found ones.
- `niters`: the number of sub-problems actually solved before stopping.
"""
struct DeflationResult{M1,M2,R,S} <: AbstractResult
    minimum::M1
    minimizer::M2
    minimizers::R
    results::S
    converged::Bool
    niters::Int
end

"""
The upper bound of the deflation constraint: the `y_upper` field of both
options structs, used as the auxiliary variable bound by [`DeflatedAlg`](@ref)
and as the fixed bound `M` by [`DeflationMultistartAlg`](@ref).
"""
_deflation_bound(options::AbstractDeflationOptions) = options.y_upper

"""
The solver used to solve the deflated sub-problems: the sub-solver of
[`DeflatedAlg`](@ref) or the wrapped multistart algorithm of
[`DeflationMultistartAlg`](@ref).
"""
_deflation_solver(alg::DeflatedAlg) = alg.sub_alg
_deflation_solver(alg::DeflationMultistartAlg) = alg.multi_alg

"""
The options of the solver used for the `i`-th deflation sub-problem.
"""
function _deflation_sub_options(options::DeflationOptions, i::Int)
    return options.sub_options isa Function ? options.sub_options(i) : options.sub_options
end
_deflation_sub_options(options::DeflationMultistartOptions, i::Int) = options.multi_options

"""
The distance between `x` and a known solution `x_k`, as configured in
`options`: a custom function, a `Q`-weighted Euclidean distance, or the
plain Euclidean distance.
"""
function deflation_distance(
    x::AbstractVector, xk::AbstractVector, options::AbstractDeflationOptions
)
    distance = options.distance
    weight = options.weight
    if distance !== nothing
        return float(distance(x, xk))
    elseif weight !== nothing
        δ = x .- xk
        return sqrt(float(dot(δ, weight * δ)))
    else
        return norm(x .- xk)
    end
end

"""
The deflation function `m(x; x_k) = max(d(x, x_k) - radius, 0)⁻ᵖ + sigma`.
"""
function deflation_function(
    x::AbstractVector, xk::AbstractVector, options::AbstractDeflationOptions
)
    @unpack p, sigma, radius = options
    d = deflation_distance(x, xk, options)
    d = max(d - radius, 0.0)
    # Keep the value finite inside the exclusion hyper-ball: a huge (but
    # finite) value already makes those points infeasible for the sub-solver
    # and is friendlier to it than `Inf`.
    d = max(d, 1e-12)
    return d^(-p) + sigma
end

"""
The sum of the deflation functions over all previously found solutions.
"""
function deflation_sum(
    x::AbstractVector, found::Vector{<:AbstractVector}, options::AbstractDeflationOptions
)
    s = 0.0
    for xk in found
        s += deflation_function(x, xk, options)
    end
    return s
end

"""
A copy of `model` whose box and constraint containers are freshly allocated,
so variables and constraints can be appended without mutating the original.
The (possibly immutable, empty-tuple) inequality constraint container is
replaced by an extensible vector.
"""
function _copy_model_for_deflation(model::VecModel)
    fs = model.ineq_constraints.fs
    ineq = if fs isa Tuple
        NonconvexCore.VectorOfFunctions(NonconvexCore.IneqConstraint[])
    else
        NonconvexCore.VectorOfFunctions(copy(fs))
    end
    return VecModel(
        model.objective,
        model.eq_constraints,
        ineq,
        model.sd_constraints,
        model.set_constraints,
        copy(model.set_variables),
        copy(model.box_min),
        copy(model.box_max),
        copy(model.init),
        copy(model.integer),
    )
end

"""
The shared preparation of the deflated model: checks the feasibility of the
deflation bounds, snapshots the found solutions (so later additions to
`found` do not change the constraint) and copies the model. Returns
`nothing` when no solution has been found yet (no deflation constraint is
added).
"""
function _prepare_deflated_model(model, found, options)
    isempty(found) && return nothing
    _check_deflation_bounds(length(found), options.sigma, _deflation_bound(options))
    fs = [copy(xk) for xk in found]
    return fs, _copy_model_for_deflation(model)
end

function _check_deflation_bounds(nfound, sigma, bound)
    ylb = nfound * sigma
    if ylb > bound
        throw(
            ArgumentError(
                "The deflation constraint is infeasible with `sigma` = $sigma, " *
                "$nfound known solutions and `y_upper` = $bound: " *
                "the sum of the deflation terms is at least $ylb. " *
                "Reduce `sigma` or increase `y_upper`.",
            ),
        )
    end
    return nothing
end

"""
Builds the model of the next deflation sub-problem for [`DeflatedAlg`](@ref):
the original model extended with the deflation variable `y ∈ [k·σ, y_upper]`
and the constraint `∑ⱼ m(x; xⱼ) ≤ y` over the previously found solutions.
Returns the model and a feasible starting point `[x0; y0]` where `y0` is the
value of the deflation constraint at `x0` (plus a small margin), clipped to
the bounds of `y`. When no solution has been found yet, the original model
and `x0` are returned unchanged.
"""
function deflated_model(
    model::VecModel,
    found::Vector{<:AbstractVector},
    x0::AbstractVector,
    options::DeflationOptions,
)
    prep = _prepare_deflated_model(model, found, options)
    prep === nothing && return model, copy(x0)
    fs, dm = prep
    @unpack y_upper, sigma = options
    n = length(x0)
    ylb = length(found) * sigma
    addvar!(dm, ylb, y_upper; init=y_upper)
    y0 = clamp(deflation_sum(x0, fs, options) + 1.0, ylb, y_upper)
    add_ineq_constraint!(dm, function (xz)
        return deflation_sum(xz[1:n], fs, options) - xz[end]
    end)
    return dm, [x0; y0]
end

"""
Builds the model of the next deflation sub-problem for
[`DeflationMultistartAlg`](@ref): the original model extended with the
fixed-bound deflation constraint `∑ⱼ m(x; xⱼ) ≤ y_upper` over the previously
found solutions. No auxiliary variable is added, so the multistart sampler
explores only the original variables. When no solution has been found yet,
the original model and `x0` are returned unchanged.
"""
function deflated_model(
    model::VecModel,
    found::Vector{<:AbstractVector},
    x0::AbstractVector,
    options::DeflationMultistartOptions,
)
    prep = _prepare_deflated_model(model, found, options)
    prep === nothing && return model, copy(x0)
    fs, dm = prep
    add_ineq_constraint!(dm, function (x)
        return deflation_sum(x, fs, options) - options.y_upper
    end)
    return dm, copy(x0)
end

"""
Validates that the solver can handle the deflated sub-problems: they add a
nonlinear inequality constraint, so a solver without that capability would
fail only after the first (undeflated) solve. A dummy deflated model is
built to reuse the exact capability machinery and error messages of
`checksubproblem!`.
"""
function check_deflation_capabilities(
    model::VecModel,
    solver::AbstractOptimizer,
    x0::AbstractVector,
    options::AbstractDeflationOptions,
)
    dm, _ = deflated_model(model, [x0], x0, options)
    checksubproblem!(
        dm,
        solver;
        hint="The deflation meta-algorithm adds a nonlinear inequality deflation constraint to the model, which the sub-solver must support.",
    )
    return nothing
end

"""
The distinctness threshold: the minimum distance (in the configured
distance) between a new solution and every known solution.
"""
function _deflation_threshold(options::AbstractDeflationOptions, nfound, tol)
    @unpack p, sigma, radius = options
    ycap = max(_deflation_bound(options) - nfound * sigma, 0.0)
    return max(tol, radius + 100.0 * ycap^(-1.0 / p))
end

"""
Returns `true` when the new point `x` is not a new distinct solution. Two
signals are used:

1. `x` is too close to a previously found solution in the configured
   distance: closer than `max(tol, radius + 100·(y_upper - K̃σ)⁻¹ᐟᵖ)` where
   `K̃` is the number of known solutions. The second term accounts for the
   exclusion region around each known solution (Section 4.5 of the paper):
   a solver that fails to escape to a new minimizer converges to the
   boundary of that region, possibly overshooting it by an amount that
   depends on the solver's tolerances.
2. The deflation constraint is active at (numerically) its upper bound
   `y_upper`, i.e. the solver converged to a point on the exclusion
   boundary at the cap.
"""
function deflation_failed(
    x::AbstractVector,
    xfull::AbstractVector,
    found::Vector{<:AbstractVector},
    options::AbstractDeflationOptions,
    tol::Real,
)
    isempty(found) && return false
    threshold = _deflation_threshold(options, length(found), tol)
    for xk in found
        deflation_distance(x, xk, options) <= threshold && return true
    end
    return _deflation_cap_failure(x, xfull, found, options)
end

# For the auxiliary-variable formulation, the constraint is active at the
# cap when both the returned `y` and the constraint value `m` are pinned at
# (numerically) `y_upper`.
function _deflation_cap_failure(
    x::AbstractVector,
    xfull::AbstractVector,
    found::Vector{<:AbstractVector},
    options::DeflationOptions,
)
    length(xfull) > length(x) || return false
    y = xfull[end]
    m = deflation_sum(x, found, options)
    bound = _deflation_bound(options)
    return y >= bound * (1 - 1e-8) && m >= y * (1 - 1e-8)
end

# For the fixed-bound formulation, the constraint is active at the cap when
# its value `m` is (numerically) equal to the fixed bound `y_upper`.
function _deflation_cap_failure(
    x::AbstractVector,
    xfull::AbstractVector,
    found::Vector{<:AbstractVector},
    options::DeflationMultistartOptions,
)
    m = deflation_sum(x, found, options)
    return m >= _deflation_bound(options) * (1 - 1e-8)
end

mutable struct DeflationWorkspace{
    A<:Union{DeflatedAlg,DeflationMultistartAlg},
    X<:AbstractVector,
    O<:AbstractDeflationOptions,
    F<:Vector{<:AbstractVector},
    R<:Vector{<:AbstractResult},
    K,
} <: Workspace
    sub_workspace::Workspace
    model::VecModel
    alg::A
    x0::X
    options::O
    found::F
    results::R
    kwargs::K
end

function Workspace(
    model::VecModel,
    alg::DeflatedAlg,
    x0::AbstractVector;
    options::DeflationOptions,
    kwargs...,
)
    return _deflation_workspace(model, alg, x0, options, kwargs)
end

function Workspace(
    model::VecModel,
    alg::DeflationMultistartAlg,
    x0::AbstractVector;
    options::DeflationMultistartOptions,
    kwargs...,
)
    return _deflation_workspace(model, alg, x0, options, kwargs)
end

function Workspace(
    model::VecModel, alg::Union{DeflatedAlg,DeflationMultistartAlg}; options, kwargs...
)
    return Workspace(model, alg, NonconvexCore.getinit(model); options=options, kwargs...)
end

function _deflation_workspace(model, alg, x0, options, kwargs)
    if options.iters > 1
        check_deflation_capabilities(model, _deflation_solver(alg), x0, options)
    end
    sub_workspace = Workspace(
        model,
        _deflation_solver(alg),
        copy(x0);
        options=_deflation_sub_options(options, 1),
        kwargs...,
    )
    return DeflationWorkspace(
        sub_workspace,
        model,
        alg,
        copy(x0),
        options,
        Vector{Vector{Float64}}(),
        AbstractResult[],
        kwargs,
    )
end

# Restart the whole deflation procedure: drop the found solutions and
# rebuild the first (undeflated) sub-workspace, optionally from a new start.
function reset!(workspace::DeflationWorkspace, x0=nothing)
    x0 === nothing || (workspace.x0 .= x0)
    empty!(workspace.found)
    empty!(workspace.results)
    workspace.sub_workspace = Workspace(
        workspace.model,
        _deflation_solver(workspace.alg),
        copy(workspace.x0);
        options=_deflation_sub_options(workspace.options, 1),
        workspace.kwargs...,
    )
    return workspace
end

function optimize!(workspace::DeflationWorkspace)
    @unpack options, model, alg, x0, kwargs = workspace
    @unpack iters, ctol, tol = options
    found = workspace.found
    results = workspace.results
    empty!(found)
    empty!(results)
    @info "Deflation algorithm started." maxlog = 1
    n = length(x0)
    local r
    for i in 1:iters
        sub_options = _deflation_sub_options(options, i)
        dm, x0i = deflated_model(model, found, x0, options)
        workspace.sub_workspace = Workspace(
            dm, _deflation_solver(alg), x0i; options=sub_options, kwargs...
        )
        r = optimize!(workspace.sub_workspace)
        push!(results, r)
        xnew = collect(Float64, r.minimizer[1:n])
        if !isfeasible(model, xnew; ctol=ctol)
            @warn "Deflation iteration $i did not find a feasible solution; stopping."
            break
        end
        if deflation_failed(xnew, r.minimizer, found, options, tol)
            @warn "Deflation iteration $i did not find a new solution; stopping."
            break
        end
        push!(found, xnew)
    end
    return deflation_result(workspace)
end

"""
Assembles the [`DeflationResult`](@ref) from the recorded sub-results and
found minimizers.
"""
function deflation_result(workspace::DeflationWorkspace)
    @unpack options = workspace
    results = workspace.results
    found = workspace.found
    n = length(workspace.x0)
    if isempty(found)
        # No feasible distinct solution was found: report the best sub-result
        # (e.g. the first undeflated solve).
        best_ind = argmin(r -> r.minimum, results)
        best = results[best_ind]
        return DeflationResult(
            best.minimum,
            collect(Float64, best.minimizer[1:n]),
            Vector{Vector{Float64}}(),
            options.keep_all ? copy(results) : nothing,
            false,
            length(results),
        )
    else
        minima = [results[i].minimum for i in eachindex(found)]
        best_ind = argmin(minima)
        return DeflationResult(
            minima[best_ind],
            copy(found[best_ind]),
            [copy(x) for x in found],
            options.keep_all ? copy(results) : nothing,
            length(found) == options.iters,
            length(results),
        )
    end
end
