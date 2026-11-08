struct HyperoptAlg{S<:AbstractOptimizer} <: AbstractOptimizer
    sub_alg::S
end

# Hyperopt is a meta-algorithm: all model features are forwarded to the
# sub-solver.
NonconvexCore.solver_capabilities(alg::HyperoptAlg) = solver_capabilities(alg.sub_alg)

"""
    HyperoptOptions: options performing starting point optimization using Hyperopt.jl

- `sub_options`: options for the sub-optimizer.
- `lb`: Lower bound of starting point, if don't specify it, the default value will be `nothing`, 
            then will end up be replaced by the lower bound of optimization problem.
- `ub`: Upper bound of starting point, same as above. 
- `searchspace_size::Integer`: How many potential starting points we generate.
- `iters::Integer`: Among all generated potential starting points, how many of them will be evaluated. 
- `sampler::Hyperopt.Sampler`: An instance of 'Hyperopt.Sampler', which decides search algorithm. 
- `ctol`: infeasibility tolerance for accepting a solution as feasible
- `keep_all`: if true, all the solutions of the sub-problems will be saved
"""
struct HyperoptOptions{S1,L,U,S2<:Hyperopt.Sampler,C}
    sub_options::S1
    lb::L
    ub::U
    searchspace_size::Int
    iters::Int
    sampler::S2
    ctol::C
    keep_all::Bool
end
function HyperoptOptions(;
    sub_options,
    lb = nothing,
    ub = nothing,
    iters = 40,
    searchspace_size = iters,
    sampler = RandomSampler(),
    ctol = 1e-5,
    keep_all = true,
)
    return HyperoptOptions(
        sub_options,
        lb,
        ub,
        searchspace_size,
        iters,
        sampler,
        ctol,
        keep_all,
    )
end

struct HyperoptWorkspace{S<:Workspace,X<:AbstractVector,O<:HyperoptOptions} <: Workspace
    sub_workspace::S
    x0::X
    options::O
end

function Workspace(
    model::VecModel,
    alg::HyperoptAlg,
    x0::AbstractVector;
    options,
    kwargs...,
)
    sub_options =
        options.sub_options isa Function ? options.sub_options(1) : options.sub_options
    return HyperoptWorkspace(
        Workspace(model, alg.sub_alg, copy(x0); options = sub_options, kwargs...),
        x0,
        options,
    )
end

"""
    HyperoptResult

When using multiple x0 in [`optimize`](@ref), return this result, including following fields: 
- `minimum`: the optimal value among all searches.
- `minmizer`: the optimal solutions found.
- `results`: all the search results.
- `optimal_ind`: the index of the optimal solution in `results`.
"""
struct HyperoptResult{M1,M2,R,O} <: AbstractResult
    minimum::M1
    minimizer::M2
    results::R
    optimal_ind::O
end

function optimize!(workspace::HyperoptWorkspace)
    @unpack options, x0, sub_workspace = workspace
    @unpack model = sub_workspace
    @unpack lb, ub, searchspace_size, ctol, keep_all = options
    @unpack sampler, iters = options

    lb = lb === nothing ? getmin(model) : lb
    ub = ub === nothing ? getmax(model) : ub
    @assert all(isfinite, lb) && all(isfinite, ub) throw(
        "Please use finite bounds for the starting point search.",
    )

    @info "Multistart algorithm started. "
    @assert searchspace_size >= 1 "searchspace_size must be a positive integer. "

    # Generate a search space
    _sampler = sampler isa Hyperopt.Hyperband ? sampler.inner : sampler
    params, candidates = get_sobol_candidates(lb, ub, x0, searchspace_size)
    objective =
        (i, x0; state = nothing) -> begin
            if sampler isa Hyperopt.Hyperband
                @assert options.sub_options isa Function
                sub_options = options.sub_options(i)
                _sub_workspace = @set sub_workspace.options = sub_options
            else
                _sub_workspace = sub_workspace
            end
            # Warm-start from the previous rung's minimizer when Hyperband
            # passes one back through `state`.
            _x0 = state === nothing ? x0 : state
            reset!(_sub_workspace, _x0)
            return optimize!(_sub_workspace), x0
        end
    _ho = Hyperopt.Hyperoptimizer(
        iterations = iters,
        sampler = sampler,
        objective = objective,
        params = params,
        candidates = candidates,
    )
    ho = switch_objective(
        _ho,
        (args...; kwargs...) -> begin
            if length(args) == 2 && sampler isa Hyperopt.Hyperband
                res, _ = _ho.objective(args[1], args[2]; kwargs...)
            else
                res, _ = _ho.objective(args...)
            end
            # Hyperopt's hyperband appends the rung losses to results itself;
            # only record evaluations for the other samplers.
            if !(sampler isa Hyperopt.Hyperband)
                push!(_ho.results, res.minimum)
            end
            if sampler isa Hyperopt.Hyperband
                return isfeasible(model, res.minimizer, ctol = ctol) ? res.minimum : Inf,
                res.minimizer
            else
                return res
            end
        end,
    )
    if _sampler isa Union{Hyperopt.LHSampler,Hyperopt.CLHSampler}
        searchspace_size == iters || throw(
            ArgumentError(
                "LHSampler and CLHSampler require `searchspace_size == iters` (got $searchspace_size and $iters).",
            ),
        )
        Hyperopt.init!(_sampler, ho)
    end
    if sampler isa Hyperopt.Hyperband
        optimal = Hyperopt.hyperband(ho)
        return HyperoptResult(optimal.loss, optimal.state, nothing, nothing)
    else
        hor = map(ho) do (args)
            ho.objective(values(args)...)
        end
        mapped = map(
            res -> (isfeasible(model, res.minimizer, ctol = ctol) ? res.minimum : Inf),
            hor,
        )
        ind = argmin(mapped)
        isfinite(mapped[ind]) || @warn "No feasible solution was found among the multistart results; returning the best result found."
        optimal = hor[ind]
        if keep_all
            return HyperoptResult(optimal.minimum, optimal.minimizer, identity.(hor), ind)
        else
            return HyperoptResult(optimal.minimum, optimal.minimizer, nothing, nothing)
        end
    end
end

function switch_objective(ho, objective)
    (; iterations, params, candidates, history, results, sampler) = ho
    return Hyperopt.Hyperoptimizer(;
        iterations,
        params,
        candidates,
        history,
        results,
        sampler,
        objective,
    )
end

function get_sobol_candidates(lb, ub, x0, n)
    sobol_seq_generator = SobolSeq(lb, ub)
    return (:x0,), ([[x0]; collect(next!(sobol_seq_generator) for _ = 1:n-1)],)
end

function RandomSampler(args...; kwargs...)
    return Hyperopt.RandomSampler(args...; kwargs...)
end
function Hyperband(args...; kwargs...)
    return Hyperopt.Hyperband(args...; kwargs...)
end
function LHSampler(args...; kwargs...)
    return Hyperopt.LHSampler(args...; kwargs...)
end
function CLHSampler(dims = Hyperopt.Continuous())
    return Hyperopt.CLHSampler(dims = [dims])
end

function GPSampler(args...; kwargs...)
    error(
        "GPSampler was removed in NonconvexMultistart 0.2.0 because Hyperopt removed " *
        "GPSampler upstream (Hyperopt >= 0.5). Use RandomSampler, LHSampler or " *
        "CLHSampler instead.",
    )
end
