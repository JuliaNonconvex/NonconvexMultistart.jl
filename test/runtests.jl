using NonconvexMultistart, NonconvexIpopt, LinearAlgebra, Test
import Hyperopt

f(x::AbstractVector) = sqrt(x[2])
g(x::AbstractVector, a, b) = (a * x[1] + b)^3 - x[2]

alg = HyperoptAlg(IpoptAlg())

@testset "Sampler - $spl_name" for (spl_name, spl) in [
    ("RandomSampler", RandomSampler()),
    ("Hyperband", Hyperband(R = 100, η = 3, inner = RandomSampler())),
    ("LHSampler", LHSampler()),
    ("CLHSampler", CLHSampler()),
]
    if spl_name == "Hyperband"
        options = HyperoptOptions(
            sub_options = max_iter ->
                IpoptOptions(first_order = true, max_iter = ceil(Int, max_iter)),
            sampler = spl,
        )
    else
        options =
            HyperoptOptions(sub_options = IpoptOptions(first_order = true), sampler = spl)
    end
    m = Model(f)
    addvar!(m, [0.0, 0.0], [10.0, 10.0])
    add_ineq_constraint!(m, x -> g(x, 2, 0))
    add_ineq_constraint!(m, x -> g(x, -1, 1))

    r = NonconvexMultistart.optimize(m, alg, [1.234, 2.345], options = options)
    @test abs(r.minimum - sqrt(8 / 27)) < 1e-6
    @test norm(r.minimizer - [1 / 3, 8 / 27]) < 1e-6
end

@testset "LHS validation" begin
    m = Model(f)
    addvar!(m, [0.0, 0.0], [10.0, 10.0])
    add_ineq_constraint!(m, x -> g(x, 2, 0))
    add_ineq_constraint!(m, x -> g(x, -1, 1))

    # Regression test: mismatched searchspace_size/iters used to surface as a
    # cryptic internal error from Hyperopt.
    options = HyperoptOptions(
        sub_options = IpoptOptions(first_order = true),
        sampler = LHSampler(),
        searchspace_size = 5,
        iters = 10,
    )
    @test_throws ArgumentError NonconvexMultistart.optimize(
        m, alg, [1.234, 2.345], options = options,
    )
end

@testset "Hyperband results bookkeeping" begin
    # Regression test: evaluations used to be double-pushed into the results
    # vector (once by the wrapper and once by Hyperopt.hyperband), violating
    # the invariant length(results) == length(sampler.budgets) that Hyperopt
    # checks. Replicate the optimizer's Hyperoptimizer construction and check
    # the invariant after a run.
    x0 = [1.234, 2.345]
    lb, ub = [0.0, 0.0], [10.0, 10.0]
    params, candidates = NonconvexMultistart.get_sobol_candidates(lb, ub, x0, 8)
    objective = (i, x0i; state = nothing) -> begin
        _x0 = state === nothing ? x0i : state
        (minimum = f(_x0), minimizer = _x0, fcalls = 1), x0i
    end
    sampler = Hyperband(R = 100, η = 3, inner = RandomSampler())
    _ho = Hyperopt.Hyperoptimizer(
        iterations = 8,
        sampler = sampler,
        objective = objective,
        params = params,
        candidates = candidates,
    )
    ho = NonconvexMultistart.switch_objective(
        _ho,
        (args...; kwargs...) -> begin
            if length(args) == 2
                res, _ = _ho.objective(args[1], args[2]; kwargs...)
            else
                res, _ = _ho.objective(args...)
            end
            if !(sampler isa Hyperopt.Hyperband)
                push!(_ho.results, res.minimum)
            end
            return isfinite(res.minimum) ? res.minimum : Inf, res.minimizer
        end,
    )
    optimal = Hyperopt.hyperband(ho)
    @test isfinite(optimal.loss)
    @test length(ho.results) == length(sampler.budgets)

@testset "solver capabilities" begin
    @test solver_capabilities(HyperoptAlg(IpoptAlg())) == solver_capabilities(IpoptAlg())
