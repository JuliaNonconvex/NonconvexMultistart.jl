# Additional tests for NonconvexMultistart (src/hyperopt.jl) targeting paths
# not exercised by the main suite: the removed GPSampler stub, the sampler
# constructor wrappers, HyperoptOptions field handling (lb/ub/ctol/keep_all),
# HyperoptResult bookkeeping, CLHSampler dims, and HyperoptAlg wrapping a
# cheap local sub-algorithm.

@testitem "sampler constructor wrappers" begin
    using NonconvexMultistart, Test
    import Hyperopt
    import Random

    @test RandomSampler() isa Hyperopt.RandomSampler
    @test LHSampler() isa Hyperopt.LHSampler
    @test CLHSampler() isa Hyperopt.CLHSampler
    # The dims argument is forwarded to Hyperopt's CLHSampler.
    @test CLHSampler(Hyperopt.Continuous()) isa Hyperopt.CLHSampler

    hb = Hyperband(R=100, η=3, inner=RandomSampler())
    @test hb isa Hyperopt.Hyperband
    @test hb.R == 100
    @test hb.η == 3
    @test hb.inner isa Hyperopt.RandomSampler

    # GPSampler was removed upstream (Hyperopt >= 0.5); the exported stub
    # raises a descriptive error for every call signature.
    err = try
        GPSampler()
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("removed", sprint(showerror, err))
    @test_throws ErrorException GPSampler(Random.Xoshiro(1))
    @test_throws ErrorException GPSampler(1; a=2)
end

@testitem "HyperoptOptions fields" begin
    using NonconvexMultistart, NonconvexIpopt, Test
    import Hyperopt

    sub = IpoptOptions(first_order=true)
    lb = [0.0, 0.0]
    ub = [2.0, 2.0]
    o = HyperoptOptions(
        sub_options=sub,
        lb=lb,
        ub=ub,
        iters=7,
        searchspace_size=10,
        sampler=LHSampler(),
        ctol=1e-4,
        keep_all=false,
    )
    @test o.sub_options === sub
    @test o.lb === lb
    @test o.ub === ub
    @test o.iters == 7
    @test o.searchspace_size == 10
    @test o.sampler isa Hyperopt.LHSampler
    @test o.ctol == 1e-4
    @test o.keep_all === false

    @testset "defaults" begin
        o2 = HyperoptOptions(sub_options=sub)
        @test o2.lb === nothing
        @test o2.ub === nothing
        @test o2.iters == 40
        @test o2.searchspace_size == 40
        @test o2.sampler isa Hyperopt.RandomSampler
        @test o2.ctol == 1e-5
        @test o2.keep_all === true
    end
end

@testitem "lb/ub/ctol and result bookkeeping" begin
    using NonconvexMultistart, NonconvexIpopt, LinearAlgebra, Test

    f(x::AbstractVector) = (x[1] - 1)^2 + (x[2] - 2)^2
    m = Model(f)
    addvar!(m, [0.0, 0.0], [3.0, 3.0])
    alg = HyperoptAlg(IpoptAlg())

    @testset "keep_all = true stores every sub-solve" begin
        options = HyperoptOptions(
            sub_options=IpoptOptions(first_order=true, print_level=0),
            lb=[0.5, 1.0],
            ub=[2.5, 2.5],
            iters=6,
            searchspace_size=6,
            sampler=RandomSampler(),
            ctol=1e-6,
            keep_all=true,
        )
        r = NonconvexMultistart.optimize(m, alg, [0.5, 1.5], options=options)
        @test r isa NonconvexMultistart.HyperoptResult
        @test length(r.results) == options.iters
        @test 1 <= r.optimal_ind <= length(r.results)
        @test r.minimum == r.results[r.optimal_ind].minimum
        @test r.minimizer ≈ [1.0, 2.0] atol = 1e-6
        @test abs(r.minimum) < 1e-6
    end

    @testset "keep_all = false drops results and optimal_ind" begin
        options = HyperoptOptions(
            sub_options=IpoptOptions(first_order=true, print_level=0),
            iters=5,
            sampler=RandomSampler(),
            keep_all=false,
        )
        r = NonconvexMultistart.optimize(m, alg, [0.5, 1.5], options=options)
        @test r isa NonconvexMultistart.HyperoptResult
        @test r.results === nothing
        @test r.optimal_ind === nothing
        @test r.minimizer ≈ [1.0, 2.0] atol = 1e-6
        @test abs(r.minimum) < 1e-6
    end
end

@testitem "HyperoptAlg wrapping a cheap local algorithm" begin
    using NonconvexMultistart, NonconvexMMA, NonconvexCore, LinearAlgebra, Test

    f(x) = sum((x .- 1.0) .^ 2)
    m = Model(f)
    addvar!(m, fill(-5.0, 2), fill(5.0, 2))

    alg = HyperoptAlg(MMA02())
    options = HyperoptOptions(
        sub_options=MMAOptions(maxiter=30, verbose=false), iters=4, sampler=RandomSampler()
    )
    r = NonconvexMultistart.optimize(m, alg, fill(0.0, 2), options=options)
    @test norm(r.minimizer - ones(2)) < 1e-2
    @test abs(r.minimum) < 1e-3
    # Hyperopt forwards the sub-solver's capabilities.
    @test solver_capabilities(alg) == solver_capabilities(MMA02())
end
