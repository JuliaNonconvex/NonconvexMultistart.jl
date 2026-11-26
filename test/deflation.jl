@testitem "Deflation finds multiple minimizers" begin
    using NonconvexMultistart, NonconvexIpopt, LinearAlgebra, Test

    # Himmelblau's function: 4 global minima with f = 0.
    f(x) = (x[1]^2 + x[2] - 11)^2 + (x[1] + x[2]^2 - 7)^2
    minima = [
        (3.0, 2.0), (-2.805118, 3.131312), (-3.779310, -3.283186), (3.584428, -1.848126)
    ]
    m = Model(f)
    addvar!(m, [-5.0, -5.0], [5.0, 5.0])
    options = DeflationOptions(sub_options=IpoptOptions(first_order=true), iters=3)
    r = NonconvexMultistart.optimize(
        m, DeflatedAlg(IpoptAlg()), [0.0, 0.0], options=options
    )
    @test r.converged
    @test r.niters == 3
    @test length(r.minimizers) == 3
    @test r.minimum ≈ 0.0 atol = 1e-6
    for x in r.minimizers
        @test f(x) < 1e-6
        @test minimum(norm(x .- collect(v)) for v in minima) < 1e-3
    end
    for i in 1:3, j in (i + 1):3
        @test norm(r.minimizers[i] .- r.minimizers[j]) > 1e-2
    end
    @test length(r.results) == 3
end

@testitem "Deflation stops when no new solution exists" begin
    using NonconvexMultistart, NonconvexIpopt, Test

    # x^2 has a single minimizer at 0: after finding it, deflation cannot
    # escape to another basin and the run must stop early.
    f(x) = x[1]^2
    m = Model(f)
    addvar!(m, -5.0, 5.0)
    options = DeflationOptions(sub_options=IpoptOptions(first_order=true), iters=3)
    r = NonconvexMultistart.optimize(m, DeflatedAlg(IpoptAlg()), [2.0], options=options)
    @test !r.converged
    @test r.niters == 2
    @test length(r.minimizers) == 1
    @test abs(r.minimizers[1][1]) < 1e-4
    @test r.minimum < 1e-6
end

@testitem "Deflation detects basin escape failure" begin
    using NonconvexMultistart, NonconvexIpopt, Test

    # cos on [0, 10] has two minima of equal value, π and 3π, but a local
    # solver started at 5.0 always descends into the π basin; the deflated
    # sub-problem converges to the boundary of the exclusion region around
    # π instead of 3π, which must be detected and reported as unsuccessful.
    f(x) = cos(x[1])
    m = Model(f)
    addvar!(m, 0.0, 10.0)
    options = DeflationOptions(sub_options=IpoptOptions(first_order=true), iters=2)
    r = NonconvexMultistart.optimize(m, DeflatedAlg(IpoptAlg()), [5.0], options=options)
    @test !r.converged
    @test r.niters == 2
    @test length(r.minimizers) == 1
    @test abs(r.minimizers[1][1] - pi) < 1e-5
end

@testitem "Deflation radius stops boundary solutions" begin
    using NonconvexMultistart, NonconvexIpopt, Test

    # With radius=1 on x^2, the deflated sub-problem cannot get closer than
    # 1 to the minimizer 0, but the resulting boundary point is not a new
    # minimizer: the run must stop with a single minimizer.
    f(x) = x[1]^2
    m = Model(f)
    addvar!(m, -5.0, 5.0)
    options = DeflationOptions(
        sub_options=IpoptOptions(first_order=true), iters=2, radius=1.0
    )
    r = NonconvexMultistart.optimize(m, DeflatedAlg(IpoptAlg()), [2.0], options=options)
    @test !r.converged
    @test r.niters == 2
    @test length(r.minimizers) == 1
    @test abs(r.minimizers[1][1]) < 1e-4
end

@testitem "Deflation with weight matrix and custom distance" begin
    using NonconvexMultistart, NonconvexIpopt, LinearAlgebra, Test

    f(x) = (x[1]^2 + x[2] - 11)^2 + (x[1] + x[2]^2 - 7)^2
    minima = [
        (3.0, 2.0), (-2.805118, 3.131312), (-3.779310, -3.283186), (3.584428, -1.848126)
    ]

    # A uniformly scaled weight matrix has the same geometry as the plain
    # Euclidean distance.
    m = Model(f)
    addvar!(m, [-5.0, -5.0], [5.0, 5.0])
    options = DeflationOptions(
        sub_options=IpoptOptions(first_order=true), iters=2, weight=[2.0 0.0; 0.0 2.0]
    )
    r = NonconvexMultistart.optimize(
        m, DeflatedAlg(IpoptAlg()), [0.0, 0.0], options=options
    )
    @test r.converged
    @test length(r.minimizers) == 2
    for x in r.minimizers
        @test minimum(norm(x .- collect(v)) for v in minima) < 1e-3
    end

    # A custom (Chebyshev) distance measure.
    m2 = Model(f)
    addvar!(m2, [-5.0, -5.0], [5.0, 5.0])
    options2 = DeflationOptions(
        sub_options=IpoptOptions(first_order=true),
        iters=2,
        distance=(x, xk) -> maximum(abs.(x .- xk)),
    )
    r2 = NonconvexMultistart.optimize(
        m2, DeflatedAlg(IpoptAlg()), [0.0, 0.0], options=options2
    )
    @test r2.converged
    @test length(r2.minimizers) == 2
    for x in r2.minimizers
        @test minimum(norm(x .- collect(v)) for v in minima) < 1e-3
    end
end

@testitem "Deflation with MMA sub-solver" begin
    using NonconvexMultistart, NonconvexMMA, LinearAlgebra, Test

    f(x) = sqrt(x[2])
    g(x, a, b) = (a * x[1] + b)^3 - x[2]
    m = Model(f)
    addvar!(m, [0.0, 0.0], [10.0, 10.0])
    add_ineq_constraint!(m, x -> g(x, 2, 0))
    add_ineq_constraint!(m, x -> g(x, -1, 1))
    options = DeflationOptions(sub_options=MMAOptions(maxiter=2000), iters=1)
    r = NonconvexMultistart.optimize(
        m, DeflatedAlg(MMA87()), [1.234, 2.345], options=options
    )
    @test r.converged
    @test abs(r.minimum - sqrt(8 / 27)) < 1e-6
    @test norm(r.minimizer - [1 / 3, 8 / 27]) < 1e-6
end

@testitem "Deflation keep_all, function sub-options and reset" begin
    using NonconvexMultistart, NonconvexIpopt, Test

    f(x) = (x[1]^2 + x[2] - 11)^2 + (x[1] + x[2]^2 - 7)^2
    m = Model(f)
    addvar!(m, [-5.0, -5.0], [5.0, 5.0])

    options = DeflationOptions(
        sub_options=i -> IpoptOptions(first_order=true), iters=2, keep_all=false
    )
    r = NonconvexMultistart.optimize(
        m, DeflatedAlg(IpoptAlg()), [0.0, 0.0], options=options
    )
    @test r.converged
    @test r.results === nothing
    @test length(r.minimizers) == 2

    # Reusing the workspace restarts the deflation from scratch and gives
    # the same solutions.
    vm, vx0, _ = NonconvexCore.tovecmodel(m, [0.0, 0.0])
    w = Workspace(vm, DeflatedAlg(IpoptAlg()), vx0; options=options)
    r1 = NonconvexMultistart.optimize!(w)
    r2 = NonconvexMultistart.optimize!(w)
    @test length(r2.minimizers) == length(r1.minimizers)
    @test r2.minimizers == r1.minimizers
end

@testitem "Deflation without an explicit initial point" begin
    using NonconvexMultistart, NonconvexIpopt, Test

    f(x) = (x[1]^2 + x[2] - 11)^2 + (x[1] + x[2]^2 - 7)^2
    m = Model(f)
    addvar!(m, [-5.0, -5.0], [5.0, 5.0]; init=[-5.0, -5.0])
    options = DeflationOptions(sub_options=IpoptOptions(first_order=true), iters=1)
    r = NonconvexMultistart.optimize(m, DeflatedAlg(IpoptAlg()); options=options)
    @test r.minimum < 1e-4
end

@testitem "Deflation capability checks" begin
    using NonconvexMultistart, NonconvexIpopt, Test

    @test solver_capabilities(DeflatedAlg(IpoptAlg())).nonlinear_ineq === true

    # A sub-solver without nonlinear inequality constraint support cannot
    # run the deflated sub-problems and must fail with a clear error.
    struct NoIneqAlg <: NonconvexCore.AbstractOptimizer end
    NonconvexCore.solver_capabilities(::NoIneqAlg) = ConstraintCapabilities(
        linear_eq=false,
        nonlinear_eq=false,
        linear_ineq=false,
        nonlinear_ineq=false,
        box=true,
        continuous=true,
        binary=false,
        integer=false,
        infinite_bounds=false,
    )
    f(x) = x[1]^2
    m = Model(f)
    addvar!(m, -1.0, 1.0)
    options = DeflationOptions(sub_options=nothing, iters=2)
    @test_throws ArgumentError NonconvexMultistart.optimize(
        m, DeflatedAlg(NoIneqAlg()), [0.5], options=options
    )
end

@testitem "Deflation options validation" begin
    using NonconvexMultistart, Test

    @test_throws ArgumentError DeflationOptions(sub_options=nothing, p=0.0)
    @test_throws ArgumentError DeflationOptions(sub_options=nothing, iters=0)
    @test_throws ArgumentError DeflationOptions(sub_options=nothing, radius=-1.0)
    @test_throws ArgumentError DeflationOptions(sub_options=nothing, sigma=-1.0)
    @test_throws ArgumentError DeflationOptions(sub_options=nothing, tol=0.0)
    @test_throws ArgumentError DeflationOptions(
        sub_options=nothing, distance=(x, xk) -> 0.0, weight=1.0
    )
end

@testitem "Deflation multistart finds all minimizers" begin
    using NonconvexMultistart, NonconvexIpopt, LinearAlgebra, Test

    # The multistart sampler explores the whole box, so all 4 Himmelblau
    # minima are found even though a single local solve from (0, 0) only
    # reaches 3 of them (see the plain deflation tests above).
    f(x) = (x[1]^2 + x[2] - 11)^2 + (x[1] + x[2]^2 - 7)^2
    minima = [
        (3.0, 2.0), (-2.805118, 3.131312), (-3.779310, -3.283186), (3.584428, -1.848126)
    ]
    m = Model(f)
    addvar!(m, [-5.0, -5.0], [5.0, 5.0])
    multi_options = HyperoptOptions(
        sub_options=IpoptOptions(first_order=true),
        iters=16,
        searchspace_size=16,
        sampler=LHSampler(),
    )
    options = DeflationMultistartOptions(multi_options=multi_options, iters=4)
    r = NonconvexMultistart.optimize(
        m, DeflationMultistartAlg(HyperoptAlg(IpoptAlg())), [0.0, 0.0], options=options
    )
    @test r.converged
    @test r.niters == 4
    @test length(r.minimizers) == 4
    @test r.minimum < 1e-6
    for x in r.minimizers
        @test f(x) < 1e-6
        @test minimum(norm(x .- collect(v)) for v in minima) < 1e-3
    end
    # All 4 distinct minima are found.
    found_inds = Set(argmin(norm(x .- collect(v)) for v in minima) for x in r.minimizers)
    @test length(found_inds) == 4
    for i in 1:4, j in (i + 1):4
        @test norm(r.minimizers[i] .- r.minimizers[j]) > 1e-2
    end
    @test length(r.results) == 4
end

@testitem "Deflation multistart stops when no new solution exists" begin
    using NonconvexMultistart, NonconvexIpopt, Test

    # x^2 has a single minimizer: even with multistart restarts, the
    # deflated sub-problem can only converge to the boundary of the
    # exclusion region around 0, and the run must stop.
    f(x) = x[1]^2
    m = Model(f)
    addvar!(m, -5.0, 5.0)
    multi_options = HyperoptOptions(
        sub_options=IpoptOptions(first_order=true),
        iters=8,
        searchspace_size=8,
        sampler=LHSampler(),
    )
    options = DeflationMultistartOptions(multi_options=multi_options, iters=3)
    r = NonconvexMultistart.optimize(
        m, DeflationMultistartAlg(HyperoptAlg(IpoptAlg())), [2.0], options=options
    )
    @test !r.converged
    @test r.niters == 2
    @test length(r.minimizers) == 1
    @test abs(r.minimizers[1][1]) < 1e-4
    @test r.minimum < 1e-6
end

@testitem "Deflation multistart keep_all and reset" begin
    using NonconvexMultistart, NonconvexIpopt, Test

    f(x) = (x[1]^2 + x[2] - 11)^2 + (x[1] + x[2]^2 - 7)^2
    m = Model(f)
    addvar!(m, [-5.0, -5.0], [5.0, 5.0])
    multi_options = HyperoptOptions(
        sub_options=IpoptOptions(first_order=true),
        iters=16,
        searchspace_size=16,
        sampler=LHSampler(),
    )
    options = DeflationMultistartOptions(
        multi_options=multi_options, iters=2, keep_all=false
    )
    r = NonconvexMultistart.optimize(
        m, DeflationMultistartAlg(HyperoptAlg(IpoptAlg())), [0.0, 0.0], options=options
    )
    @test r.converged
    @test r.results === nothing
    @test length(r.minimizers) == 2

    # Reusing the workspace restarts the deflation from scratch and gives
    # the same solutions.
    vm, vx0, _ = NonconvexCore.tovecmodel(m, [0.0, 0.0])
    w = Workspace(vm, DeflationMultistartAlg(HyperoptAlg(IpoptAlg())), vx0; options=options)
    r1 = NonconvexMultistart.optimize!(w)
    r2 = NonconvexMultistart.optimize!(w)
    @test length(r2.minimizers) == length(r1.minimizers)
    @test r2.minimizers == r1.minimizers
end

@testitem "Deflation multistart capability checks" begin
    using NonconvexMultistart, NonconvexIpopt, Test

    alg = DeflationMultistartAlg(HyperoptAlg(IpoptAlg()))
    @test solver_capabilities(alg).nonlinear_ineq === true

    # A multistart algorithm whose sub-solver cannot handle nonlinear
    # inequality constraints cannot run the deflated sub-problems.
    struct NoIneqAlg <: NonconvexCore.AbstractOptimizer end
    NonconvexCore.solver_capabilities(::NoIneqAlg) = ConstraintCapabilities(
        linear_eq=false,
        nonlinear_eq=false,
        linear_ineq=false,
        nonlinear_ineq=false,
        box=true,
        continuous=true,
        binary=false,
        integer=false,
        infinite_bounds=false,
    )
    f(x) = x[1]^2
    m = Model(f)
    addvar!(m, -1.0, 1.0)
    options = DeflationMultistartOptions(multi_options=nothing, iters=2)
    @test_throws ArgumentError NonconvexMultistart.optimize(
        m, DeflationMultistartAlg(HyperoptAlg(NoIneqAlg())), [0.5], options=options
    )
end

@testitem "Deflation multistart options validation" begin
    using NonconvexMultistart, Test

    @test_throws ArgumentError DeflationMultistartOptions(multi_options=nothing, p=0.0)
    @test_throws ArgumentError DeflationMultistartOptions(multi_options=nothing, iters=0)
    @test_throws ArgumentError DeflationMultistartOptions(
        multi_options=nothing, radius=-1.0
    )
    @test_throws ArgumentError DeflationMultistartOptions(multi_options=nothing, sigma=-1.0)
    @test_throws ArgumentError DeflationMultistartOptions(multi_options=nothing, tol=0.0)
    @test_throws ArgumentError DeflationMultistartOptions(
        multi_options=nothing, distance=(x, xk) -> 0.0, weight=1.0
    )
end
