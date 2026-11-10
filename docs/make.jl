using DocumenterFragments: build_fragment

# `@ref` links to NonconvexCore bindings (e.g. `optimize`) only resolve in
# the assembled Nonconvex.jl documentation site, so tolerate broken
# cross-references and docstring lookups in this standalone preview build.
build_fragment(
    @__DIR__; checkdocs=:public, doctest=false, warnonly=[:cross_references, :docs_block]
)
