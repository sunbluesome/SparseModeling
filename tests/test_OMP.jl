using Test
using Random
using LinearAlgebra
using Plots
pyplot()
include("../src/MP/OMP.jl")

random_state = 0
rng = MersenneTwister(random_state)
n_components = 512
n_features = 100
n_nonzero_coef = 17

# generate dictionary
D = randn(n_features, n_components)
D = mapslices(normalize, D, dims=1)

# generate sparse code
x0 = zeros(n_components)
idxs = shuffle(1:n_components)[1:n_nonzero_coef]
x0[idxs] .= randn(n_nonzero_coef)

# encode signal
y = vec(D * x0)

omp = OMP(n_nonzero_coef, 0)
x, S = predict(omp, D, y)

p1 = plot(x0; label = :original, t = :bar, leg=true)
p2 = plot(x0; label = :predict, t = :bar, leg=true)
l = @layout([a; b])
plot(p1, p2; layout = l)
dir = dirname(@__FILE__)
savefig(dir * "/omp_test.png")
close()

@testset "OMP" begin
    @test x0 ≈ x atol=1e-8
    @test sum(S) == n_nonzero_coef
end;
