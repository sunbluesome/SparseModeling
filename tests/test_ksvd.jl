using Test
using Random
using LinearAlgebra
# using PyPlot
include("../src/DictionaryLearning/ksvd.jl")
include("../src/DictionaryLearning/utils.jl")


random_state = 0
sig = 0.1
k0 = 4
n_iter = 50

rng = MersenneTwister(random_state)
A0 = randn(rng, 30, 60)
A0 = mapslices(normalize, A0, dims=1)
Y = zeros(30, 4000)

# generate data
for i in 1:size(Y)[2]
    S = shuffle(rng, 1:size(A0)[2])[1:k0]
    Y[:, i] = A0[:, S] * randn(rng, k0) + randn(30) .* sig
end

ksvd = KSVD(sig, size(A0)[2], k0)
A, X, log = predict(ksvd, Y, A0; n_iter=n_iter, threshold=0.99)

score = percent_recovery_of_atoms(A, A0)

@test score > 80