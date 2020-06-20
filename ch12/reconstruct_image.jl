using Random
using LinearAlgebra
using Images
using ImageView
using Plots
gr()
using Printf
include("../src/DictionaryLearning/ksvd.jl")
include("../src/DictionaryLearning/utils.jl")
include("../src/MP/OMP.jl")


random_state = 0
rng = MersenneTwister(random_state)
patch_size = 8
step = patch_size
max_patches = 25500
sig = 20
spike_frac = 0.25
m = 256
k0 = 4
n_iter = 15
n_iter_ksvd = 25
missingValue = 0

dir = dirname(@__FILE__)
img_org = channelview(load(dir * "/barbara.png")) .* 255
img_noise = img_org + randn(rng, size(img_org)...) .* sig
mask = rand(rng, size(img_org)) .> spike_frac

# add spike
img = img_noise .* mask


heatmap(img, yflip=true, c=:grays, clim=(0, 255), aspect_ratio=1)
dir = dirname(@__FILE__)
savefig(dir * "/barbara_spike.png")

A = generate_dct(patch_size, m)
for i in 1:n_iter_ksvd
    # extract_patches
    patches = extract_patches_2d(img, patch_size, max_patches, rng)
    patches_1d = zeros(patch_size^2, max_patches)
    for i in 1:max_patches
        patches_1d[:, i] .= vec(patches[i, :, :])
    end

    # ksvd
    ksvd = KSVD(sig, m, k0)
    global A, X, log = predict(ksvd, patches_1d, missingValue; initial_dictionary=A, n_iter=n_iter)
end

# image reconstruction
X_pred = sparse_coding_with_mask(img, A, k0, sig, mask, patch_size)
patches_re = A*X_pred

img_re = reconstruct_from_patches_2d(patches_re, size(img)

heatmap(img_re, yflip=true, c=:grays, clim=(0, 255), aspect_ratio=1)
dir = dirname(@__FILE__)
savefig(dir * "/barbara_recon.png")
