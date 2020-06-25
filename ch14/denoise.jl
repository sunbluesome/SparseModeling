using Random
using LinearAlgebra
using FileIO
using PyPlot
using Printf
include("../src/DictionaryLearning/ksvd.jl")
include("../src/DictionaryLearning/utils.jl")
include("../src/MP/OMP.jl")


random_state = 0
rng = MersenneTwister(random_state)
patch_size = 8
step = Int(patch_size / 2)
max_patches = 2000
sig = 20
m = 16
k0 = 4
n_iter = 15

dir = dirname(@__FILE__)
img_org = convert(Matrix{Float64}, load(dir * "/../data/barbara.png")) .* 255
# add noise
img = img_org + randn(rng, size(img_org)...) .* sig

PyPlot.imshow(img, cmap=:gray, vmin=0, vmax=255)
PyPlot.colorbar()
PyPlot.savefig(dir * "/barbara_noise.png")
PyPlot.close()

# A = generate_dct_dictionary(patch_size, m)
# extract_patches
patches = extract_patches_2d(img, patch_size, max_patches, rng)
patches_1d = zeros(patch_size^2, max_patches)
for j in 1:max_patches
    patches_1d[:, j] .= vec(patches[j, :, :])
end

# ksvd
ksvd = KSVD(sig, m, k0, PSVD())
# A, X, log = predict(ksvd, patches_1d; initial_dictionary=A, n_iter=n_iter)
A, X, log = predict(ksvd, patches_1d; n_iter=n_iter)
show_dict(A)
PyPlot.savefig(dir * "/dict_ksvd.png")
PyPlot.close()

# image reconstruction
X_pred = sparse_coding(img, A, k0, sig, patch_size, step)
patches_re = A*X_pred

img_re = reconstruct_from_patches_1d(patches_re, size(img), (patch_size, patch_size))

PyPlot.imshow(img_re, cmap=:gray, vmin=0, vmax=255)
PyPlot.colorbar()
PyPlot.savefig(dir * @sprint("/barbara_recon_%d.png", step))
PyPlot.close()
