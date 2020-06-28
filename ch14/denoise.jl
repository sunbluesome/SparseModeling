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
step = 4
max_patches = 25000
sig = 20
m = 256
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
patches_2d = extract_patches_2d(img, patch_size, max_patches, rng)
patches_1d = cvtPatches_2dto1d(patches_2d)

# ksvd
ksvd = KSVD(sig, m, k0, PSVD())
# A, X, log = predict(ksvd, patches_1d; initial_dictionary=A, n_iter=n_iter)
A, X, log = predict(ksvd, patches_1d; n_iter=n_iter)

show_dict(A, patch_size)
PyPlot.savefig(dir * @sprintf("/dict_ksvd_%d.png", step))
PyPlot.close()

# image reconstruction
X_pred = sparse_coding(img, A, k0, sig, patch_size, step)
patches_1d_recon = A*X_pred

patches_2d_recon = cvtPatches_1dto2d(patches_1d_recon, patch_size)
img_recon = reconstruct_from_patches_2d(patches_2d_recon, size(img), step)

PyPlot.imshow(img_recon, cmap=:gray, vmin=0, vmax=255)
PyPlot.colorbar()
PyPlot.savefig(dir * @sprintf("/barbara_recon_%d.png", step))
PyPlot.close()
