using Random
using LinearAlgebra
using FileIO
using PyPlot
using Printf
include("../../src/DictionaryLearning/ksvd.jl")
include("../../src/DictionaryLearning/utils.jl")
include("../../src/MP/OMP.jl")


random_state = 0
rng = MersenneTwister(random_state)
patch_size = 8
step = 4
max_patches = 25000
sig = 20
m = 16
k0 = 4
n_iter = 15

dir = dirname(@__FILE__)

img_org = convert(Matrix{Float64}, load(dir * "/../../data/barbara.png")) .* 255
# add noise
img = img_org + randn(rng, size(img_org)...) .* sig
psnr = get_psnr(img_org, img)

PyPlot.imshow(img, cmap=:gray, vmin=0, vmax=255)
PyPlot.title(@sprintf("PSNR=%.03f", psnr))
PyPlot.colorbar()
PyPlot.savefig(dir * "/barbara_noise.png")
PyPlot.close()

# dct dictionary
A = generate_dct_dictionary(patch_size, m)
A = mapslices(normalize, A, dims=1)

# extract_patches
patches_2d = extract_patches_2d(img, patch_size, max_patches, rng)
patches_1d = cvtPatches_2dto1d(patches_2d)

show_dict(A, patch_size)
PyPlot.savefig(dir * @sprintf("/dict_dct_%d_%d.png", max_patches, step))
PyPlot.close()

# activity
t = 0.2
acts = get_activity(A, patch_size)
S_texture = acts .> t
S_component = acts .<= t

# image reconstruction
X_pred = sparse_coding(img, A, k0, sig, patch_size, step)
patches_1d_recon_texture = A[:, S_texture] * X_pred[S_texture, :]
patches_1d_recon_component = A[:, S_component] * X_pred[S_component, :]

patches_2d_recon_texture = cvtPatches_1dto2d(patches_1d_recon_texture,
                                             patch_size)
patches_2d_recon_component = cvtPatches_1dto2d(patches_1d_recon_component,
                                               patch_size)
img_recon_texture = reconstruct_from_patches_2d(patches_2d_recon_texture,
                                                size(img),
                                                step)
img_recon_component = reconstruct_from_patches_2d(patches_2d_recon_component,
                                                  size(img),
                                                  step)

PyPlot.imshow(img_recon_texture, cmap=:gray)
PyPlot.title("texture")
PyPlot.colorbar()
PyPlot.savefig(dir * @sprintf("/barbara_recon_texture_%d_%d.png", max_patches, step))
PyPlot.close()

PyPlot.imshow(img_recon_component, cmap=:gray, vmin=0, vmax=255)
PyPlot.title("component")
PyPlot.colorbar()
PyPlot.savefig(dir * @sprintf("/barbara_recon_component_%d_%d.png", max_patches, step))
PyPlot.close()