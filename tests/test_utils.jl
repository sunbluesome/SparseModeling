using Test
include("../src/DictionaryLearning/utils.jl")

img = reshape(1:64^2, 64, 64)
patch_size = 8
step = 4

patches_2d = extract_patches_2d_step(img, patch_size, step)
patches_1d = cvtPatches_2dto1d(patches_2d)
patches_2d_recon = cvtPatches_1dto2d(patches_1d, patch_size)
img_recon = reconstruct_from_patches_2d(patches_2d_recon, size(img), step)

@testset "utils" begin
    @test patches_2d == patches_2d_recon
    @test img == img_recon
end;