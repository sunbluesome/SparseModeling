using Statistics
using PyPlot
using Random

# based on mutual coherence
function percent_recovery_of_atoms(A::AbstractMatrix{T},
                                   A0::AbstractMatrix{U};
                                   threshold=0.99) where {T <: Real, U <: Real}
    num = 0
    for m in 1:size(A0)[2]
        a = A0[:, m]
        if maximum(abs.(a' * A)) > threshold
            num += 1
        end
    end
    100 * num / size(A)[2]
end


function extract_patches_2d_step(img::AbstractMatrix{T}, patch_size::Integer, step::Integer) where {T <: Real}
    img_h = size(img)[1]
    img_w = size(img)[2]
    @assert mod(img_h - patch_size, step) == 0 && mod(img_w - patch_size, step) == 0

    yy = 1:step:(img_h - patch_size + 1)
    xx = 1:step:(img_w - patch_size + 1)
    n_p = length(xx) * length(yy)
    patches = zeros(n_p, patch_size, patch_size)
    for (k,(i,j)) in zip(1:n_p, Iterators.product(yy, xx))
        patches[k,:,:] = img[i:(i - 1) + patch_size, j:(j - 1) + patch_size]
    end
    patches
end
extract_patches_2d_step(img::AbstractMatrix{T},
                        patch_size::Integer) where {T <: Real} = extract_patches_2d_step(img, patch_size, 1)


function extract_patches_2d(img::AbstractMatrix{T}, patch_size::Integer,
                            max_patches::Integer, rng::AbstractRNG) where {T <: Real}
    yy = 1:(size(img)[1] - patch_size + 1)
    xx = 1:(size(img)[2] - patch_size + 1)
    n_p = xx[end] * yy[end]
    patches = zeros(n_p, patch_size, patch_size)
    for (k,(i,j)) in zip(1:n_p, Iterators.product(yy, xx))
        patches[k,:,:] = img[i:(i - 1) + patch_size, j:(j - 1) + patch_size]
    end

    max_patches = min(size(patches)[1], max_patches)
    idxs = shuffle(rng, 1:max_patches)
    return patches[idxs, :, :]
end
extract_patches_2d(img::AbstractMatrix{T}, patch_size::Integer,
                   max_patches::Integer) where {T <: Real} = extract_patches_2d(img, patch_size, max_patches, MersenneTwister())

function cvtPatches_2dto1d(patches_2d::AbstractArray{T, 3}) where {T <: Real}
    n_p, p_h, p_w = size(patches_2d)
    patches_1d = zeros(p_h * p_w, n_p)
    for j in 1:n_p
        patches_1d[:, j] .= vec(patches_2d[j, :, :])
    end
    patches_1d
end

function cvtPatches_1dto2d(patches_1d::AbstractArray{T, 2}, patch_size::Tuple{U, U}) where {T <: Real, U <: Real}
    patches_2d = reshape(patches_1d, patch_size[1], patch_size[2], :)
    permutedims(patches_2d, [3, 1, 2])
end
cvtPatches_1dto2d(patches_1d::AbstractMatrix{T}, patch_size::Integer) where {T <: Real} = cvtPatches_1dto2d(patches_1d, (patch_size, patch_size))


function reconstruct_from_patches_1d(patches_1d::AbstractMatrix{T},
                                     img_size::Tuple{U, U},
                                     patch_size::Tuple{S, S}) where {T <: Real, U <: Integer, S <: Integer}
    i_h, i_w = img_size
    p_h, p_w = patch_size
    img = zeros(img_size)
    patches_2d = reshape(patches_1d, p_h, p_w, size(patches_1d)[2])

    n_h = i_h - p_h + 1
    n_w = i_w - p_w + 1
    for (k, (i, j)) in enumerate(Iterators.product(1:n_h, 1:n_w))
        img[i:(i-1) + p_h, j:(j-1) + p_w] .+= patches_2d[:, :, k]
    end

    for i in 1:i_h
        for j in 1:i_w
            img[i, j] /= minimum([i, p_h, i_h - (i - 1)]) * minimum([j, p_w, i_w - (j - 1)])
        end
    end
    img
end

function reconstruct_from_patches_2d(patches_2d::AbstractArray{T, 3},
                                     img_size::Tuple{U, U},
                                     step::Integer) where {T <: Real, U <: Integer}
    i_h, i_w = img_size[1:2]
    p_h, p_w = size(patches_2d)[2:3]
    @assert mod(i_h - p_h, step) == 0 && mod(i_w - p_w, step) == 0
    img = zeros(img_size)
    img_cnt = zeros(img_size)

    n_h = i_h - p_h + 1
    n_w = i_w - p_w + 1

    yy = 1:step:(i_h - p_h + 1)
    xx = 1:step:(i_w - p_w + 1)
    n_p = length(xx) * length(yy)
    for (k,(i,j)) in zip(1:n_p, Iterators.product(yy, xx))
        img[i:(i - 1) + p_h, j:(j - 1) + p_w] .+= patches_2d[k, :, :]
        img_cnt[i:(i - 1) + p_h, j:(j - 1) + p_w] .+= 1
    end

    img ./ img_cnt
end

function reconstruct_from_patches_2d(patches_2d::AbstractArray{T, 3}, img_size::Tuple{U, U}) where {T <: Real, U <: Integer}
    i_h, i_w = img_size[1:2]
    p_h, p_w = size(patches_2d)[2:3]
    img = zeros(img_size)

    n_h = i_h - p_h + 1
    n_w = i_w - p_w + 1
    for (k, (i, j)) in enumerate(Iterators.product(1:n_h, 1:n_w))
        img[i:(i-1) + p_h, j:(j-1) + p_w] .+= patches_2d[k, :, :]
    end

    for i in 1:i_h
        for j in 1:i_w
            img[i, j] /= minimum([i, p_h, i_h - (i - 1)]) * minimum([j, p_w, i_w - (j - 1)])
        end
    end
    img
end

function generate_dct_dictionary(patch_size::T, n_atom::U) where {T <: Integer, U <: Integer}
    A_1d = zeros(Float64, patch_size, n_atom)
    for k in 1:n_atom
        for i in 1:patch_size
            A_1d[i, k] = cos((i - 1) * (k - 1) * pi / 11)
        end
        if k != 1
            A_1d[:, k] .-= mean(A_1d[:, k])
        end
    end
    kron(A_1d, A_1d)
end


function show_dict(A::AbstractMatrix{T},
                   patch_size::Tuple{U, U};
                   figsize::Tuple{U, U}=(4, 4),
                   vmin::S=nothing,
                   vmax::S=nothing) where {T <: Real, U <: Integer, S <: Union{Nothing, Real}}
    m = Int(ceil(sqrt(size(A)[2])))
    patches_2d = cvtPatches_1dto2d(A, patch_size)
    fig, axes = subplots(m, m, figsize=figsize)

    for i in 1:size(patches_2d)[1]
        axes[i].imshow(patches_2d[i, :, :],
                       cmap=:gray,
                       interpolation=:Nearest,
                       vmin=vmin,
                       vmax=vmax)
        axes[i].axis(:off)
    end
    fig
end
show_dict(A::AbstractMatrix{T}, patch_size::U) where {T <: Real, U <: Integer} = show_dict(A, (patch_size, patch_size))


# PSNR
function get_psnr(im, recon)
    10. * log(maximum(im) / sqrt(mean((im .- recon)^2)))
end