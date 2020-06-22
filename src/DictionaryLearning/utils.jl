using Statistics

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

function extract_patches_2d(img::AbstractMatrix{T}, patch_size::Integer) where {T <: Real}
    yy = 1:(size(img)[1] - patch_size + 1)
    xx = 1:(size(img)[2] - patch_size + 1)
    n_p = xx[end] * yy[end]
    patches = zeros(n_p, patch_size, patch_size)
    for (k,(i,j)) in zip(1:n_p, Iterators.product(yy, xx))
        patches[k,:,:] = img[i:(i - 1) + patch_size, j:(j - 1) + patch_size]
    end
    patches
end

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


function reconstruct_from_patches_2d(patches::AbstractMatrix{T}, img_size::Tuple{U, U}) where {T <: Real, U <: Integer}
    i_h, i_w = img_size[1:2]
    p_h, p_w = size(patches)[1:2]
    img = zeros(img_size)

    n_h = i_h - p_h + 1
    n_w = i_w - p_w + 1
    for (p, (i, j)) in zip(patches, Iterators.product(1:n_h, 1:n_w))
        img[i:i + p_h, j:j + p_w] += p
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