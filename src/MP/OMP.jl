using Printf
mutable struct OMP
    k::Int16
    eps::Float64
end
OMP(k::Integer) = OMP(Int16(k), 0)
OMP(k::Integer, eps::Real) = OMP(Int16(k), Float64(eps))

"""
predict method for OMP
"""
function predict(omp::OMP, A::AbstractMatrix{T}, b::AbstractVector{U}) where {T<:Real, U<:Real}
    m = size(A)[2]
    x = zeros(m)
    r = copy(b)
    rr = r' * r
    S = zeros(Int32, m)

    for i in 1:omp.k

        errs = ones(m) .* Inf
        for j in 1:m
            aj = @view A[:, j]
            zj = aj' * r / norm(aj)
            errs[j] = norm(aj * zj - r)
        end

        # update support
        S[argmin(errs)] = 1

        # update x
        support = findall(x -> x == 1, S)
        As = A[:, support]
        x[support] = As' * pinv(As * As') * b

        # update residual
        r = b - A * x
        if r' * r < omp.eps
            break
        end
    end

    (x, S)
end


function predict(omp::OMP, Afull::AbstractMatrix{T}, bfull::AbstractVector{U}, mask::BitArray{1}) where {T<:Real, U<:Real}
    b = @view bfull[mask]
    A = @view Afull[mask, :]

    m = size(A)[2]
    x = zeros(m)
    r = copy(b)
    rr = r' * r
    S = zeros(Int32, m)

    for i in 1:omp.k

        errs = ones(m) .* Inf
        for j in 1:m
            aj = @view A[:, j]
            zj = aj' * r / norm(aj)
            errs[j] = norm(aj * zj - r)
        end

        # update support
        S[argmin(errs)] = 1

        # update x
        support = findall(x -> x == 1, S)
        As = A[:, support]
        x[support] = As' * pinv(As * As') * b

        # update residual
        r = b - A * x
        if r' * r < omp.eps
            break
        end
    end

    (x, S)
end


function sparse_coding(img, A, k0, sig, patch_size, step)
    patches = extract_patches_2d(img, patch_size, step)
    n = length(patches[1,:,:])
    q = zeros(size(A)[2], size(patches)[1])

    omp = OMP(k0)
    for i in 1:size(patches)[1]
        if i % 1000 == 0
            @printf("process:\t %d\n", i)
        end
        omp.eps = n * (sig^2) * 1.1
        q[:, i], S = predict(omp, A, vec(patches[i, :, :]))
    end
    q
end
sparse_coding(img, A, k0, sig, patch_size) =  parse_coding(img, A, k0, sig, patch_size, 1)

function sparse_coding_with_mask(img, A, k0, sig, mask, patch_size, step)
    patches = extract_patches_2d(img, patch_size, step)
    mask_patches = extract_patches_2d(mask, patch_size, step)
    q = zeros(size(A)[2], size(patches)[1])

    omp = OMP(k0)
    for i in 1:size(patches)[1]
        if i % 1000 == 0
            @printf("process:\t %d\n", i)
        end
        A_mask = A[vec(mask_patches[i, :, :]) .== 1, :]
        patch_mask = patches[i,:,:][mask_patches[i, :, :] .== 1]
        omp.eps = length(patch_mask) * (sig^2) * 1.1
        q[:, i], S = predict(omp, A_mask, patch_mask)
    end
    q
end
sparse_coding_with_mask(img, A, k0, sig,
                        mask, patch_size) = sparse_coding_with_mask(img, A, k0, sig,
                                                                    mask, patch_size, 1) 
