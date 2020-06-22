using LinearAlgebra
using LowRankApprox
using Statistics
using Printf
include("../MP/OMP.jl")

abstract type PredictMethod end
mutable struct ApproxSVD <: PredictMethod end
mutable struct PSVD <: PredictMethod end

mutable struct KSVD
   sig::Float64
   m::Int32
   k0::Int32
   method::PredictMethod
end
KSVD(sig::Real, m::Integer, k0::Integer) = KSVD(Float64(sig), Int32(m), Int32(k0), ApproxSVD())

function predict(ksvd::KSVD,
                 Y::AbstractMatrix;
                 initial_dictionary::Union{AbstractMatrix, Nothing}=nothing,
                 n_iter::Integer=50)
    if isnothing(initial_dictionary)
        temp = Y[:, 1:ksvd.m]
        A = mapslices(normalize, temp, dims=1)
    else
        A = initial_dictionary
    end

    X = zeros(size(A)[2], size(Y)[2])
    eps = size(A)[1] * (ksvd.sig^2)
    omp = OMP(ksvd.k0, eps)

    log = zeros(n_iter)
    for k in 1:n_iter
        for i in 1:size(Y)[2]
            X[:, i], _ = predict(omp, A, Y[:, i])
        end

        for j in 1:ksvd.m
            support = findall(X[j, :] .!= 0)
            length(support) == 0 && continue

            if typeof(ksvd.method) == ApproxSVD
                A[:, j] .= 0
                residual_err = Y[:, support] - A * X[:, support]
                g = X[j, support]
                d = residual_err * g
                d = normalize(d)
                g = residual_err * d
                A[:, j] .= d
                X[j, support] .= g
            else
                X[j, support] .= 0
                residual_err = Y[:, support] - A * X[:, support]
                U, s, V = psvd(residual_err)
                A[:, j] = U[:, 1]
                X[j, support] = s[1] .* V[:, 1]
            end
        end
        opt = mean(abs.(Y - A * X))
        log[k] = opt
        s = @sprintf("%d:\t %f", k, log[k])
        println(s)
    end
    return A, X, log
end

function predict(ksvd::KSVD,
                 Y::AbstractMatrix,
                 A0::AbstractMatrix;
                 n_iter::Integer=50,
                 threshold=0.95)

    temp = Y[:, 1:ksvd.m]
    A = mapslices(normalize, temp, dims=1)

    X = zeros(size(A)[2], size(Y)[2])
    eps = size(A)[1] * (ksvd.sig^2)
    omp = OMP(ksvd.k0, eps)

    log = zeros(n_iter, 2)
    for k in 1:n_iter
        for i in 1:size(Y)[2]
            X[:, i], _ = predict(omp, A, Y[:, i])
        end

        for j in 1:ksvd.m
            support = findall(X[j, :] .!= 0)
            length(support) == 0 && continue

            if typeof(ksvd.method) == ApproxSVD
                A[:, j] .= 0
                residual_err = Y[:, support] - A * X[:, support]
                g = X[j, support]
                d = residual_err * g
                d = normalize(d)
                g = residual_err * d
                A[:, j] .= d
                X[j, support] .= g
            else
                X[j, support] .= 0
                residual_err = Y[:, support] - A * X[:, support]
                U, s, V = psvd(residual_err)
                A[:, j] = U[:, 1]
                X[j, support] = s[1] .* V[:, 1]
            end
        end
        opt = mean(abs.(Y .- A * X))
        opt2 = percent_recovery_of_atoms(A, A0, threshold=threshold)
        log[k, :] .= [opt, opt2]
        s = @sprintf("%d:\t %f, %f", k, opt, opt2)
        println(s)
    end
    return A, X, log
end

function predict(ksvd::KSVD,
                 Y::AbstractMatrix,
                 A0::AbstractMatrix,
                 initial_dictionary::AbstractMatrix;
                 n_iter::Integer=50,
                 threshold=0.95)

    A = initial_dictionary

    X = zeros(size(A)[2], size(Y)[2])
    eps = size(A)[1] * (ksvd.sig^2)
    omp = OMP(ksvd.k0, eps)

    log = zeros(n_iter, 2)

    for k in 1:n_iter
        for i in 1:size(Y)[2]
            X[:, i], _ = predict(omp, A, Y[:, i])
        end

        for j in 1:ksvd.m
            support = findall(X[j, :] .!= 0)
            length(support) == 0 && continue

            if typeof(ksvd.method) == ApproxSVD
                A[:, j] .= 0
                residual_err = Y[:, support] - A * X[:, support]
                g = X[j, support]
                d = residual_err * g
                d = normalize(d)
                g = residual_err * d
                A[:, j] .= d
                X[j, support] .= g
            else
                X[j, support] .= 0
                residual_err = Y[:, support] - A * X[:, support]
                U, s, V = psvd(residual_err)
                A[:, j] = U[:, 1]
                X[j, support] = s[1] .* V[:, 1]
            end
        end

        opt = mean(abs.(Y .- A * X))
        opt2 = percent_recovery_of_atoms(A, A0, threshold=threshold)
        log[k, :] .= [opt, opt2]
        s = @sprintf("%d:\t %f, %f", k, opt, opt2)
        println(s)
    end
    return A, X, log
end


"""
KSVD with missing value.
"""
function predict(ksvd::KSVD,
                 Y::AbstractMatrix,
                 missingValue::Real;
                 initial_dictionary::Union{AbstractMatrix, Nothing}=nothing,
                 n_iter::Integer=50)
    if isnothing(initial_dictionary)
        temp = Y[:, 1:ksvd.m]
        A = mapslices(normalize, temp, dims=1)
    else
        A = initial_dictionary
    end

    mask_valid = Y .!= missingValue
    mask_invalid = .!mask_valid
    X = zeros(size(A)[2], size(Y)[2])
    omp = OMP(ksvd.k0)

    log = zeros(n_iter)
    for k in 1:n_iter
        for i in 1:size(Y)[2]
            omp.eps = sum(mask_valid[:, i]) * (ksvd.sig^2) * 1.1
            y = @view Y[mask_valid[:, i], i]
            As = @view A[mask_valid[:, i], :]
            X[:, i], _ = predict(omp, As, y)

            # inpainting
            Y[mask_invalid[:, i], i] = A[mask_invalid[:, i], :] * X[:, i]
        end

        for j in 1:ksvd.m
            support = findall(X[j, :] .!= 0)
            length(support) == 0 && continue

            if typeof(ksvd.method) == ApproxSVD
                A[:, j] .= 0
                residual_err = Y[:, support] - A * X[:, support]
                g = X[j, support]
                d = residual_err * g
                d = normalize(d)
                g = residual_err * d
                A[:, j] .= d
                X[j, support] .= g
            else
                X[j, support] .= 0
                residual_err = Y[:, support] - A * X[:, support]
                U, s, V = psvd(residual_err)
                A[:, j] = U[:, 1]
                X[j, support] = s[1] .* V[:, 1]
            end
        end
        opt = mean(abs.(Y[mask_valid] - (A * X)[mask_valid]))
        log[k] = opt
        s = @sprintf("%d:\t %f", k, log[k])
        println(s)
    end
    return A, X, log
end

# DCT
function generate_dct(patch_size::Integer, dict_size::Integer)
    A_1D = zeros(patch_size, dict_size)
    for k in 1:dict_size
        for i in 1:patch_size
            A_1D[i, k] = cos(i * (k-1) * pi / dict_size)
        end

        if k != 1
            A_1D[:, k] .-= mean(A_1D[:, k])
        end
    end
    kron(A_1D, A_1D)
end