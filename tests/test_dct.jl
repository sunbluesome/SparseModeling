
using Test
using Random
using LinearAlgebra
using PyPlot
include("../src/DictionaryLearning/utils.jl")

patch_size = 8
n_atom = 11

A_2d = generate_dct_dictionary(patch_size, n_atom)

function show_dict(A::AbstractMatrix{T};
                   figsize::Tuple{U, U}=(4, 4),
                   vmin::S=nothing,
                   vmax::S=nothing) where {T <: Real, U <: Integer, S <: Union{Nothing, Real}}
    n = Int(sqrt(size(A)[1]))
    m = Int(sqrt(size(A)[2]))
    A_show = reshape(A, n, n, m, m)
    fig, axes = subplots(m, m, figsize=figsize)
    for row in 1:m
        for col in 1:m
            axes[row, col].imshow(A_show[:, :, col, row],
                                  cmap=:gray,
                                  interpolation=:Nearest,
                                  vmin=vmin,
                                  vmax=vmax)
            axes[row, col].axis(:off)
        end
    end
    fig
end

show_dict(A_2d)

