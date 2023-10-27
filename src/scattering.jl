using Dierckx

"""
    a₀ : Bohr radius (in cm)
"""
const a₀ = ustrip(BohrRadius |> u"cm") # 0.529 Å

"""
    Rₐ(elm::Element)

Classic formula for the atomic screening radius in cm
"""
Rₐ(elm::Element) = a₀ * z(elm)^-0.333333333333
Rₐ(elm::Vector{Element}) = a₀ .* z(elm)^-0.333333333333

"""
Algorithms implementing the elastic scattering cross-section

    σₜ(::Type{ScreenedRutherford}, elm::Element, E::Float64)
"""
abstract type ElasticScatteringCrossSection end

"""
Basic screened Rutherford algorithm where V(r) = (-Ze²/r)exp(-r/R) where R=a₀Z⁻¹/³ is solved using
the first Born approximation.
"""
abstract type ScreenedRutherfordType <: ElasticScatteringCrossSection end

struct ScreenedRutherford <: ScreenedRutherfordType end

"""
Liljequist's simple refinement of the basic ScreenedRutherford algorithm.

Journal of Applied Physics, 65, 24-31 (1989) as corrected in J. Appl. Phys. 68 (7) 3061-3065 (1990)
"""
struct Liljequist1989 <: ScreenedRutherfordType end

"""
Browning's scattering cross section according to a draft 1994 article
"""
struct Browning1994 <: ScreenedRutherfordType end

"""
Browning's scattering cross section

Appl. Phys. Lett. 58, 2845 (1991); https://doi.org/10.1063/1.104754
"""
struct Browning1991 <: ScreenedRutherfordType end


"""
    ξ(::Type{<:ScreenedRutherfordType}, elm::Element, E::Float64)::Float64 

  * E in eV
"""
function ξ(::Type{<:ScreenedRutherfordType}, elm::Element, E::Real)
    Rₑ, mc² = ustrip((PlanckConstant * SpeedOfLightInVacuum * RydbergConstant) |> u"eV"), ustrip(ElectronMass * SpeedOfLightInVacuum^2 |> u"eV")
    return 0.5 * π * a₀^2 * (4.0 * z(elm) * ((E + mc²) / (E + 2.0 * mc²)) * (Rₑ / E))^2 # As corrected in Liljequist1989
end 
function ξ(::Type{<:ScreenedRutherfordType}, elm::AbstractVector{Element}, E::AbstractVector{<:Real})
    Rₑ, mc² = ustrip((PlanckConstant * SpeedOfLightInVacuum * RydbergConstant) |> u"eV"), ustrip(ElectronMass * SpeedOfLightInVacuum^2 |> u"eV")
    return 0.5 * π * a₀^2 * (4.0 .* z.(elm) .* ((E .+ mc²) ./ (E .+ 2.0 .* mc²)) .* (Rₑ ./ E)).^2 # As corrected in Liljequist1989
end

"""
    ϵ(elm::Element, E::Float64)

Screening factor.
"""
ϵ(::Type{<:ScreenedRutherfordType}, elm::Element, E::Real) = 2.0 * (kₑ(E) * Rₐ(elm))^2
ϵ(::Type{<:ScreenedRutherfordType}, elm::Vector{Element}, E::Vector{<:Real}) = 2.0 * (kₑ(E) .* Rₐ(elm))^2
#kₑ just gives e wavenumber ß need to input vector of Es to this 
# Glen - need to implement this continuously


# A spline interpolation based on Liljequist Table III 
const LiljequistCorrection = begin
    zs = Float64[4, 6, 13, 20, 26, 29, 32, 38, 47, 50, 56, 64, 74, 79, 82]
    es =
        1000.0 * [
            0.1,
            0.15,
            0.2,
            0.3,
            0.4,
            0.5,
            0.7,
            1,
            1.5,
            2,
            3,
            4,
            5,
            7,
            10,
            15,
            20,
            30,
            40,
            50,
            70,
            100,
        ] # eV
    tbl3 = [
        1.257 1.211 1.188 1.165 1.153 1.147 1.139 1.134 1.129 1.127 1.123 1.121 1.119 1.116 1.113 1.111 1.11 1.11 1.11 1.112 1.115 1.119
        1.506 1.394 1.330 1.257 1.217 1.192 1.163 1.140 1.123 1.115 1.107 1.103 1.100 1.097 1.095 1.093 1.093 1.093 1.094 1.096 1.099 1.104
        3.14 2.589 2.301 1.993 1.824 1.714 1.576 1.458 1.352 1.294 1.23 1.196 1.175 1.15 1.132 1.118 1.111 1.105 1.103 1.103 1.104 1.107
        3.905 3.192 2.823 2.429 2.211 2.065 1.878 1.713 1.558 1.466 1.358 1.295 1.254 1.202 1.161 1.128 1.111 1.094 1.087 1.084 1.081 1.081
        7.061 5.076 4.102 3.207 2.786 2.536 2.24 1.997 1.781 1.655 1.508 1.421 1.363 1.288 1.225 1.17 1.141 1.111 1.097 1.088 1.08 1.076
        10.154 7.184 5.521 3.987 3.31 2.933 2.517 2.199 1.929 1.777 1.603 1.501 1.433 1.344 1.269 1.202 1.166 1.129 1.110 1.098 1.087 1.079
        13.213 8.783 6.638 4.674 3.796 3.309 2.781 2.39 2.068 1.891 1.691 1.575 1.497 1.397 1.31 1.232 1.189 1.144 1.12 1.106 1.09 1.08
        16.187 9.955 7.475 5.351 4.378 3.813 3.176 2.694 2.297 2.081 1.838 1.698 1.605 1.484 1.379 1.282 1.227 1.167 1.134 1.114 1.091 1.075
        16.265 13.22 9.985 6.781 5.391 4.624 3.785 3.156 2.643 2.365 2.057 1.882 1.765 1.615 1.483 1.36 1.289 1.207 1.162 1.133 1.098 1.071
        14.33 13.534 10.829 7.385 5.802 4.936 4.005 3.319 2.763 2.464 2.132 1.944 1.82 1.659 1.518 1.387 1.31 1.222 1.172 1.133 1.101 1.071
        15.97 13.946 11.072 7.779 6.189 5.285 4.293 3.554 2.951 2.624 2.259 2.052 1.913 1.735 1.579 1.433 1.347 1.247 1.189 1.14 1.105 1.068
        25.349 25.137 19.078 11.778 8.611 6.957 5.298 4.19 3.367 2.946 2.493 2.242 2.077 1.866 1.682 1.512 1.411 1.293 1.224 1.151 1.12 1.073
        43.544 47.121 37.079 21.021 13.996 10.545 7.322 5.369 4.072 3.464 2.849 2.524 2.315 2.053 1.828 1.622 1.501 1.358 1.274 1.178 1.145 1.085
        52.567 58.958 46.173 24.508 15.831 11.792 8.101 5.87 4.388 3.7 3.015 2.655 2.426 2.14 1.896 1.674 1.543 1.389 1.298 1.218 1.157 1.09
        43.737 52.101 45.521 25.950 16.794 12.472 8.537 6.160 4.576 3.843 3.115 2.735 2.493 2.193 1.938 1.705 1.568 1.407 1.312 1.247 1.164 1.093
    ]
    # Spline it first at each energy over Z
    zspl = [Spline1D(zs, tbl3[:, ie], k = 3) for ie in eachindex(es)]
    # Then use this to estimate the interpolation over the range Z=1 to 92
    [Spline1D(es, [zspl[ie](Float64(z)) for ie in eachindex(es)], k = 3) for z = 1:92]
end

"""
    σₜᵣ(::Type{ScreenedRutherford}, elm::Element, E::Float64)

The transport cross-section in cm².  The transport cross-section gives the correct transport
mean free path - the mean free path in the direction of initial propagation after an infinite
number of collisions.
"""
function σₜᵣ(::Type{ScreenedRutherford}, elm::Element, E::Real)
    ϵv = ϵ(ScreenedRutherford, elm, E)
    return ξ(ScreenedRutherford, elm, E) * (log(2.0 * ϵv + 1) - 2.0 * ϵv / (2.0 * ϵv + 1.0))
end
function σₜᵣ(::Type{Liljequist1989}, elm::Element, E::Real)
    return σₜᵣ(ScreenedRutherford, elm, E) / LiljequistCorrection[z(elm)](E)
end 

function σₜᵣ(::Type{ScreenedRutherford}, elm::AbstractVector{Element}, E::AbstractVector{<:Real})
    ϵv = ϵ(ScreenedRutherford, elm, E)
    return ξ(ScreenedRutherford, elm, E) .* (log(2.0 .* ϵv .+ 1) .- 2.0 .* ϵv / (2.0 .* ϵv .+ 1.0))
end
function σₜᵣ(::Type{Liljequist1989}, elm::AbstractVector{Element}, E::AbstractVector{<:Real})
    return σₜᵣ(ScreenedRutherford, elm, E) ./ LiljequistCorrection[z(elm)](E)
end 

"""
    σₜ(::Type{<:ElasticScatteringCrossSection}, elm::Element, E::Real)
    σₜ(::Type{<:ElasticScatteringCrossSection}, mat::AbstractMaterial, E::Real)

Total cross section per atom in cm².
"""
function σₜ(::Type{ScreenedRutherford}, elm::Element, E::T)::T where T
    ϵv = ϵ(ScreenedRutherford, elm, E)
    return ξ(ScreenedRutherford, elm, E) * (2.0 * ϵv^2 / (2.0 * ϵv + 1.0))
end
function σₜ(::Type{Liljequist1989}, elm::Element, E::T)::T where T
    return σₜ(ScreenedRutherford, elm, E) / LiljequistCorrection[z(elm)](E)
end
function σₜ(::Type{S}, mat::AbstractMaterial{T}, E::T)::T where {S<:ElasticScatteringCrossSection, T}
    res::T = zero(T)
    for elm in elms(mat)
        res += σₜ(S, mat, elm, E)
    end
    return res
    return sum(elm -> σₜ(S, mat, elm, E), elms(mat))
end
function _σₜ(::Type{S}, mat::AbstractMaterial{T}, E::T)::T where {S<:ElasticScatteringCrossSection, T}
    return sum(elm -> σₜ(S, mat, elm, E), elms(mat))
end
function _σₜ(::Type{S}, mat::VectorizedMaterial{<:Any, T}, E::T)::T where {S<:ElasticScatteringCrossSection, T}
    res::T = zero(T)
    for (elm, afrac) in zip(elms_vector(mat), atomicfracs(mat))
        res += σₜ(S, elm, E) * afrac
    end
    return res
    return sum(x -> σₜ(S, x[1], E) * x[2], zip(elms_vector(mat), atomicfracs(mat)))
end
function σₜ(::Type{S}, mat::VectorizedMaterial{<:Any, T}, E::T)::T where {S<:ElasticScatteringCrossSection, T}
    return sum(index -> σₜ(S, mat, index, E), eachindex(mat))
end
function __σₜ(::Type{S}, mat::VectorizedMaterial{<:Any, T}, E::T)::T where {S<:ElasticScatteringCrossSection, T}
    return sum(x -> σₜ(S, x[1], E) * x[2], zip(elms_vector(mat), atomicfracs(mat)))
end

"""
    σₜ(::Type{<:ElasticScatteringCrossSection}, mat::AbstractMaterial, elm::AbstractMaterial, E::Real)

Total cross section of `elm` per atom of `mat` in cm².
"""
function σₜ(::Type{S}, mat::AbstractMaterial{T}, elm::Element, E::T)::T where {S<:ElasticScatteringCrossSection, T}
    return σₜ(S, elm, E) * atomicfrac(mat, elm)
end

# Only use with eachindex(mat)
function σₜ(::Type{S}, mat::VectorizedMaterial{<:Any, T}, index::Integer, E::T)::T where {S<:ElasticScatteringCrossSection, T}
    return σₜ(S, elm_nocheck(mat, index), E) * atomicfrac(mat, index)
end

"""
    _allσₜ!(::Type{<:ElasticScatteringCrossSection}, σ::AbstractVector, mat::AbstractMaterial, E::Real, elms::AbstractVector{Element})::Vector
    _allσₜ!(::Type{<:ElasticScatteringCrossSection}, σ::AbstractVector, mat::VectorizedMaterial, E::Real)::Vector

Total cross section of each element per atom of `mat` in cm² as a vector.
"""
function _allσₜ!(::Type{S}, σ::AbstractVector{T}, mat::AbstractMaterial{T}, E::T, elms::AbstractVector{Element}) where {S<:ElasticScatteringCrossSection, T}
    σ .= _atomicfrac.(Ref(mat), elms)
    σ .*=  σₜ.(S, elms, E) ./ sum(σ)
    #return σₜ.(Ref(S), elms, Ref(E)) .* atomicfrac.(Ref(mat), elms)
end
function _allσₜ!(::Type{S}, σ::AbstractVector{T}, mat::VectorizedMaterial{N, T}, E::T, elms::AbstractVector{Element}) where {S<:ElasticScatteringCrossSection, N, T}
    σ .= σₜ.(S, elms, E) .* atomicfracs(mat)
end

function _allσₜ(::Type{S}, mat::AbstractMaterial{T}, E::T, elms::AbstractVector{Element}) where {S<:ElasticScatteringCrossSection, T}
    σ = MVector{length(elms), T} |> zero
    return _allσₜ!(S, σ, mat, E, elms)
end

function fracσₜ(::Type{S}, mat::AbstractMaterial{T}, E::T, elms::AbstractVector{Element}) where {S<:ElasticScatteringCrossSection, T}
    return σₜ.(S, Ref(mat), elms, E)
    #return SVector{length(mat), T}(σₜ(S, mat, elm, E) for elm in elms)
end
function fracσₜ(::Type{S}, mat::VectorizedMaterial{N, T}, E::T, elms::SVector{N, Element}) where {S<:ElasticScatteringCrossSection, N, T}
    σₜ.(S, elms, E) .* atomicfracs(mat)
end

# Vectorised form of everything
function σₜ(::Type{ScreenedRutherford},  elm::AbstractVector{Element}, E::AbstractVector{<:Real})
    ϵv = ϵ(ScreenedRutherford, elm, E)
    return ξ(ScreenedRutherford, elm, E) .* (2.0 .* ϵv^2 ./ (2.0 .* ϵv .+ 1.0))
end
function σₜ(::Type{Liljequist1989},  elm::AbstractVector{Element}, E::AbstractVector{<:Real})
    return σₜ(ScreenedRutherford, elm, E) ./ LiljequistCorrection[z(elm)](E)
end

function σₜ(::Type{Browning1991}, elm::Element, E::T)::T where T
    e = 0.001 * E
    u = log10(8.0 * e * z(elm)^-1.33)
    return 4.7e-18 * (z(elm)^1.33 + 0.032 * z(elm)^2) / (
        (e + 0.0155 * (z(elm)^1.33) * sqrt(e)) * (1.0 - 0.02 * sqrt(z(elm)) * exp(-u^2))
    )
end
function σₜ(::Type{Browning1994}, elm::Element, E::T) where T
    e = 0.001 * E
    return 3.0e-18 * z(elm)^1.7 /
           (e + 0.005 * z(elm)^1.7 * sqrt(e) + 0.0007 * z(elm)^2 / sqrt(e))
end


"""
    δσδΩ(::Type{ScreenedRutherford}, θ::Float64, elm::Element, E::Float64)::Float64

The *differential* screened Rutherford cross-section per atom. 
"""
function δσδΩ(::Type{ScreenedRutherford}, θ::T, elm::Element, E::T)::T where T
    return ξ(ScreenedRutherford, elm, E) *
           (1.0 - cos(θ) + ϵ(ScreenedRutherford, elm, E)^-1)^-2
end
function δσδΩ(::Type{Liljequist1989}, θ::T, elm::Element, E::T)::T where T
    return δσδΩ(ScreenedRutherford, θ, elm, E) / LiljequistCorrection[z(elm)](E)
end

function δσδΩ(::Type{ScreenedRutherford}, θ::Float64, elm::AbstractVector{Element}, E::AbstractVector{<:Real})
    return ξ(ScreenedRutherford, elm, E) *
           (1.0 .- cos(θ) .+ ϵ(ScreenedRutherford, elm, E)^-1)^-2
end
function δσδΩ(::Type{Liljequist1989}, θ::Float64, elm::AbstractVector{Element}, E::AbstractVector{<:Real})
    return δσδΩ(ScreenedRutherford, θ, elm, E) ./ LiljequistCorrection[z(elm)](E)
end

"""
    λ(ty::Type{<:ElasticScatteringCrossSection}, θ::Float64, elm::Element, E::Float64)::Float64

The mean free path.  The mean distance between elastic scattering events. 
"""
λ(σ::T, N::T) where {T<:AbstractFloat} = (σ * N)^-1 
λ(::Type{S}, elm::Element, E::T, N::T) where {S<:ElasticScatteringCrossSection, T} = λ(σₜ(S, elm, E), N)
function λ(::Type{S}, mat::AbstractMaterial{T}, elm::Element, E::T)::T where {S<:ElasticScatteringCrossSection, T}
    return λ(S, elm, E, atoms_per_cm³(mat, elm))
end
function λ(::Type{S}, mat::VectorizedMaterial{<:Any, T}, index::Integer, E::T) where {S<:ElasticScatteringCrossSection, T}
    return λ(S, elm_nocheck(mat, index), E, atoms_per_cm³(mat, index)) 
end
function λ(::Type{S}, mat::AbstractMaterial{T}, E::T)::T where {S<:ElasticScatteringCrossSection, T}
    return λ(σₜ(S, mat, E), atoms_per_cm³(mat))
end

# Useful when σ can be precomputed
function λ(mat::VectorizedMaterial{N, T}, σ::StaticVector{N, T}) where {N, T}
    return λ(sum(index -> σ[index] * atomicfrac(mat, index), eachindex(σ)), atoms_per_cm³(mat))
end
function λ(mat::ParametricMaterial{N, T}, σ::StaticVector{N, T}, pos::AbstractVector) where {N, T}
    return evaluateat(_mat ->  λ(_mat, σ), mat, pos)
end

# For parametric material, could lead to memory allocation
function λ(::Type{S}, mat::ParametricMaterial{<:Any, T}, E::T, pos::AbstractVector{T}) where {S<:ElasticScatteringCrossSection, T}
    return λ(mat, σₜ.(S, elms_vector(mat), E), pos)
end

"""
    Base.rand(::Type{<:ElasticScatteringCrossSection}, mat::AbstractMaterial, E::Real, floattype::Type{<:AbstractFloat}=Float64)::NTuple{3,}

Returns a randomly selected elastic scattering event description.  The result is ( λ, θ, ϕ ) where
λ is a randomized mean free path for the first scattering event.  θ is a randomized scattering
angle on (0.0, π) and ϕ is a randomized azimuthal angle on [0, 2π).

The algorithm considers scattering by any element in the material and picks the shortest randomized
path.  This implementation depends on two facts: 1) We are looking for the first scattering event
so we consider all the elements and pick the one with the shortest path. 2) The process is memoryless.
"""
function Base.rand(
    ::Type{S}, mat::AbstractMaterial{T}, E::T
)::NTuple{3, T} where {S<:ElasticScatteringCrossSection, T<:AbstractFloat}
    elm′::Element = elements[119]
    λ′::T = 1.0e308
    for elm in elms(mat)
        l = -λ(S, mat, elm, E) * log(rand(T))
        if (l < λ′)
            (elm′, λ′) = (elm, l)
        end
    end
    @assert elm′ != elements[119] "Are there any elements in $mat?  Is the density ($(mat[:Density])) too low?"
    return (λ′, rand(S, elm′, E), T(2π) * rand(T))
end
function Base.rand(
    ::Type{S}, mat::VectorizedMaterial{<:Any, T}, E::T
)::NTuple{3, T} where {S<:ElasticScatteringCrossSection, T<:AbstractFloat}
    elm′::Element = elements[119]
    λ′::T = 1.0e308
    for index in eachindex(mat)
        l = -λ(S, mat, index, E) * log(rand(T))
        if (l < λ′)
            (elm′, λ′) = (elm_nocheck(mat, index), l)
        end
    end
    @assert elm′ != elements[119] "Are there any elements in $mat?  Is the density ($(mat[:Density])) too low?"
    return (λ′, rand(S, elm′, E), T(2π) * rand(T))
end

function randλelm(
    ::Type{S}, mat::AbstractMaterial{T}, E::T
)::Tuple{T, Element} where {S<:ElasticScatteringCrossSection, T<:AbstractFloat}
    elm′::Element = elements[119]
    𝜆::T = typemax(T)
    for elm in elms(mat)
        l = -λ(S, mat, elm, E) * log(rand(T))
        if (l < 𝜆)
            (elm′, 𝜆) = (elm, l)
        end
    end
    if 𝜆 ≈ typemax(T) error("Failed to pick element for $mat") end
    return 𝜆, elm′
end
function randλelm(
    ::Type{S}, mat::VectorizedMaterial{<:Any, T}, E::T
)::Tuple{T, Element} where {S<:ElasticScatteringCrossSection, T<:AbstractFloat}
    elms = elms_vector(mat)
    σ = fracσₜ(S, mat, E, elms)
    𝜆 = λ(sum(σ), atoms_per_cm³(mat)) * -log(rand(T))
    return 𝜆, pickrand(elms, σ)
end
# alternate method, guarantees no memory allocation
function randλelm_(
    ::Type{S}, mat::VectorizedMaterial{<:Any, T}, E::T
)::Tuple{T, Element} where {S<:ElasticScatteringCrossSection, T<:AbstractFloat}
    elm′::Element = elements[119]
    𝜆::T = typemax(T)
    for index in eachindex(mat)
        l = -λ(S, mat, index, E) * log(rand(T))
        if (l < 𝜆)
            (elm′, 𝜆) = (elm(mat, index), l)
        end
    end
    if 𝜆 ≈ typemax(T) error("Failed to pick element for $mat") end
    return 𝜆, elm′
end

function randelm(mat::VectorizedMaterial{N, T}, σ::StaticVector{N, T}) where {N, T}
    pickrand(elms_vector(mat), σ .* atomicfracs(mat))
end
function randelm(mat::ParametricMaterial{N, T}, σ::StaticVector{N, T}, pos::AbstractVector) where {N, T}
    evaluateat(_mat -> randelm(_mat, σ), mat, pos)
end
# alternate method, guarantees no memory allocation
function randelm_(mat::VectorizedMaterial{N, T}, σ::StaticVector{N, T}) where {N, T}
    elm′::Element = elements[119]
    𝜆::T = typemax(T)
    for index in eachindex(σ)
        l = -log(rand(T)) / (σ[index] * atomicfrac(mat, index))
        if l < 𝜆
            (𝜆, elm′) = l, elm(mat, index)
        end
    end
    if 𝜆 ≈ typemax(T) error("Failed to pick element for $mat and $σ") end
    return elm′
end

function randλelm(
    ::Type{S},
    mat::ParametricMaterial{<:Any, T},
    E::T,
    pos::AbstractVector{T},
    dir::AbstractVector{T},
    rtol::T=0.01,
    maxiters::Integer=5,
    quad::Any=nothing,
) where {S<:ElasticScatteringCrossSection, T<:AbstractFloat}
    σ = σₜ.(S, elms_vector(mat), Ref(E))
    r::T = -log(rand(T))
    𝜆::T = r *  λ(mat, σ, pos)
    for _ in 1:maxiters
        𝜆old = 𝜆
        Λ′ = quadrature(zero(T), 𝜆, quad) do l
            λ(mat, σ, pos .+ l .* dir)
        end
        𝜆 = r * Λ′ / 𝜆
        if abs((𝜆 - 𝜆old) / 𝜆) < rtol 
            break
        end
    end
    elm = randelm(mat, σ, pos .+ 𝜆 .* dir)
    return 𝜆, elm
end


"""
    λₜᵣ(ty::Type{<:ElasticScatteringCrossSection}, θ::Float64, elm::Element, E::Float64)::Float64
    λₜᵣ(ty::Type{<:ScreenedRutherfordType}, mat::Material, elm::Element, E::Float64)

The transport mean free path. The mean distance in the initial direction of propagation between
elastic scattering events.

  * N is the number of atoms per cm³
  * E is the electron kinetic energy in eV 
"""
function λₜᵣ(
    ty::Type{<:ElasticScatteringCrossSection},
    elm::Element,
    E::Real,
    N::Real,
)
    return (σₜᵣ(ty, elm, E) * N)^-1
end
function λₜᵣ(ty::Type{<:ScreenedRutherfordType}, mat::Material, elm::Element, E::Real)
    return λₜᵣ(ty, elm, E, atoms_per_cm³(mat, elm))
end

"""
    Base.rand(::Type{<:ScreenedRutherfordType}, elm::Element, E::Float64)::Float64

Draw an angle distributed according to the angular dependence of the differential screened Rutherford cross-section.
"""
function Base.rand(ty::Type{<:ScreenedRutherfordType}, elm::Element, E::T)::T where T
    Y = rand(T)
    return acos(1.0 + (Y - 1.0) / (ϵ(ty, elm, E) * Y + 0.5))
end
function Base.rand(ty::Type{Browning1994}, elm::Element, E::T)::T where T
    α, R = 7.0e-3 / (0.001 * E), rand(T)
    return acos(1.0 - 2.0 * α * R / (1.0 + α - R))
end
