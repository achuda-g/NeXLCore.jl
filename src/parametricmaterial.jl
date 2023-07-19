using StaticArrays

function volume_conserving_density(elms::SVector{N, Element}) where N
    ρ_pure = SVector{N, Float64}(density.(elms))
    function _density(c)
        return 1. / sum(c ./ ρ_pure)
    end
    return _density
end

"""
Holds basic data about a material including name, composition in mass fraction and optional propreties.

By default, Material assumes nominal terrestrial atomic weights.  However, it is possible to assign custom
atomic weights on a per element-basis for non-terrestrial materials.

The mass fraction and atomic weight are immutable but the `Properties` can be modified.

    Material(
        name::AbstractString,
        massfrac::AbstractDict{Element,U},
        atomicweights::AbstractDict{Element,V} = Dict{Element,Float64}(),
        properties::AbstractDict{Symbol,Any} = Dict{Symbol,Any}(),
    ) where { U <: AbstractFloat, V <: AbstractFloat }


**Properties**

    :Density # Density in g/cm³
    :Description # Human friendly
    :Pedigree #  Quality indicator for compositional data ("SRM-XXX", "CRM-XXX", "NIST K-Glass", "Stoichiometry", "Wet-chemistry by ???", "WDS by ???", "???")
    :Conductivity = :Insulator | :Semiconductor | :Conductor
    :OtherUserProperties # Other properties can be defined as needed
"""
struct ParametricMaterial{T<:AbstractFloat, N}
    name::String
    elms::SVector{N, Element}
    massfrac::Vector{T}
    massfracfunc!::Function
    a::SVector{N, T} # Optional: custom atomic weights for the keys in this Material
    ρ::Function
    properties::Dict{Symbol,Any} # :PureDensities, :Description, :Pedigree, :Conductivity, ... + user defined

    """
        Material(
            name::AbstractString,
            massfrac::AbstractDict{Element,U},
            atomicweights::AbstractDict{Element,V} = Dict{Element,Float64}(),
            properties::AbstractDict{Symbol,Any} = Dict{Symbol,Any}(),
        ) where { U <: AbstractFloat, V <: AbstractFloat }
    """
    function ParametricMaterial{T}(
        name::AbstractString,
        elms::AbstractVector{Element},
        massfracfunc!::Function,
        atomicweights::Union{AbstractArray{T}, Nothing}=nothing,
        ρ::Union{Nothing, Function}=nothing,
        properties::AbstractDict{Symbol,Any}=Dict{Symbol,Any}(),
    ) where {T<:AbstractFloat}
        N = length(elms)
        elms = SVector{N, Element}(elms) # needed?
        massfrac = ones(N) ./ N
        if isnothing(ρ)
            ρ = volume_conserving_density(elms)
        end
        if isnothing(atomicweights)
            atomicweights = a.(elms)
        end
        atomicweights = SVector{N, T}(atomicweights)
        new{T,N}(name, elms, massfrac, massfracfunc!, atomicweights, ρ, properties)
    end
    
    function ParametricMaterial(args...; kwargs...)
        ParametricMaterial{Float64}(args..., kwargs...)
    end
end

function massfractions(mat::ParametricMaterial, x::AbstractArray)
    mat.massfracfunc!(mat.massfrac, x)
    return mat.massfrac
end

function density(mat::ParametricMaterial, x::AbstractArray)
    c = massfractions(mat, x)
    return mat.ρ(c)
end
function density(mat::ParametricMaterial)
    return mat.ρ(mat.massfrac)
end

function atoms_per_cm³(mat::ParametricMaterial)
    return atoms_per_g.(mat.elms) .* mat.massfrac .* density(mat)
end

"""
ParametricMaterial(
    "FeNi",
    [n"Fe", n"Ni"],
    massfracfunc!,
)

function matfracfunc!(massfrac::Vector, x::AbstractArray)
    massfrac[0] = sin(x[1])
    massfrac[1] = 1 - massfrac[0]
end
"""


