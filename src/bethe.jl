using QuadGK
using GeometryBasics: Point, Rect3, Sphere, GeometryPrimitive, origin, widths, radius

"""
`Position` : A point in 3-D.  Ultimately, derived from StaticArray. Glen - redefinition here as scattering is first included.
"""
const Position = Point{3,Float64} # Glen - moved here from mc.jl

"""
Particle represents a type that may be simulated using a transport Monte Carlo.  It must provide
these methods:

    position(el::Particle)::Position
    previous(el::Particle)::Position
    energy(el::Particle)::Float64

The position of the current and previous elastic scatter locations which are stored in that Particle type.

    T(prev::Position, curr::Position, energy::Energy) where {T <: Particle }
    T(el::T, 𝜆::Float64, 𝜃::Float64, 𝜑::Float64, ΔE::Float64) where {T <: Particle }

Two constructors: One to create a defined Particle and the other to create a new Particle based off
another which is translated by `λ` at a scattering angle (`θ`, `ϕ`) which energy change of `ΔE`

    transport(pc::T, mat::Material)::NTuple{4, Float64} where {T <: Particle }

A function that generates the values of ( `λ`, `θ`, `ϕ`, `ΔE`) for the specified `Particle` in the specified `Material`.
"""
abstract type Particle end # Glen - moved here from mc.jl

struct Electron <: Particle
    previous::Position
    current::Position
    energy::Float64 # eV

    """
        Electron(prev::Position, curr::Position, energy::Float64)
        Electron(el::Electron, 𝜆::Float64, 𝜃::Float64, 𝜑::Float64, ΔE::Float64)::Electron
    
    Create a new `Electron` from this one in which the new `Electron` is a distance `𝜆` from the
    first along a trajectory that is `𝜃` and `𝜑` off the current trajectory.
    """
    Electron(prev::AbstractArray{Float64}, curr::AbstractArray{Float64}, energy::Float64) =
        new(prev, curr, energy)

    function Electron(el::Electron, 𝜆::Float64, 𝜃::Float64, 𝜑::Float64, ΔE::Float64)
        (u, v, w) = LinearAlgebra.normalize(position(el) .- previous(el))
        sc =
            1.0 - abs(w) > 1.0e-8 ? #
            Position( #
                u * cos(𝜃) + sin(𝜃) * (u * w * cos(𝜑) - v * sin(𝜑)) / sqrt(1.0 - w^2), #
                v * cos(𝜃) + sin(𝜃) * (v * w * cos(𝜑) + u * sin(𝜑)) / sqrt(1.0 - w^2), #
                w * cos(𝜃) - sqrt(1.0 - w^2) * sin(𝜃) * cos(𝜑), # 
            ) :
            Position( #
                sign(w) * sin(𝜃) * cos(𝜑), #
                sign(w) * sin(𝜃) * sin(𝜑), #
                sign(w) * cos(𝜃),
            )
        return new(position(el), position(el) .+ 𝜆 * sc, el.energy + ΔE)
    end
end

Base.show(io::IO, el::Electron) = print(io, "Electron[$(position(el)), $(energy(el)) eV]")
Base.position(el::Particle) = el.current
previous(el::Particle) = el.previous
energy(el::Particle) = el.energy
# Energy loss expressions
"""
An abstract type to describe kinetic energy loss by electrons. 
"""
abstract type BetheEnergyLoss end

"""
The Bethe algorithm of kinetic energy loss by electrons.
"""
struct Bethe <: BetheEnergyLoss end

"""
The Joy-Luo algorithm of Bethe kinetic energy loss by electrons.
SCANNING Vol. 11, 176-180 (1989) 
"""
struct JoyLuo <: BetheEnergyLoss end

"""
    dEds(::Type{<:BetheEnergyLoss}, e::Float64, elm::Element, ρ::Float64; mip::Type{<:NeXLMeanIonizationPotential}=Berger1982)
    dEds(::Type{<:BetheEnergyLoss}, e::Float64, mat::Material, inclDensity=true; mip::Type{<:NeXLMeanIonizationPotential}=Berger1982)

Calculate the loss per unit path length for an electron in the specified element and density.  The results in energy
loss in eV/Å.  Implemented by `Type{Bethe}` and `Type{JoyLuo}`.
"""
dEds(
    ::Type{Bethe},
    e::Float64,
    elm::Element,
    ρ::Float64,
    mip::Type{<:NeXLMeanIonizationPotential} = Berger1982,
) = (-785.0e8 * ρ * z(elm)) / (a(elm) * e) * log(1.166e / J(mip, elm))

function dEds(
    ::Type{JoyLuo},
    e::Float64,
    elm::Element,
    ρ::Float64,
    mip::Type{<:NeXLMeanIonizationPotential} = Berger1982,
)
    # Zero allocation
    k = 0.731 + 0.0688 * log(10.0, z(elm))
    j = J(mip, z(elm))
    jp = j / (1.0 + k * j / e)
    return ((-785.0e8 * ρ * z(elm)) / (a(elm) * e)) * log(1.166 * e / jp)
end
function dEds(
    ty::Type{<:BetheEnergyLoss},
    e::Float64,
    mat::Material,
    mip::Type{<:NeXLMeanIonizationPotential} = Berger1982,
)
    ρ = density(mat)
    return sum(keys(mat)) do el
        dEds(ty, e, el, ρ, mip) * mat[el]
    end
end
function dEds(
    ty::Type{<:BetheEnergyLoss},
    e::Float64,
    pos::AbstractVector,
    mat::ParametricMaterial,
    mip::Type{<:NeXLMeanIonizationPotential} = Berger1982,
)
    ρ = density(mat, pos)

    return sum(enumerate(mat.elms)) do (i, el)
        dEds(ty, e, el, ρ, mip) * mat.massfrac[i]
    end
end

function dEds(
    ty::Type{<:BetheEnergyLoss},
    e::Float64,
    mat::ParametricMaterial,
    mfp::Float64,
    θ′::Float64,
    ϕ′::Float64,
    pc::Electron,
    mip::Type{<:NeXLMeanIonizationPotential} = Berger1982,
)
    pos = position(Electron(pc, mfp, θ′, ϕ′, 0.0)) #ToDo: Optimise this
    c = massfractions(mat, pos)
    ρ = density(mat, pos)

    return sum(dEds(ty, e, elm, ρ, mip) * mat.massfrac[i] for (i, elm) in enumerate(mat.elms))
end

"""
    range(::Type{BetheEnergyLoss}, mat::Material, e0::Float64, inclDensity = true)

Calculates the electron range using numeric quadrature of a BetheEnergyLoss algorithm.
"""
Base.range(
    ty::Type{<:BetheEnergyLoss},
    mat::Material,
    e0::Float64,
    inclDensity = true;
    emin = 50.0,
    mip::Type{<:NeXLMeanIonizationPotential} = Berger1982,
) =
    quadgk(e -> 1.0 / dEds(ty, e, mat, mip), e0, emin, rtol = 1.0e-6)[1] *
    (inclDensity ? 1.0 : density(mat))

struct Kanaya1972 end

"""
    range(::Type{Kanaya1972}, mat::Material, e0::Float64, inclDensity = true)

Calculates the Kanaya-Okayama electron range.
Kanaya K, Okayama S (1972) Penetration and energy-loss theory of electrons in solid targets. J Appl Phys 5:43
"""
function Base.range(::Type{Kanaya1972}, mat::Material, e0::Float64, inclDensity = true)
    ko(elm, e0) = 0.0276 * a(elm) * (0.001 * e0)^1.67 / z(elm)^0.89
    return (1.0e-4 / mapreduce(elm -> mat[elm] / ko(elm, e0), +, keys(mat))) /
           (inclDensity ? density(mat) : 1.0)
end
