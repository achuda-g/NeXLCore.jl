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
    dEdρs(::Type{<:BetheEnergyLoss}, e::AbstractFloat, elm::Element, ::Type{<:NeXLMeanIonizationPotential}=Berger1982)
    dEdρs(::Type{<:BetheEnergyLoss}, e::AbstractFloat, mat::AbstractMaterial, ::Type{<:NeXLMeanIonizationPotential}=Berger1982)

Calculate the loss per unit mass path length for an electron in the specified element. The results in energy
loss in eV/gcm<sup>2</sup>.  Implemented by `Type{Bethe}` and `Type{JoyLuo}`.
"""
function dEdρs(
    ::Type{Bethe},
    e::T,
    elm::Element,
    ::Type{MIP} = Berger1982,
) where {T<:AbstractFloat, MIP<:NeXLMeanIonizationPotential}
    j = T(J(MIP, elm))
    (T(-785.0e8) * z(T, elm)) / (a(T, elm) * e) * log(T(1.166) * e / j)
end
function dEdρs(
    ::Type{JoyLuo},
    e::T,
    elm::Element,
    ::Type{MIP} = Berger1982,
) where {T<:AbstractFloat, MIP<:NeXLMeanIonizationPotential}
    # Zero allocation
    k = T(0.731) + T(0.0688) * log10(z(T, elm))
    j = T(J(MIP, z(elm)))
    jp = j / (one(T) + k * j / e)
    return ((T(-785.0e8) * z(T, elm)) / (a(T, elm) * e)) * log(T(1.166) * e / jp)
end
function dEdρs(
    ::Type{BEL},
    e::Real,
    mat::AbstractMaterial{T},
    ::Type{MIP} = Berger1982,
) where {T<:AbstractFloat, BEL<:BetheEnergyLoss, MIP<:NeXLMeanIonizationPotential}
    res = zero(T)
    for el in elms(mat)
        res += dEdρs(BEL, e, el, MIP) * mat[el]
    end
    return res
end

"""
    dEds(::Type{<:BetheEnergyLoss}, e::AbstractFloat, elm::Element, ρ::Real, ::Type{<:NeXLMeanIonizationPotential}=Berger1982)
    dEds(::Type{<:BetheEnergyLoss}, e::AbstractFloat, mat::AbstractMaterial, ::Type{<:NeXLMeanIonizationPotential}=Berger1982)
    dEds(::Type{<:BetheEnergyLoss}, e::AbstractFloat, mat::ParametricMaterial, pos::AbstractVector, ::Type{<:NeXLMeanIonizationPotential}=Berger1982)

Calculate the loss per unit path length for an electron in the specified element and density.  The results in energy
loss in eV/cm.  Implemented by `Type{Bethe}` and `Type{JoyLuo}`.
"""
function dEds(
    ::Type{BEL},
    e::AbstractFloat,
    elm::Element,
    ρ::Real,
    ::Type{MIP} = Berger1982,
) where {BEL<:BetheEnergyLoss, MIP<:NeXLMeanIonizationPotential}
    return dEdρs(BEL, e, elm, MIP) * ρ
end
function dEds(
    ::Type{BEL},
    e::AbstractFloat,
    mat::AbstractMaterial,
    ::Type{MIP} = Berger1982,
) where {BEL<:BetheEnergyLoss, MIP<:NeXLMeanIonizationPotential}
    return dEdρs(BEL, e, mat, MIP) * density(mat)
end

"""
    range(::Type{BetheEnergyLoss}, mat::AbstractMaterial, e0::AbstractFloat, inclDensity = true; 
        emin::AbstractFloat = 50.0, mip::Type{<:NeXLMeanIonizationPotential} = Berger1982,
    )

Calculates the electron range using numeric quadrature of a BetheEnergyLoss algorithm.
"""
Base.range(
    ty::Type{<:BetheEnergyLoss},
    mat::AbstractMaterial,
    e0::AbstractFloat,
    inclDensity = true;
    emin::AbstractFloat = 50.0,
    mip::Type{<:NeXLMeanIonizationPotential} = Berger1982,
) =
    quadgk(e -> 1.0 / dEds(ty, e, mat, mip), e0, emin, rtol = 1.0e-6)[1] *
    (inclDensity ? 1.0 : density(mat))

struct Kanaya1972 end

"""
    range(::Type{Kanaya1972}, mat::AbstractMaterial, e0::AbstractFloat, inclDensity = true)

Calculates the Kanaya-Okayama electron range.
Kanaya K, Okayama S (1972) Penetration and energy-loss theory of electrons in solid targets. J Appl Phys 5:43
"""
function Base.range(::Type{Kanaya1972}, mat::AbstractMaterial, e0::AbstractFloat, inclDensity = true)
    ko(elm, e0) = 0.0276 * a(elm) * (0.001 * e0)^1.67 / z(elm)^0.89
    return (1.0e-4 / mapreduce(elm -> mat[elm] / ko(elm, e0), +, elms(mat))) /
           (inclDensity ? density(mat) : 1.0)
end
function Base.range(::Type{Kanaya1972}, mat::VectorizedMaterial, e0::AbstractFloat, inclDensity = true)
    ko(elm, e0) = 0.0276 * a(elm) * (0.001 * e0)^1.67 / z(elm)^0.89
    return (1.0e-4 / sum(i -> massfrac(mat, i) / ko(elm_nocheck(mat, i), e0), eachindex(mat))) /
           (inclDensity ? density(mat) : 1.0)
end
