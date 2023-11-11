"""
   mₑ : Electron rest mass (in eV)
"""
const mₑ = ustrip((ElectronMass * SpeedOfLightInVacuum^2) |> u"eV")
mₑ_(::Type{T}) where T = ustrip(T, u"eV", ElectronMass * SpeedOfLightInVacuum^2)

"""
Relativistic γ for v in cm/s
"""
γₑ(v::Quantity{T}) where T<:AbstractFloat = one(T) / sqrt(one(T) - (v/T(SpeedOfLightInVacuum))^2) |> NoUnits
γₑ(v::T) where T<:AbstractFloat = one(T) / sqrt(one(T) - (v/ustrip(T, u"cm/s", SpeedOfLightInVacuum))^2)

"""
Electron kinetic energy in eV for v in cm/s.
"""
Ekₑ(v::Quantity{T}) where T<:AbstractFloat = (γₑ(v) - one(T)) * T(ElectronMass) * T(SpeedOfLightInVacuum)^2 |> u"eV"
Ekₑ(v::T) where T<:AbstractFloat = ustrip(u"eV", (γₑ(v) - one(T)) * T(ElectronMass) * T(SpeedOfLightInVacuum)^2)

"""
Electron velocity in cm/s for the specified kinetic energy in eV.
"""
function vₑ(Ek::Quantity{T}) where T<:AbstractFloat
    γ = one(T) + Ek/(T(ElectronMass)*T(SpeedOfLightInVacuum)^2)
    sqrt(one(T) - (one(T) / γ)^2) * T(SpeedOfLightInVacuum) |> u"cm/s"
end
function vₑ(Ek::T) where T<:AbstractFloat
    γ = one(T) + Ek/mₑ_(T)
    sqrt(one(T) - (one(T) / γ)^2) * ustrip(T, u"cm/s", SpeedOfLightInVacuum)
end


"""
    λₑ(E::Union{AbstractFloat, Quantity{AbstractFloat}})

Wavelength of an electron (in cm if input is a number).
    * `E` is assumed to be in eV if units not present
"""
function λₑ(E::Quantity{T}) where T<:AbstractFloat
    v = vₑ(E)
    T(PlanckConstant) * sqrt(one(T) - (v/T(SpeedOfLightInVacuum))^2) / (T(ElectronMass)*v)
end
λₑ(E::AbstractFloat) = ustrip(u"cm", λₑ(E*u"eV"))
"""
    kₑ(E::Union{AbstractFloat, Quantity{AbstractFloat}})

Electron wavenumber (inverse wavelength) (in rad⋅cm⁻¹ if input is a number).
    * E is assumed to be in eV if units not present
"""
kₑ(E::Quantity{T}) where T<:AbstractFloat = T(2π) / λₑ(E)
kₑ(E::AbstractFloat) = ustrip(u"cm^-1", kₑ(E*u"eV"))


"""
Electrons per second per nA of current.
"""
electrons_per_second(ic::Quantity{T}) where T = ic/T(ElementaryCharge) |> u"s^-1"
electrons_per_second(ic::AbstractFloat) = ustrip(electrons_per_second(ic*u"nA"))
