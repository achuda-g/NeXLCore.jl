using GeometryBasics: Point, Rect3, Sphere, GeometryPrimitive, origin, widths, radius, Point3
import GeometryBasics: direction
using LinearAlgebra: dot, norm
using Random: rand
using QuadGK

"""
`Position` : A point in 3-D.  Ultimately, derived from StaticArray.
"""
const Position = Point{3,Float64}

"""
The MonteCarlo uses the shapes defined in GeometryBasics basics as the foundation for its 
sample construction mechanisms.  However, GeometryBasics basics does not provide all the 
necessary methods.  Three additional methods are 

    isinside(r::Shape, pos::Position)

Is `pos` strictly inside `r`?

    intersection(r::Shape, pos0::Particle, pos1::Particle)::Float64

Return a number `f` which represent the fraction of the distance from `pos0` to `pos1` that
first intersects the `Shape` `r`.  The intersection point will equal `pos0 .+ f*(pos1 .- pos0)`.
If `f` is between 0.0 and 1.0 then the intersection is on the interval between `pos0` and `pos1`.
If the ray from `pos0` towards `pos2` does not intersect `r` then this function returns Inf64.
"""
const RectangularShape = Rect3{Float64}

isinside(rr::Rect3{<:Real}, pos::AbstractArray{<:Real}) =
    all(pos .≥ minimum(rr)) && all(pos .≤ maximum(rr)) # write for voxels i, i + 1

function intersection(
    rr::Rect3{T},
    pos1::AbstractArray{T},
    pos2::AbstractArray{T},
) where {T<:Real}
    _isinside = isinside(rr, pos1)
    t::T = typemax(T)
    corner1, corner2 = minimum(rr), maximum(rr)
    _between(t, j) = corner1[j] ≤ pos1[j] + t * (pos2[j] - pos1[j]) ≤ corner2[j]
    for i in eachindex(pos1)
        j, k = i % 3 + 1, (i + 1) % 3 + 1
        if pos2[i] != pos1[i]
            v = pos2[i] - pos1[i]
            t1 = (corner1[i] - pos1[i]) / v
            if (t1 > zero(T)) && (t1 < t) && (_isinside || (_between(t1, j) && _between(t1, k)))
                t = t1
            end
            t2 = (corner2[i] - pos1[i]) / v
            if (t2 > zero(T)) && (t2 < t) && (_isinside || (_between(t2, j) && _between(t2, k)))
                t = t2
            end
        end
    end
    return t
end

const SphericalShape = Sphere{Float64}

isinside(sr::Sphere{<:Real}, pos::AbstractArray{<:Real}) =
    norm(pos .- origin(sr)) < radius(sr)

function intersection(
    sr::Sphere{<:Real},
    pos0::AbstractArray{T},
    pos1::AbstractArray{T},
) where T
    d, m = pos1 .- pos0, pos0 .- origin(sr)
    ma2, b = -2.0 * dot(d, d), 2.0 * dot(m, d)
    f = b^2 + ma2 * 2.0 * (dot(m, m) - radius(sr)^2)
    if f >= 0.0
        up, un = (b + sqrt(f)) / ma2, (b - sqrt(f)) / ma2
        return min(up < 0.0 ? typemax(T) : up, un < 0.0 ? typemax(T) : un)
    end
    return typemax(T)
end

"""
    random_point_inside(shape)

Generate a randomized point that is guaranteed to be in the interior of the shape.
"""
function random_point_inside(shape)
    res = Position(origin(shape) .+ rand(Position) .* widths(shape))
    while !isinside(shape, res)
        res = Position(origin(shape) .+ rand(Position) .* widths(shape))
    end
    return res
end

# Glen - moved this to another script ? Where
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
abstract type Particle{T} end
Base.eltype(::Particle{T}) where T = T

struct Electron{T} <: Particle{T}
    previous::Point3{T}
    current::Point3{T}
    energy::T # eV

    """
        Electron(prev::Position, curr::Position, energy::Float64)
        Electron(el::Electron, 𝜆::Float64, 𝜃::Float64, 𝜑::Float64, ΔE::Float64)::Electron
    
    Create a new `Electron` from this one in which the new `Electron` is a distance `𝜆` from the
    first along a trajectory that is `𝜃` and `𝜑` off the current trajectory.
    """
    Electron(prev::AbstractArray{T}, curr::AbstractArray{T}, energy::T) where T =
        new{T}(prev, curr, energy)

    function Electron(prev::AbstractArray{T1}, curr::AbstractArray{T2}, energy::T3) where {T1<:Real, T2<:Real, T3<:Real}
        T = promote_type(T1, T2, T3)
        @debug "type promoted: ($T1, $T2, $T3) -> $T"
        new{T}(T.(prev), T.(curr), T.(energy))
    end

    function Electron{T}(el::Electron{T}, 𝜆::T, 𝜃::T, 𝜑::T, ΔE::T) where T
        (u, v, w) = LinearAlgebra.normalize(position(el) .- previous(el))
        cθ, sθ, cϕ, sϕ = cos(𝜃), sin(𝜃), cos(𝜑), sin(𝜑)
        sc =
            one(T) - abs(w) > sqrt(eps(T)) ? #
            Point3{T}( #
                u * cθ + sθ * (u * w * cϕ - v * sϕ) / sqrt(1.0 - w^2), #
                v * cθ + sθ * (v * w * cϕ + u * sϕ) / sqrt(1.0 - w^2), #
                w * cθ - sqrt(1.0 - w^2) * sθ * cϕ, # 
            ) :
            Point3{T}( #
                sign(w) * sθ * cϕ, #
                sign(w) * sθ * sϕ, #
                sign(w) * cθ,
            )
        return new(position(el), position(el) .+ 𝜆 * sc, el.energy + ΔE)
    end
end

Base.show(io::IO, el::Electron) = print(io, "Electron[$(position(el)), $(energy(el)) eV]")
Base.position(el::Particle) = el.current
current(el::Particle) = position(el)
previous(el::Particle) = el.previous
energy(el::Particle) = el.energy
direction(el::Particle) = normalize(position(el) - previous(particle))


"""
    transport(pc::Electron, mat::ParametricMaterial, ecx=Liljequist1989, bethe=JoyLuo)::NTuple{4, Float64}

The default function defining elastic scattering and energy loss for an Electron.

Returns ( `λ`, `θ`, `ϕ`, `ΔE`) where `λ` is the mean path length, `θ` is the elastic scatter angle, `ϕ` is the azimuthal elastic scatter
angle and `ΔE` is the energy loss for transport over the distance `λ`. 'Num_iterations' is the number of desired iterations for the integrations.
"""
function transport(
    pc::Electron{T},
    mat::AbstractMaterial{T}; #Function - elements fixed with mass fractions changing
    ecx::Type{<:ElasticScatteringCrossSection}=Liljequist1989,
    bethe::Type{<:BetheEnergyLoss}=JoyLuo,
)::NTuple{4, T} where T
    (𝜆′, θ′, ϕ′) = rand(ecx, mat, pc.energy) 
    return (𝜆′, θ′, ϕ′, 𝜆′ * dEds(bethe, pc.energy, mat))
end

"""
    pathlength(el::Particle)

Length of the line segment represented by `el`.
"""
pathlength(el::Particle) = norm(position(el) .- previous(el))

intersection(r, p::Particle) = intersection(r, previous(p), position(p))


"""
    Region

A `Region` combines a geometric primative and a `Material` (with `:Density` property) and may fully contain zero or more child `Region`s.
"""

abstract type AbstractRegion end
material(reg::AbstractRegion) = reg.material
shape(reg::AbstractRegion) = reg.shape
parent(reg::AbstractRegion) = reg.parent
children(reg::AbstractRegion) = reg.children
name(reg::AbstractRegion) = reg.name
haschildren(reg::AbstractRegion) = length(children(reg)) > 0

struct Region{S, M} <: AbstractRegion
    shape::S
    material::M
    parent::Union{Nothing,AbstractRegion}
    children::Set{AbstractRegion}
    name::String

    function Region(
        sh::GeometryPrimitive{3},
        mat::AbstractMaterial,
        parent::Union{Nothing,AbstractRegion},
        name::Union{Nothing,String} = nothing,
        ntests = 1000,
    )
        #@assert mat[:Density] > 0.0 # Glen - removed
        name = something(
            name,
            isnothing(parent) ? "Root" : "$(parent.name)[$(length(parent.children)+1)]",
        )
        res = new{typeof(sh), typeof(mat)}(sh, mat, parent, Set{AbstractRegion}(), name)
        if !isnothing(parent)
            @assert all(
                _ -> isinside(parent.shape, random_point_inside(sh)),
                Base.OneTo(ntests),
            ) "The child $sh is not fully contained within the parent $(parent.shape)."
            @assert all(
                ch -> all(
                    _ -> !isinside(ch.shape, random_point_inside(sh)),
                    Base.OneTo(ntests),
                ),
                parent.children,
            ) "The child $sh overlaps a child of the parent shape."
            push!(parent.children, res)
        end
        return res
    end
end

Base.show(io::IO, reg::Region) = print(
    io,
    "Region[$(reg.name), $(reg.shape), $(reg.material), $(length(reg.children)) children]",
)

"""
    childmost_region(reg::Region, pos::Position)::Region

Find the inner-most `Region` within `reg` containing the point `pos`.
"""
function childmost_region_(reg::AbstractRegion, pos::AbstractArray{<:Real})
    res = findfirst(ch -> isinside(shape(ch), pos), children(reg))
    return !isnothing(res) ? childmost_region(children(reg)[res], pos) : reg
end

function childmost_region(reg::AbstractRegion, pos::AbstractArray{<:Real})
    for ch in children(reg)
        if isinside(shape(ch), pos)
            return childmost_region(ch, pos)
        end
    end
    return reg
end

isinside(reg::AbstractRegion, pos::AbstractArray{<:Real}) = isinside(shape(reg), pos)
isinside(reg::Any, p::Particle) = isinside(reg, position(p))

"""
    take_step(p::T, reg::Region, 𝜆::Float64, 𝜃::Float64, 𝜑::Float64)::Tuple{T, Region, Bool} where { T<: Particle}

Returns a `Tuple` containing a new `Particle` and the child-most `Region` in which the new `Particle` is found based
on a scatter event consisting a translation of up to `𝜆` mean-free path along a new direction given relative
to the current direction of `p` via the scatter angles `𝜃` and `𝜑`.

Returns the updated `Particle` reflecting the last trajectory step and the Region for the next step.
"""
function take_step(
    p::P,
    reg::AbstractRegion,
    𝜆::T,
    𝜃::T,
    𝜑::T,
    ΔE::T,
    ϵ::T,
)::Tuple{P, Bool} where {P<:Particle, T}
    newP::P = P(p, 𝜆, 𝜃, 𝜑, ΔE)
    pos1, pos2 = previous(newP), position(newP)
    t = intersection(shape(reg), pos1, pos2)
    for ch in children(reg)
        tch = intersection(shape(ch), pos1, pos2)
        if tch < t 
            t = tch
        end
    end
    scatter = t > one(T)
    if !scatter # Enter new Region
        𝜆new = t*𝜆 + ϵ
        newP = P(p, 𝜆new, 𝜃, 𝜑, 𝜆new / 𝜆 * ΔE)
    end
    return (newP, scatter)
end


"""
trajectory(eval::Function, p::T, reg::Region, scf::Function=transport; minE::Float64=50.0) where {T <: Particle}
trajectory(eval::Function, p::T, reg::Region, scf::Function, terminate::Function) where { T <: Particle }

Run a single particle trajectory from `p` to `minE` or until the particle exits `reg`.

  * `eval(part::T, region::Region)` a function evaluated at each scattering point
  * `p` defines the initial position, direction and energy of the particle (often created with `gun(T, ...)`)
  * `reg` The outer-most region for the trajectory (usually created with `chamber()`)
  * `scf` A function from (<:Particle, Material) -> ( λ, θ, ϕ, ΔE ) that implements the transport dynamics
  * `minE` Stopping criterion
  * `terminate` a function taking `T` and `Region` that returns false except on the last step (like `terminate = (pc,r)->pc.energy < 50.0`)
"""
function trajectory(
    eval::Function,
    p::P,
    reg::AbstractRegion,
    scf::Function,
    terminate::Function,
) where {P<:Particle}
    pc::P = p
    nextr::AbstractRegion = childmost_region(reg, position(p))
    T = eltype(p)
    θ::T = zero(T)
    ϕ::T = zero(T)
    while (!terminate(pc, reg)) && isinside(shape(reg), position(pc))
        prevr = nextr
        (λ, θₙ, ϕₙ, ΔZ) = scf(pc, material(prevr))
        (pc, scatter) = take_step(pc, prevr, λ, θ, ϕ, ΔZ, T(1e-14))
        if scatter
            (θ, ϕ) = (θₙ, ϕₙ) # scatter true? New angles. False? Old angles.
        else
            (θ, ϕ) = (zero(T), zero(T))
            nextr = childmost_region(isnothing(parent(reg)) ? reg : parent(reg), position(pc))
        end
        eval(pc, prevr)
    end
end
function trajectory(
    eval::Function,
    p::T,
    reg::AbstractRegion,
    scf::Function = (t::T, mat::AbstractMaterial) -> transport(t, mat); 
    minE::Real = 50.0,
) where {T<:Particle}
    term(pc::T, _::AbstractRegion) = pc.energy < minE
    trajectory(eval, p, reg, scf, term)
end
