using GeometryBasics: Point, Rect3, Sphere, GeometryPrimitive, origin, widths, radius
using LinearAlgebra: dot, norm
using Random: rand
using QuadGK

#=
"""
`Position` : A point in 3-D.  Ultimately, derived from StaticArray.
"""
const Position = Point{3,Float64}
=#
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

isinside(rr::RectangularShape, pos::AbstractArray{<:Real}) =
    all(pos .≥ minimum(rr)) && all(pos .≤ maximum(rr)) # write for voxels i, i + 1

function intersection(
    rr::RectangularShape,
    pos1::AbstractArray{<:Real},
    pos2::AbstractArray{<:Real},
)
    _between(a, b, c) = (a > b) && (a < c)
    t = Inf64
    corner1, corner2 = minimum(rr), maximum(rr)
    for i in eachindex(pos1)
        j, k = i % 3 + 1, (i + 1) % 3 + 1
        if pos2[i] != pos1[i]
            u = (corner1[i] - pos1[i]) / (pos2[i] - pos1[i])
            if (u > 0.0) &&
               (u <= t) && #
               _between(pos1[j] + u * (pos2[j] - pos1[j]), corner1[j], corner2[j]) && # 
               _between(pos1[k] + u * (pos2[k] - pos1[k]), corner1[k], corner2[k])
                t = u
            end
            u = (corner2[i] - pos1[i]) / (pos2[i] - pos1[i])
            if (u > 0.0) &&
               (u <= t) && #
               _between(pos1[j] + u * (pos2[j] - pos1[j]), corner1[j], corner2[j]) && # 
               _between(pos1[k] + u * (pos2[k] - pos1[k]), corner1[k], corner2[k])
                t = u
            end
        end
    end
    return t
end

const SphericalShape = Sphere{Float64}

isinside(sr::SphericalShape, pos::AbstractArray{<:Real}) =
    norm(pos .- origin(sr)) < radius(sr)

function intersection(
    sr::SphericalShape,
    pos0::AbstractArray{<:Real},
    pos1::AbstractArray{<:Real},
)
    d, m = pos1 .- pos0, pos0 .- origin(sr)
    ma2, b = -2.0 * dot(d, d), 2.0 * dot(m, d)
    f = b^2 + ma2 * 2.0 * (dot(m, m) - radius(sr)^2)
    if f >= 0.0
        up, un = (b + sqrt(f)) / ma2, (b - sqrt(f)) / ma2
        return min(up < 0.0 ? Inf64 : up, un < 0.0 ? Inf64 : un)
    end
    return Inf64
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
abstract type Particle end

struct Electron <: Particle
    previous::Position
    current::Position
    energy::Real # eV

    """
        Electron(prev::Position, curr::Position, energy::Float64)
        Electron(el::Electron, 𝜆::Float64, 𝜃::Float64, 𝜑::Float64, ΔE::Float64)::Electron
    
    Create a new `Electron` from this one in which the new `Electron` is a distance `𝜆` from the
    first along a trajectory that is `𝜃` and `𝜑` off the current trajectory.
    """
    Electron(prev::AbstractArray{<:Real}, curr::AbstractArray{<:Real}, energy::Real) =
        new(prev, curr, energy)

    function Electron(el::Electron, 𝜆::Real, 𝜃::Real, 𝜑::Real, ΔE::Real)
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
current(el::Particle) = position(el)
previous(el::Particle) = el.previous
energy(el::Particle) = el.energy
direction(el::Particle) = direction(position(el), previous(el))


"""
    transport(pc::Electron, mat::ParametricMaterial, ecx=Liljequist1989, bethe=JoyLuo)::NTuple{4, Float64}

The default function defining elastic scattering and energy loss for an Electron.

Returns ( `λ`, `θ`, `ϕ`, `ΔE`) where `λ` is the mean path length, `θ` is the elastic scatter angle, `ϕ` is the azimuthal elastic scatter
angle and `ΔE` is the energy loss for transport over the distance `λ`. 'Num_iterations' is the number of desired iterations for the integrations.
"""
function transport(
    pc::Electron,
    mat::AbstractMaterial; #Function - elements fixed with mass fractions changing
    ecx::Type{<:ElasticScatteringCrossSection}=Liljequist1989,
    bethe::Type{<:BetheEnergyLoss}=JoyLuo,
)
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

abstract type AbstractRegion{M} end
material(reg::AbstractRegion) = reg.material
shape(reg::AbstractRegion) = reg.shape
parent(reg::AbstractRegion) = reg.parent
children(reg::AbstractRegion) = reg.children
name(reg::AbstractRegion) = reg.name

struct Region{M} <: AbstractRegion{M}
    shape::GeometryPrimitive{3, <:AbstractFloat}
    material::M
    parent::Union{Nothing,AbstractRegion}
    children::Vector{AbstractRegion}
    name::String

    function Region(
        sh::T,
        mat::M,
        parent::Union{Nothing,AbstractRegion},
        name::Union{Nothing,String} = nothing,
        ntests = 1000,
    ) where {T, M}
        #@assert mat[:Density] > 0.0 # Glen - removed
        name = something(
            name,
            isnothing(parent) ? "Root" : "$(parent.name)[$(length(parent.children)+1)]",
        )
        res = new{M}(sh, mat, parent, AbstractRegion[], name)
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
        else
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
function childmost_region(reg::AbstractRegion, pos::AbstractArray{<:Real})::AbstractRegion
    res = findfirst(ch -> isinside(shape(ch), pos), children(reg))
    return !isnothing(res) ? childmost_region(children(reg)[res], pos) : reg
end

function intersection(reg::AbstractRegion, pos1::AbstractArray{<:Real}, pos2::AbstractArray{<:Real})
    t1 = intersection(shape(reg), pos1, pos2)
    t2 = minimum(ch -> intersection(ch, pos1, pos2), children(reg))
    return min(t1, t2)
end

isinside(reg::AbstractRegion, pos::AbstractArray{<:Real}) = isinside(shape(reg), pos)

"""
    take_step(p::T, reg::Region, 𝜆::Float64, 𝜃::Float64, 𝜑::Float64)::Tuple{T, Region, Bool} where { T<: Particle}

Returns a `Tuple` containing a new `Particle` and the child-most `Region` in which the new `Particle` is found based
on a scatter event consisting a translation of up to `𝜆` mean-free path along a new direction given relative
to the current direction of `p` via the scatter angles `𝜃` and `𝜑`.

Returns the updated `Particle` reflecting the last trajectory step and the Region for the next step.
"""
function take_step(
    p::T,
    reg::Region,
    𝜆::Real,
    𝜃::Real,
    𝜑::Real,
    ΔE::Real,
    ϵ::Real = 1.0e-12,
) where {T<:Particle}
    newP, nextReg = T(p, 𝜆, 𝜃, 𝜑, ΔE), reg
    t = min(
        intersection(shape(reg), newP), # Leave this Region?
        (intersection(shape(ch), newP) for ch in children(reg))..., # Enter a new child Region?
    )
    scatter = t > 1.0
    if !scatter # Enter new region
        newP = T(p, (t + ϵ) * 𝜆, 𝜃, 𝜑, (t + ϵ) * ΔE)
        nextReg = childmost_region(isnothing(parent(reg)) ? reg : parent(reg), position(newP))
    end
    return (newP, nextReg, scatter)
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
    p::T,
    reg::AbstractRegion,
    scf::Function,
    terminate::Function,
) where {T<:Particle}
    (pc, nextr) = (p, childmost_region(reg, position(p)))
    θ, ϕ = 0.0, 0.0
    while (!terminate(pc, reg)) && isinside(shape(reg), position(pc))
        prevr = nextr
        (λ, θₙ, ϕₙ, ΔZ) = scf(pc, nextr.material) 
        (pc, nextr, scatter) = take_step(pc, nextr, λ, θ, ϕ, ΔZ)
        (θ, ϕ) = scatter ? (θₙ, ϕₙ) : (0.0, 0.0) # scatter true? New angles. False? Old angles. 
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
