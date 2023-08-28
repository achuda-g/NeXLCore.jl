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

isinside(rr::RectangularShape, pos::AbstractArray{Float64}) =
    all(pos .> minimum(rr)) && all(pos .< maximum(rr)) # write for voxels i, i + 1

function intersection(
    rr::RectangularShape,
    pos1::AbstractArray{Float64},
    pos2::AbstractArray{Float64},
)::Float64
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

isinside(sr::SphericalShape, pos::AbstractArray{Float64}) =
    norm(pos .- origin(sr)) < radius(sr)

function intersection(
    sr::SphericalShape,
    pos0::AbstractArray{Float64},
    pos1::AbstractArray{Float64},
)::Float64
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
function random_point_inside(shape)::Position
    res = Position(origin(shape) .+ rand(Position) .* widths(shape))
    while !isinside(shape, res)
        res = Position(origin(shape) .+ rand(Position) .* widths(shape))
    end
    return res
end

#= # Glen - moved this to another script
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
=#

"""
    transport(pc::Electron, mat::Material, ecx=Liljequist1989, bethe=JoyLuo)::NTuple{4, Float64}
    transport(pc::Electron, mat::Material, num_iterations::Int, ecx=Liljequist1989, bethe=JoyLuo)::NTuple{4, Float64}

The default function defining elastic scattering and energy loss for an Electron.

Returns ( `λ`, `θ`, `ϕ`, `ΔE`) where `λ` is the mean path length, `θ` is the elastic scatter angle, `ϕ` is the azimuthal elastic scatter
angle and `ΔE` is the energy loss for transport over the distance `λ`. 'Num_iterations' is the number of desired iterations for the integrations.
"""
function transport(
    pc::Electron,
    mat::Material, #Function - elements fixed with mass fractions changing
    ecx::Type{<:ElasticScatteringCrossSection} = Liljequist1989,
    bethe::Type{<:BetheEnergyLoss} = JoyLuo,
)::NTuple{4,Float64}
    (𝜆′, θ′, ϕ′) = rand(ecx, mat, pc.energy) 
    return (𝜆′, θ′, ϕ′, 𝜆′ * dEds(bethe, pc.energy, mat))
end

function transport( #should work with parametric material now - Glen 
    pc::Electron,
    mat::ParametricMaterial,
    num_iterations::Int,
    ecx::Type{<:ElasticScatteringCrossSection} = Liljequist1989,
    bethe::Type{<:BetheEnergyLoss} = JoyLuo
)::NTuple{4,Float64}
    (𝜆′, θ′, ϕ′) = rand(ecx, pc, mat, pc.energy, num_iterations) 
    stopval = dEds(bethe, pc.energy, position(pc), mat)
    for i in 1:num_iterations
        integral, error = quadgk(x -> dEds(bethe, pc.energy, mat, x, θ′, ϕ′, pc), 0, 𝜆′)
        stopping_val = integral / stopval 
        stopval = stopping_val
    end
    return (𝜆′, θ′, ϕ′, 𝜆′ * stopval)
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

struct VoxelShape
    index::NTuple
    parent::AbstractRegion
end

struct Voxel{M} <: AbstractRegion{M}
    shape::VoxelShape
    material::M
    parent::AbstractRegion
    children::Vector{Nothing}
    #name::String

    function Voxel(
        index::NTuple,
        mat::M,
        parent::AbstractRegion,
        #name::String = "",
    ) where {M<:Material}
        @assert mat[:Density] > 0.0
        #name = name * "$index"
        shape = VoxelShape(index, parent)
        return new{M}(shape, mat, parent, Vector{Nothing}()) # Glen - nothing in children region
    end
end

function rect(vx::VoxelShape)
    i, j, k = vx.index
    RectangularShape(nodes(vx.parent, i, j, k), vx.parent.voxel_sizes)
end

function isinside(vx::VoxelShape, pos::AbstractArray{Float64})
    all(pos .> nodes(vx.parent, i, j, k)) && all(pos .< nodes(vx.parent, i+1, j+1, k+1)) # write for voxels i, i + 1
end

function intersection( # how to make this more efficient? 
    vx::VoxelShape,
    pos1::AbstractArray{Float64},
    pos2::AbstractArray{Float64},
)::Float64
    _between(a, b, c) =  b < a < c #(a > b) && (a < c)
    t = Inf64
    i, j, k = vx.index
    #corner1, corner2 = vx.parent.nodes[i,j,k], vx.parent.nodes[i+1,j+1,k+1] # is this correct?
    corner1, corner2 = nodes(vx.parent, i,j,k), nodes(vx.parent, i+1,j+1,k+1) # is this correct?
    for i in eachindex(pos1)
        j, k = i % 3 + 1, (i + 1) % 3 + 1
        if pos2[i] != pos1[i]
            u = (corner1[i] - pos1[i]) / (pos2[i] - pos1[i])
            if (u > 0.0) && (u <= t)
                t = u
            end
            u = (corner2[i] - pos1[i]) / (pos2[i] - pos1[i])
            if (u > 0.0) && (u <= t)
                t = u
            end
        end
    end
    return t
end

struct VoxelisedRegion{M} <: AbstractRegion{M}
    shape::GeometryPrimitive{3, Float64}
    parent::Union{Nothing, AbstractRegion}
    children::Vector{AbstractRegion}
    material::M
    voxels::Array{Voxel, 3}
    nodes::Vector{Vector{Float64}}
    name::String
    voxel_sizes::NTuple{3, Float64}
    num_voxels::Tuple{Int64, Int64, Int64}

    function VoxelisedRegion(
        sh::RectangularShape,
        mat::M,
        mat_func::Function,
        parent::Union{Nothing,AbstractRegion},
        num_voxels::Tuple{Int64, Int64, Int64},
        name::Union{Nothing,String} = nothing,
    ) where M
        name = something(
            name,
            isnothing(parent) ? "Root" : "$(parent.name)[$(length(parent.children)+1)]",
        )

        voxel_sizes = (sh.widths[1] / num_voxels[1], sh.widths[2] / num_voxels[2], sh.widths[3] / num_voxels[3])

        #nodes = [(sh.origin[1] + i * voxel_sizes[1], 
        #sh.origin[2] + j * voxel_sizes[2], 
        #sh.origin[3] + k * voxel_sizes[3] ) for i in 0:num_voxels[1], j in 0:num_voxels[2], k in 0:num_voxels[3]]
        nodes = [sh.origin[i] .+ collect(0:num_voxels[i]) .* voxel_sizes[i] for i in 1:3]
        
        voxels = Array{Voxel}(undef, num_voxels[1], num_voxels[2], num_voxels[3])
        res = new{M}(sh, parent, Vector{AbstractRegion}(), mat, voxels, nodes, name, voxel_sizes, num_voxels)

        for i in 1:num_voxels[1]
            for j in 1:num_voxels[2]
                for k in 1:num_voxels[3]
                    pos = Position(nodes[1][i] + nodes[1][i+1], nodes[2][j] + nodes[2][j+1], nodes[3][k] + nodes[3][k+1]) .* 0.5
                    voxels[i, j, k] = Voxel((i, j, k), mat_func(pos), res)
                end
            end
        end

        if !isnothing(parent)
	    tolerance = eps(Float64)
	    vertices = [
    		sh.origin + Point(tolerance, tolerance, tolerance),
    		sh.origin + Point(0, 0, sh.widths[3]) - Point(0, 0, tolerance) + Point(tolerance, tolerance, 0),
    		sh.origin + Point(0, sh.widths[2], 0) - Point(0, tolerance, 0) + Point(tolerance, 0, tolerance),
    		sh.origin + Point(0, sh.widths[2], sh.widths[3]) - Point(0, tolerance, tolerance) + Point(tolerance, 0, 0),
    		sh.origin + Point(sh.widths[1], 0, 0) - Point(tolerance, 0, 0) + Point(0, tolerance, tolerance),
    		sh.origin + Point(sh.widths[1], 0, sh.widths[3]) - Point(tolerance, 0, tolerance) + Point(0, tolerance, 0),
    		sh.origin + Point(sh.widths[1], sh.widths[2], 0) - Point(tolerance, tolerance, 0) + Point(0, 0, tolerance),
    		sh.origin + Point(sh.widths[1], sh.widths[2], sh.widths[3]) - Point(tolerance, tolerance, tolerance),
	    ]
	    @assert all(isinside(parent.shape, v) for v in vertices) "The child $sh is not fully contained within the parent $(parent.shape)."

	    @assert all(
    		ch -> all(!isinside(ch.shape, v) for v in vertices),
    		parent.children,
	    ) "The child $sh overlaps a child of the parent shape."

	    push!(parent.children, res)
        else
        end
        return res
    end
end

nodes(vr::VoxelisedRegion, i::Integer, j::Integer, k::Integer) = (vr.nodes[1][i], vr.nodes[2][j], vr.nodes[3][k])
    

struct Region{M} <: AbstractRegion{M}
    shape::GeometryPrimitive{3,Float64}
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
function childmost_region(reg::Region, pos::AbstractArray{Float64})::AbstractRegion
    res = findfirst(ch -> isinside(ch.shape, pos), reg.children)
    return !isnothing(res) ? childmost_region(reg.children[res], pos) : reg
end
# Glen - check if the below is necessary
function childmost_region(reg::Voxel, pos::AbstractArray{Float64})::Union{VoxelisedRegion, Voxel} # in a voxel, the voxelisedregion will be passed here
    # take_step function determines if the voxel has a parent, if so then the parent is input here
    # from this parent, the child region containing the position is chosen 
    res = findfirst(ch -> isinside(ch.shape, pos), reg.children)
    return !isnothing(res) ? childmost_region(reg.children[res], pos) : reg # res is NOT nothing? child found, pos contained in child.
    # res = nothing? No child contains the position, region input is returned. 
end
# Glen - check if the above is necessary
function childmost_region(reg::AbstractRegion, pos::AbstractArray{Float64})::AbstractRegion # Glen - to deal with the recursive calling of this function 
    if reg isa Region
        return childmost_region(Region(reg), pos)
    elseif reg isa VoxelisedRegion || reg isa Voxel
        return childmost_region(Union{VoxelisedRegion, Voxel}(reg), pos)
    else
        error("Unsupported region type: ", typeof(reg))
    end
end
function childmost_region(reg::VoxelisedRegion, pos::AbstractArray{Float64})::Union{VoxelisedRegion, Voxel}
    res = findfirst(ch -> isinside(ch.shape, pos), reg.children)

    if !isnothing(res)
        return childmost_region(reg.children[res], pos)
    else
        voxel_indices = find_voxel_by_position(reg, pos)
        voxel = reg.voxels[voxel_indices...]
        return voxel
    end
end


"""
    find_voxel_by_position(voxel_boundaries, pos)

Find the next Voxel containing the point `pos`.
"""
function find_voxel_by_position(vr::VoxelisedRegion, pos) 
    return ceil.(Int, (pos - vr.shape.origin) ./ vr.voxel_sizes) # returns indices of voxels
end

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
    𝜆::Float64,
    𝜃::Float64,
    𝜑::Float64,
    ΔE::Float64,
    ϵ::Float64 = 1.0e-12,
)::Tuple{T,AbstractRegion,Bool} where {T<:Particle}
    newP, nextReg = T(p, 𝜆, 𝜃, 𝜑, ΔE), reg
    t = min(
        intersection(reg.shape, newP), # Leave this Region?
        (intersection(ch.shape, newP) for ch in reg.children)..., # Enter a new child Region?
    )
    scatter = t > 1.0
    if !scatter # Enter new region
        newP = T(p, (t + ϵ) * 𝜆, 𝜃, 𝜑, (t + ϵ) * ΔE)
        nextReg = childmost_region(isnothing(reg.parent) ? reg : reg.parent, position(newP))
    end
    return (newP, nextReg, scatter)
end

function take_step(
    p::T,
    reg::Voxel,
    𝜆::Float64,
    𝜃::Float64,
    𝜑::Float64,
    ΔE::Float64,
    ϵ::Float64 = 1.0e-12,
)::Tuple{T,AbstractRegion,Bool} where {T<:Particle}
    newP, nextReg = T(p, 𝜆, 𝜃, 𝜑, ΔE), reg
    t = min(
        intersection(reg.shape, newP),
        (intersection(ch.shape, newP) for ch in reg.children)...,
    )
    scatter = t > 1.0
    if !scatter
        newP = T(p, (t + ϵ) * 𝜆, 𝜃, 𝜑, (t + ϵ) * ΔE)
        if isa(nextReg, Voxel)
            voxel_idxs = find_voxel_by_position(nextReg.parent, position(newP))
            if all(1 .<= voxel_idxs .<= nextReg.parent.num_voxels)
                nextReg = nextReg.parent.voxels[voxel_idxs...] 
            else
                vr = nextReg.parent
                nextReg = childmost_region(isnothing(vr.parent) ? nothing : vr.parent, position(newP)) #could return nothing
            end
        end
    end
    return (newP, nextReg, scatter)
end

function take_step(
    p::T,
    reg::VoxelisedRegion,
    𝜆::Float64,
    𝜃::Float64,
    𝜑::Float64,
    ΔE::Float64,
    ϵ::Float64 = 1.0e-12,
)::Tuple{T,AbstractRegion,Bool} where {T<:Particle}
    @assert isinside(reg.shape, position(p)) position(p), minimum(reg.shape), maximum(reg.shape)
    voxel_idxs = find_voxel_by_position(reg, position(p))
    if any(1 .> voxel_idxs .|| voxel_idxs .> reg.num_voxels)
        error()
    end
    regv = reg.voxels[voxel_idxs...]
    regv=reg
    take_step(p, regv, 𝜆, 𝜃, 𝜑, ΔE, ϵ)
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
    while (!terminate(pc, reg)) && isinside(reg.shape, position(pc))
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
    scf::Function = (t::T, mat::Material) -> transport(t, mat); 
    minE::Float64 = 50.0,
) where {T<:Particle}
    term(pc::T, _::AbstractRegion) = pc.energy < minE
    trajectory(eval, p, reg, scf, term)
end

function trajectory(
    eval::Function,
    p::T,
    reg::AbstractRegion,
    mat::ParametricMaterial,
    scf::Function,
    terminate::Function,
) where {T<:Particle}
    (pc, nextr) = (p, childmost_region(reg, position(p)))
    θ, ϕ = 0.0, 0.0
    while (!terminate(pc, reg)) && isinside(reg.shape, position(pc)) 
        prevr = nextr
        (λ, θₙ, ϕₙ, ΔZ) = scf(pc, mat, 4) 
        (pc, nextr, scatter) = take_step(pc, nextr, λ, θ, ϕ, ΔZ)
        (θ, ϕ) = scatter ? (θₙ, ϕₙ) : (0.0, 0.0)
        eval(pc, prevr)
    end
end
function trajectory(
    eval::Function,
    p::T,
    reg::AbstractRegion,
    mat::ParametricMaterial,
    scf::Function = (t::T, mat::ParametricMaterial, num_it::Int) -> transport(t, mat, 4); # Adjusted the function definition to include θ, ϕ
    minE::Float64 = 50.0,
) where {T<:Particle}
    term(pc::T, _::AbstractRegion) = pc.energy < minE
    trajectory(eval, p, reg, mat, scf, term)
end


# SINGLE FUNCTIONS TO TEST RANDOM PATH LENGTHS

# This isn't working... why?
# It does all iterations within the region volume, and the regular trajectory function does not, even though nothing is different between them  
function trajectory_rand_path_check(
    eval::Function,
    p::T,
    reg::AbstractRegion,
    #total_path_length_vector::Vector{Float64},
    #scatter_angle_vector::Vector{Float64},
    scf::Function,
    terminate::Function,
) where {T<:Particle}
    (pc, nextr) = (p, childmost_region(reg, position(p)))
    θ, ϕ = 0.0, 0.0
    total_path_length = 0
    while (!terminate(pc, reg)) && isinside(reg.shape, position(pc)) # isinside function also breaks it?!
        prevr = nextr
        (λ, θₙ, ϕₙ, ΔZ) = scf(pc, nextr.material) 
        (pc, nextr, scatter) = take_step(pc, nextr, λ, θ, ϕ, ΔZ) 
        total_path_length += λ
        if !scatter 
            (θ, ϕ) = (0.0,0.0)
        else
            (θ, ϕ) = (θₙ, ϕₙ) 
            #append!(total_path_length_vector, total_path_length)
            #append!(scatter_angle_vector, θₙ)
            #total_path_length = 0
            break
        end
        eval(pc, prevr)
    end
    return total_path_length, θ
end

function trajectory_rand_path_check(
    eval::Function,
    p::T,
    reg::AbstractRegion,
    #tplv::Vector{Float64},
    #sav::Vector{Float64},
    scf::Function = (t::T, mat::Material) -> transport(t, mat); 
    minE::Float64 = 50.0,
) where {T<:Particle}
    term(pc::T, _::AbstractRegion) = pc.energy < minE
    #trajectory_rand_path_check(eval, p, reg, tplv, sav, scf, term)
    trajectory_rand_path_check(eval, p, reg, scf, term)
end

# PARAMETRIC

function trajectory_rand_path_check(
    eval::Function,
    p::T,
    reg::AbstractRegion,
    mat::ParametricMaterial,
    #total_path_length_vector::Vector{Float64},
    #scatter_angle_vector::Vector{Float64},
    scf::Function,
    terminate::Function,
) where {T<:Particle}
    (pc, nextr) = (p, childmost_region(reg, position(p)))
    θ, ϕ = 0.0, 0.0
    total_path_length = 0
    while (!terminate(pc, reg)) && isinside(reg.shape, position(pc)) 
        prevr = nextr
        (λ, θₙ, ϕₙ, ΔZ) = scf(pc, mat, 4) 
        (pc, nextr, scatter) = take_step(pc, nextr, λ, θ, ϕ, ΔZ)
        total_path_length += λ
        if !scatter 
            (θ, ϕ) = (0.0,0.0)
        else
            (θ, ϕ) = (θₙ, ϕₙ) 
            #append!(total_path_length_vector, total_path_length)
            #append!(scatter_angle_vector, θₙ)
            #total_path_length = 0
            break
        end
        eval(pc, prevr)
    end
    return total_path_length, θ
end

function trajectory_rand_path_check(
    eval::Function,
    p::T,
    reg::AbstractRegion,
    mat::ParametricMaterial,
    #tplv::Vector{Float64},
    #sav::Vector{Float64},
    scf::Function = (t::T, mat::ParametricMaterial, num_it::Int) -> transport(t, mat, 4); # Adjusted the function definition to include θ, ϕ
    minE::Float64 = 50.0,
) where {T<:Particle}
    term(pc::T, _::AbstractRegion) = pc.energy < minE
    #trajectory_rand_path_check(eval, p, reg, mat, tplv, sav, scf, term)
    trajectory_rand_path_check(eval, p, reg, mat, scf, term)
end