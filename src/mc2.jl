using GeometryBasics: Point3, Point2, Rect3, Vec3
import GeometryBasics: origin, widths, direction
using LinearAlgebra: normalize, normalize!

struct TransportSettings{ECS, BEL, MIP, T<:AbstractFloat, I<:Integer, Q}
    ϵ::T
    rtol::T
    maxiters::I
    quad::Q

    function TransportSettings(
        ::Type{ECS}, ::Type{BEL}, ::Type{MIP}; ϵ::Real=1e-13, rtol::Real=0.01, maxiters::Integer=10, nquad::Union{Integer, Nothing}=nothing
    ) where{ECS<:ElasticScatteringCrossSection, BEL<:BetheEnergyLoss, MIP<:NeXLMeanIonizationPotential}
        T = promote_type(typeof(ϵ), typeof(rtol))
        I = typeof(maxiters)
        quad = nquad isa Integer ? FastGLQ{T}(nquad) : nothing
        new{ECS, BEL, MIP, T, I, typeof(quad)}(T(ϵ), T(rtol), maxiters, quad)
    end
end

ecs(::TransportSettings{ECS}) where ECS = ECS
bel(::TransportSettings{<:Any, BEL}) where BEL = BEL
mip(::TransportSettings{<:Any, <:Any, MIP}) where MIP = MIP

function polardir(v::AbstractVector{T})::Point2{T} where T
    _v = normalize(v)
    θ = acos(max(min(_v[3], one(T)), -one(T)))
    ϕ = _v[1] == zero(T) ? T(π/2) : atan(_v[2] / _v[1])
    return Point2{T}(θ, ϕ)
end

function polaradd!(dir::AbstractVector{T}, Δ𝜃::T, Δ𝜑::T) where T
    (u, v, w) = dir
    cθ, sθ, cϕ, sϕ = cos(Δ𝜃), sin(Δ𝜃), cos(Δ𝜑), sin(Δ𝜑)
    if one(T) - abs(w) > sqrt(eps(T))
        dir[1] = u * cθ + sθ * (u * w * cϕ - v * sϕ) / sqrt(1.0 - w^2) #
        dir[2] = v * cθ + sθ * (v * w * cϕ + u * sϕ) / sqrt(1.0 - w^2) #
        dir[3] = w * cθ - sqrt(1.0 - w^2) * sθ * cϕ # 
    else
        dir[1] = sign(w) * sθ * cϕ #
        dir[2] = sign(w) * sθ * sϕ #
        dir[3] = sign(w) * cθ
    end
    normalize!(dir)
end

function polaradd(dir::V, Δ𝜃::T, Δ𝜑::T)::V where {T, V<:Union{SVector{3, T}, Point3{T}, Vec3{T}}}
    (u, v, w) = dir
    cθ, sθ, cϕ, sϕ = cos(Δ𝜃), sin(Δ𝜃), cos(Δ𝜑), sin(Δ𝜑)
    if one(T) - abs(w) > sqrt(eps(T))
        new_dir = V(
            u * cθ + sθ * (u * w * cϕ - v * sϕ) / sqrt(1.0 - w^2), #
            v * cθ + sθ * (v * w * cϕ + u * sϕ) / sqrt(1.0 - w^2), #
            w * cθ - sqrt(1.0 - w^2) * sθ * cϕ
        )
    else
        new_dir = V(
            sign(w) * sθ * cϕ, #
            sign(w) * sθ * sϕ, #
            sign(w) * cθ
        )
    end
    return normalize(new_dir)
end

function polaradd(dir::AbstractVector{T}, Δ𝜃::T, Δ𝜑::T) where {T}
    new_dir = copy(dir)
    polaradd!(new_dir, Δ𝜃, Δ𝜑)
    return new_dir
end


function distance(sh::Any, p::Particle{T}, startsinside::Bool) where T
    distance(sh, position(p), direction(p), startsinside)
end
function distance(sh::Any, p::Particle{T})::Tuple{T, Bool} where T
    return distance(sh, position(p), direction(p))
end
function distance(sh::Any, pos::AbstractVector{T}, dir::AbstractVector{T})::Tuple{T, Bool} where T
    startsinside = isinside(sh, pos)
    return distance(sh, pos, dir, startsinside), startsinside
end

function distance(sh::Rect3{T}, pos::AbstractArray{T}, dir::AbstractArray{T}, startsinside::Bool) where T
    t::T = typemax(T)
    corner1, corner2 = minimum(sh), maximum(sh)
    function interesects(t::T, i::I) where {T, I}
        j, k = mod1(i + one(I), I(3)), mod1(i + I(2), I(3))
        (corner1[j] ≤ pos[j] + t * dir[j] ≤ corner2[j]) && (corner1[k] ≤ pos[k] + t * dir[k] ≤ corner2[k])
    end
    for i in eachindex(pos)
        if dir[i] != zero(T)
            t1 = (corner1[i] - pos[i]) / dir[i]
            if (t1 > zero(T)) && (t1 < t) && (startsinside || interesects(t1, i))
                t = t1
            end
            t2 = (corner2[i] - pos[i]) / dir[i]
            if (t2 > zero(T)) && (t2 < t) && (startsinside || interesects(t2, i))
                t = t2
            end
        end
    end
    return t
end

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
abstract type Particle2{T} <: Particle{T} end
abstract type StaticParticle{T} <: Particle2{T} end
abstract type MutableParticle{T} <: Particle2{T} end

previous(p::Particle2) = error("previous position is not saved")
Base.position(p::Particle2) = p.position
current(p::Particle2) = p.position
energy(p::Particle2) = p.energy
direction(p::Particle2) = p.direction

function newparticle(p::P, λ::T, ΔE::T, Δ𝜃::T, Δ𝜑::T) where{T, P<:StaticParticle{T}}
    newpos = position(p) + λ * direction(p)
    newdir = polaradd(direction(p), Δ𝜃, Δ𝜑)
    newenergy = energy(p) + ΔE
    return P(newpos, newdir, newenergy) 
end

function newparticle(p::P, λ::T, ΔE::T) where{T, P<:StaticParticle{T}}
    newpos = position(p) + λ * direction(p)
    newenergy = energy(p) + ΔE
    return P(newpos, direction(p), newenergy) 
end

function newparticle(p::MutableParticle{T}, λ::T, ΔE::T, Δ𝜃::T, Δ𝜑::T) where T
    position(p) .+= λ * direction(p)
    polaradd!(direction(p), Δ𝜃, Δ𝜑)
    setenergy!(p, energy(p) + ΔE)
    return p
end

function newparticle(p::MutableParticle{T}, λ::T, ΔE::T) where T
    position(p) .+= λ * direction(p)
    setenergy!(p, energy(p) + ΔE)
    return p
end

function setenergy!(p::MutableParticle, E::Real)
    p.energy = E
end

struct SElectron{T<:AbstractFloat} <: StaticParticle{T}
    position::Point3{T}
    direction::Point3{T}
    energy::T

    """
        Electron(prev::Position, curr::Position, energy::Float64)
        Electron(el::Electron, 𝜆::Float64, 𝜃::Float64, 𝜑::Float64, ΔE::Float64)::Electron
    
    Create a new `Electron` from this one in which the new `Electron` is a distance `𝜆` from the
    first along a trajectory that is `𝜃` and `𝜑` off the current trajectory.
    """
    function SElectron{T}(position::Point3{T}, direction::Point3{T}, energy::T) where T
        new{T}(position, direction, energy)
    end
    function SElectron{T}(el::SElectron{T}, 𝜆::T, 𝜃::T, 𝜑::T, ΔE::T) where T
        new_dir = polaradd(direction(el), 𝜃, 𝜑)
        new_pos = position(el) + new_dir * 𝜆
        new{T}(new_pos, new_dir, energy(el) + ΔE)
    end
end

function SElectron_gen(prev::AbstractArray{<:Real}, curr::AbstractArray{<:Real}, energy::Real)
    SElectron(curr, curr .- prev, energy)
end

function SElectron(pos::AbstractArray{T1}, dir::AbstractArray{T2}, energy::T3) where {T1<:Real, T2<:Real, T3<:Real}
    T = promote_type(T1, T2, T3)
    SElectron{T}(Point3{T}(pos), normalize(Point3{T}(dir)), T(energy))
end

mutable struct MElectron{T<:AbstractFloat} <: MutableParticle{T}
    const position::MVector{3, T}
    const direction::MVector{3, T}
    energy::T

    """
        Electron(prev::Position, curr::Position, energy::Float64)
        Electron(el::Electron, 𝜆::Float64, 𝜃::Float64, 𝜑::Float64, ΔE::Float64)::Electron
    
    Create a new `Electron` from this one in which the new `Electron` is a distance `𝜆` from the
    first along a trajectory that is `𝜃` and `𝜑` off the current trajectory.
    """
    function MElectron{T}(pos::MVector{3, T}, dir::MVector{3, T}, energy::T) where T
        new{T}(pos, dir, energy)
    end
    function MElectron{T}(el::SElectron{T}, 𝜆::T, 𝜃::T, 𝜑::T, ΔE::T) where T
        polaradd!(el.direction, 𝜃, 𝜑)
        el.position .+= el.direction * 𝜆
        el.energy += ΔE
        return el
    end
end

function MElectron(pos::AbstractArray{T1}, dir::AbstractArray{T2}, energy::T3) where {T1<:Real, T2<:Real, T3<:Real}
    T = promote_type(T1, T2, T3)
    MElectron{T}(MVector{3, T}(pos), normalize!(MVector{3, T}(dir)), T(energy))
end

function MElectron_gen(prev::AbstractArray{<:Real}, curr::AbstractArray{<:Real}, energy::Real)
    MElectron(curr, curr .- prev, energy)
end

struct MElectron2{T<:AbstractFloat} <: MutableParticle{T}
    position::MVector{3, T}
    direction::MVector{3, T}
    energy::MVector{1, T}

    """
        Electron(prev::Position, curr::Position, energy::Float64)
        Electron(el::Electron, 𝜆::Float64, 𝜃::Float64, 𝜑::Float64, ΔE::Float64)::Electron
    
    Create a new `Electron` from this one in which the new `Electron` is a distance `𝜆` from the
    first along a trajectory that is `𝜃` and `𝜑` off the current trajectory.
    """
    function MElectron2{T}(pos::MVector{3, T}, dir::MVector{3, T}, energy::MVector{1, T}) where T
        new{T}(pos, dir, energy)
    end
    function MElectron2{T}(pos::MVector{3, T}, dir::MVector{3, T}, energy::T) where T
        new{T}(pos, dir, MVector{1, T}(energy))
    end
    function MElectron2{T}(el::SElectron{T}, 𝜆::T, 𝜃::T, 𝜑::T, ΔE::T) where T
        polaradd!(el.direction, 𝜃, 𝜑)
        el.position .+= el.direction * 𝜆
        el.energy[1] += ΔE
        return el
    end
end

function MElectron2(pos::AbstractArray{T1}, dir::AbstractArray{T2}, energy::T3) where {T1<:Real, T2<:Real, T3<:Real}
    T = promote_type(T1, T2, T3)
    MElectron2{T}(MVector{3, T}(pos), normalize!(MVector{3, T}(dir)), T(energy))
end

function MElectron2_gen(prev::AbstractArray{<:Real}, curr::AbstractArray{<:Real}, energy::Real)
    MElectron2(curr, curr .- prev, energy)
end

energy(el::MElectron2) = el.energy[1]

function setenergy!(p::MElectron2, E::Real)
    p.energy[1] = E
end

function randλelm(
    p::Particle{T},
    mat::AbstractMaterial,
    ::TransportSettings{ECS},
)::Tuple{T, Element} where {T, ECS}
    return randλelm(ECS, mat, energy(p))
end

function randλelm(
    p::Particle2{T},
    mat::ParametricMaterial,
    settings::TransportSettings{ECS},
)::Tuple{T, Element} where {T, ECS}
    return randλelm(
        ECS, mat, energy(p), position(p), direction(p), settings.rtol, settings.maxiters, settings.quad
    )
end

function ΔE(
    p::Particle2, 𝜆::Real, mat::AbstractMaterial, ::TransportSettings{<:Any, BEL, MIP}
) where {BEL<:BetheEnergyLoss, MIP<:NeXLMeanIonizationPotential}
    return dEds(BEL, energy(p), mat, MIP) * 𝜆
end
function ΔE(
    p::Particle2, 𝜆::Real, mat::ParametricMaterial, settings::TransportSettings{<:Any, BEL, MIP}
) where {BEL<:BetheEnergyLoss, MIP<:NeXLMeanIonizationPotential}
    # could lead to allocation depending on the types of elms_vector(mat) and massfracs(mat)
    pos, dir, e = position(p), direction(p), energy(p)
    sp = dEdρs.(BEL, Ref(e), elms_vector(mat), MIP)
    return quadrature(zero(𝜆), 𝜆, settings.quad) do l
        evaluateat(mat, pos .+ l .* dir) do m
            sum(sp .* massfracs(m)) * density(m)
        end
    end
end

function validate_step!(p::MutableParticle{T}, dE::T, newpos::AbstractVector{T}, 𝜆::T) where T
    if -dE ≥ energy(p) # Shorten path if energy is going into the negatives
        dEnew = -energy(p) + sqrt(eps(T))
        𝜆 = dEnew / dE * 𝜆
        dE = dEnew
        newpos .= position(p) .+ direction(p) .* 𝜆
    end
    setenergy!(p, energy(p) + dE)
    p.position .= newpos
end

function validate_step(p::Particle2{T}, 𝜆::T, dE::T)::Tuple{T, T} where T
    if -dE ≥ energy(p) # Shorten path if energy is going into the negatives
        dEnew = -energy(p) + sqrt(eps(T))
        𝜆 = dEnew / dE * 𝜆
        dE = dEnew
    end
    return 𝜆, dE 
end

function transport(
    p::P,
    reg::AbstractRegion,
    settings::TransportSettings,
)::Tuple{P, Bool} where {T<:AbstractFloat, P<:Particle2{T}}
#)::Tuple{SElectron{T}, Bool} where {T<:AbstractFloat, ECS, BEL, MIP}
    t, _ = distance(reg, p)
    #@info position(p), reg
    mat = material(reg)
    𝜆, elm = randλelm(p, mat, settings)
    dE = ΔE(p, 𝜆, mat, settings)
    𝜆, dE = validate_step(p, 𝜆, dE)
    scattered = t > 𝜆
    if scattered
        newp = newparticle(p, 𝜆, dE, rand(ecs(settings), elm, energy(p)+dE), rand(T)*T(2π))
    else
        𝜆new = t + settings.ϵ
        newp = newparticle(p, 𝜆new, 𝜆new / 𝜆 * dE)
    end
    @logmsg LogLevel(-10000) "is old pos in old reg? $(isinside(reg, position(p)))"
    return newp, scattered
end

function transporter(;
    ecs::Type{<:ElasticScatteringCrossSection}=Liljequist1989,
    bel::Type{<:BetheEnergyLoss}=JoyLuo,
    mip::Type{<:NeXLMeanIonizationPotential}=Berger1982,
    rtol::Real=0.01,
    maxiters::Integer=10,
    nquad::Integer=5,
    ϵ::Real=1e-15
)
    settings = TransportSettings(ecs, bel, mip; ϵ=ϵ, rtol=rtol, maxiters=maxiters, nquad=nquad)
    function _transport(p::P, reg::AbstractRegion)::Tuple{P, Bool} where {T, P<:Particle2{T}}
        transport(p, reg, settings)
    end
    return _transport
end

function trajectory2(
    eval::Function,
    p::P,
    reg::AbstractRegion,
    terminate::Function,
    settings::TransportSettings{<:Any, <:Any, <:Any, T}
) where {T, P<:Particle2{T}}
    pc::P = p
    t, startsinside = distance(reg, pc)
    if !startsinside
        pc = newparticle(pc, t + T(ϵ), zero(T))
    end
    nextreg = childmost_region(reg, position(pc))
    while !terminate(pc, reg) && !isnothing(nextreg)
        prevreg = nextreg
        (pc, scattered) = transport(pc, prevreg, settings)
        if !scattered
            nextreg = locate(position(pc), prevreg)
            if nextreg isa VoxelisedRegion
                @info position(pc), prevreg
            end
        end
        eval(pc, prevreg)
    end
end

function trajectory2(
    eval::Any,
    p::P,
    reg::AbstractRegion,
    settings::TransportSettings{<:Any, <:Any, <:Any, T};
    minE::Real=0.1
) where {T, P<:Particle2{T}}
    terminate(p::Particle, ::Any) = energy(p) < minE
    trajectory2(eval, p, reg, terminate, settings)
end

function trajectory2(
    eval::Function,
    p::P,
    reg::AbstractRegion,
    terminate::Function,
    ::Type{ECS}=Liljequist1989,
    ::Type{BEL}=JoyLuo,
    ::Type{MIP}=Berger1982;
    rtol::Real=0.01,
    maxiters::Integer=10,
    nquad::Union{Integer, Nothing}=nothing,
    ϵ::Real=1e-12
) where {T, ECS, BEL, MIP, P<:Particle2{T}}
    settings = TransportSettings(ECS, BEL, MIP; ϵ=T(ϵ), rtol=T(rtol), maxiters=maxiters, nquad=nquad)
    trajectory2(eval, p, reg, terminate, settings)
end

function trajectory2(
    eval::Function,
    p::P,
    reg::AbstractRegion,
    args::Type...;
    minE::Real=0.1,
    kwargs...
) where {T, P<:Particle2{T}}
    terminate(p::Particle, ::AbstractRegion) = energy(p) < minE
    trajectory2(eval, p, reg, terminate, args...; kwargs...)
end

abstract type AbstractVoxelRegion{DIM} <: AbstractRegion end

Base.size(::AbstractVoxelRegion{Tuple{N1, N2, N3}}) where {N1, N2, N3} = (N1, N2, N3)
Base.length(vr::AbstractVoxelRegion) = prod(size(vr))
material(vr::AbstractVoxelRegion) = error("$material is not defined for $(typeof(vr))")

function linear2cartesian(sz::NTuple{3, <:Integer}, index::I)::NTuple{3, I} where I<:Integer
    i = mod1(index, sz[1])
    index = (index-one(I)) ÷ sz[1] + one(I)
    j = mod1(index, sz[2])
    k = (index-one(I)) ÷ sz[2] + one(I)
    return (i, j, k)
end

function node(vr::AbstractVoxelRegion, index::Integer)
    node(vr, linear2cartesian(size(vr), index))
end

function populatevoxels!(res::AbstractVoxelRegion, mat_func::Any; kwargs...)
    for i in eachindex(children(res))
        res.children[i] = Voxel(i, mat_func(centroid(res, i)), res)
    end
end

function populatevoxels!(
    res::AbstractVoxelRegion, mat_temp::MaterialTemplate; massfrac_type::Type{<:AbstractFloat}=Float64, static::Bool=true, kwargs...
)
    for i in eachindex(children(res))
        pos = centroid(res, i)
        massfracs = MVector{length(mat_temp), massfrac_type} |> zero
        massfracfunc!(mat_temp)(massfracs, pos)
        ρ = densityfunc(mat_temp)(massfracs, pos)
        if static
            mat = STemplateMaterial(mat_temp, massfracs, ρ)
        else
            mat = MTemplateMaterial(mat_temp, massfracs, ρ; kwargs...)
        end
        res.children[i] = Voxel(i, mat, res)
    end
end

function point_inside(
    vr::AbstractVoxelRegion, index::NTuple{3, <:Integer}, offset::Union{Tuple, AbstractArray}
) 
    widths = child_widths(vr)
    _node = node(vr, index)
    return _node .+ offset .* widths
end
function point_inside(
    vr::AbstractVoxelRegion, index::Integer, offset::Union{Tuple, AbstractArray}
)
    point_inside(vr, linear2cartesian(size(vr), index), offset)
end

centroid(vr::AbstractVoxelRegion, index::Union{NTuple{3, <:Integer}, Integer}) = point_inside(vr, index, (0.5, 0.5, 0.5))

# Note: This function assumes pos is inside region
function childmost_index(reg::AbstractVoxelRegion{Tuple{Nx, Ny, Nz}}, pos::AbstractArray{<:Real}) where {Nx, Ny, Nz}
    idx = ceil.(typeof(Nx), (pos .- origin(reg)) ./ child_widths(reg))
    idx = max.(min.(idx, (Nx, Ny, Nz)), one(Nx))
    return Tuple(idx)
end

# Note: This function assumes pos is inside region
function childmost_region(reg::AbstractVoxelRegion, pos::AbstractArray{<:Real})
    return children(reg)[childmost_index(reg, pos)...]
end

abstract type AbstractNestedVoxelRegion{DIM, P} <: AbstractVoxelRegion{DIM} end

child_widths(vx::AbstractNestedVoxelRegion) = child_widths(parent(vx)) ./ size(vx)
shape(vx::AbstractNestedVoxelRegion) = shape(parent(vx), vx.index)
node(vx::AbstractNestedVoxelRegion, index::Union{NTuple{3, <:Integer}}) = node(parent(vx), vx.index) .+ child_widths(vx) .* index
origin(vx::AbstractNestedVoxelRegion) = npde(parent(vx), vx.index)

struct VoxelisedRegion{DIM, T} <: AbstractVoxelRegion{DIM}
    shape::Rect3{T}
    parent::AbstractRegion
    children::Array{AbstractVoxelRegion, 3}
    nodes::Vector{Vector{T}}
    name::String
    child_widths::Vec3{T}

    function VoxelisedRegion(
        sh::Rect3{T},
        parent::Union{Nothing,AbstractRegion},
        num_voxels::Union{NTuple{3, <:Integer}, AbstractVector{<:Integer}};
        name::Union{Nothing,String} = nothing,
        ntests=1000,
    ) where {T<:AbstractFloat}
        maxsize = Float64(Sys.total_memory()) / 200
        if prod(num_voxels) > maxsize
            error("Voxels are too big for memory")
        end
        name = something(
            name,
            isnothing(parent) ? "Root" : "$(parent.name)[$(length(parent.children)+1)]",
        )
        child_widths = Vec3{T}(widths(sh) ./ num_voxels)

        #nodes = [(sh.origin[1] + i * voxel_sizes[1], 
        #sh.origin[2] + j * voxel_sizes[2], 
        #sh.origin[3] + k * voxel_sizes[3] ) for i in 0:num_voxels[1], j in 0:num_voxels[2], k in 0:num_voxels[3]]
        nodes = [sh.origin[i] .+ collect(0:num_voxels[i]) .* child_widths[i] for i in 1:3]
        
        (N1, N2, N3) = num_voxels
        voxels = Array{AbstractVoxelRegion}(undef, N1, N2, N3)
        res = new{Tuple{N1, N2, N3}, T}(sh, parent, voxels, nodes, name, child_widths)
        
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
                children(parent),
            ) "The child $sh overlaps a child of the parent shape."
            push!(children(parent), res)
        end
        return res
    end
end

function VoxelisedRegion(
    sh::Rect3{T},
    parent::Union{Nothing,AbstractRegion},
    num_voxels::Union{NTuple{3, <:Integer}, AbstractVector{<:Integer}},
    mat_func::Any;
    name::Union{Nothing,String}=nothing,
    ntests::Integer=1000,
    kwargs...
) where {T<:AbstractFloat}
    res = VoxelisedRegion(sh, parent, num_voxels; name=name, ntests=ntests)
    populatevoxels!(res, mat_func; kwargs...)
    return res
end

function Base.show(io::IO, ::MIME"text/plain", vr::VoxelisedRegion)
    print(io, "$(typeof(vr)):")
    print(io, "\n name: ", name(vr))
    print(io, "\n shape: Rect(origin=$(Tuple(origin(shape(vr)))), widths=$(Tuple(widths(shape(vr)))))")
    print(io, "\n voxel_size: $(vr.voxel_sizes)")
    print(io, "\n parent: ", split("$P", "{")[1], "(name=$(name(parent(vr))))")
end
function Base.show(io::IO, vr::VoxelisedRegion)
    print(io, "$(typeof(vr))[", name(vr), ", origin=$(origin(shape(vr))), voxel_sizes=$(vr.voxel_sizes)]")
end

child_widths(vr::VoxelisedRegion) = vr.child_widths

material(::VoxelisedRegion) = error("VoxelisedRegion doesn't have a material")

shape(vr::VoxelisedRegion{<:Any, T}, index::Union{NTuple{3, <:Integer}, Integer}) where T = Rect3{T}(node(vr, index), child_widths(vr))

node(vr::VoxelisedRegion{<:Any, T}, i::Integer, j::Integer, k::Integer) where T = Vec3{T}(vr.nodes[1][i], vr.nodes[2][j], vr.nodes[3][k])
node(vr::VoxelisedRegion, index::NTuple{3, <:Integer}) = node(vr, index[1], index[2], index[3])

origin(vr::VoxelisedRegion) = origin(shape(vr))

struct NestedVoxelisedRegion{DIM, P<:AbstractVoxelRegion, I<:Integer} <: AbstractNestedVoxelRegion{DIM, P}
    index::I
    parent::P
    children::Array{AbstractVoxelRegion, 3}

    function NestedVoxelisedRegion(
        index::I,
        parent::P,
        num_voxels::Union{NTuple{3, <:Integer}, AbstractVector{<:Integer}}
    ) where {I<:Integer, P<:AbstractVoxelRegion}
        
        (N1, N2, N3) = num_voxels
        voxels = Array{AbstractVoxelRegion}(undef, N1, N2, N3)
        res = new{Tuple{N1, N2, N3}, P, I}(index, parent, voxels)
        return res
    end
end

function NestedVoxelisedRegion(
    index::Integer,
    parent::AbstractVoxelRegion,
    num_voxels::Union{NTuple{3, <:Integer}, AbstractVector{<:Integer}},
    mat_func::Any;
    kwargs...
) 
    res = NestedVoxelisedRegion(index, parent, num_voxels)
    populatevoxels!(res, mat_func; kwargs...)
    return res
end

name(vx::NestedVoxelisedRegion) = "NestedVoxelisedRegion[$(linear2cartesian(size(parent(vx)), vx.index))] of $(name(parent(vx)))"

function Base.show(io::IO, ::MIME"text/plain", vx::NestedVoxelisedRegion)
    print(io, "$(name(vx)):\n")
    print(io, "size: $(size(vx))\n")
    print(io, "shape: $(shape(vx))\n")
    print(io, "parent_size: $(size(parent(vx)))")
end
function Base.show(io::IO, vx::NestedVoxelisedRegion)
    print(io, "$(typeof(vx))[$(name(vx)), size=$(size(vx)), shape=$(shape(vx))]")
end

struct Voxel{P, I<:Integer, M} <: AbstractNestedVoxelRegion{Tuple{1, 1, 1}, P}
    index::I
    material::M
    parent::P
end

name(vx::Voxel) = "Voxel[$(linear2cartesian(size(parent(vx)), vx.index))] of $(name(parent(vx)))"
material(vx::Voxel) = vx.material

function Base.show(io::IO, ::MIME"text/plain", vx::Voxel)
    print(io, "$(name(vx)):\n")
    print(io, "material: $(material(vx))\n")
    print(io, "shape: $(shape(vx))\n")
    print(io, "parent_size: $(size(parent(vx)))")
end
function Base.show(io::IO, vx::Voxel)
    print(io, "$(typeof(vx))[$(name(vx)), mat=$(material(vx)), shape=$(shape(vx))]")
end

child_widths(vx::Voxel) = child_widths(parent(vx))
children(::Voxel) = ()
haschildren(::Voxel) = false

# warning this only works well when pos is not in a child region of reg
function distance(reg::AbstractRegion, p::Particle{T})::Tuple{T, Bool} where T
    t, startsinside = distance(shape(reg), p)
    if startsinside
        for ch in reg.children
            tch = distance(shape(ch), p, false)
            if (tch < t) t = tch end
        end
    end
    return t, startsinside
end

function distance(reg::AbstractVoxelRegion, p::Particle{T})::Tuple{T, Bool} where T
    sh = shape(reg)
    if isinside(sh, p)
        @debug "$(position(p)) is inside $(split(string(typeof(reg)), "{")[1])(shape=$sh; should be called from a Voxel instead"
        return distance(shape(reg, childmost_index(reg, position(p))), p, true), true
    end
    return distance(shape(reg), p, false), false
end

function distance(reg::Voxel, p::Particle{T})::Tuple{T, Bool} where T
    sh = shape(reg)
    if !isinside(sh, p)
        @debug "$(position(p)) doesn't start inside Voxel(shape=$sh"
        return distance(shape(parent(reg), childmost_index(parent(reg), position(p))), p, true), true
    end
    return distance(shape(reg), p, true), true
end

function distance_(reg::Voxel, p::Particle{T})::Tuple{T, Bool} where T
    return distance(shape(reg), p, true), true
end

# Note: This function assumes pos is inside region
childmost_region(reg::Voxel, ::AbstractArray{<:Real}) = reg

function locate(pos::AbstractArray{<:Real}, guess::AbstractRegion)
    if isinside(shape(guess), pos)
        return childmost_region(guess, pos)
    else
        return locate(pos, parent(guess))
    end
end

locate(::AbstractArray, ::Nothing) = nothing

locate(pos::AbstractArray{<:Real}, guess::Voxel) = locate(pos, parent(guess))
