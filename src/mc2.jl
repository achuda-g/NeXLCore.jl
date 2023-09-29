import GeometryBasics

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
abstract type MutableParticle{T} <: Particle end

direction(p::MutableParticle) = p.direction
energy(p::MutableParticle) = @inbounds p.energy[2]
function energy!(p::MutableParticle, val::Real)
    @inbounds p.energy[2] = val
end
previous_energy(p::MutableParticle) = @inbounds p.energy[1]
function previous_energy!(p::MutableParticle, val::Real)
    @inbounds p.energy[1] = val
end
function move!(p::MutableParticle, λ::Real)
    previous(p) .= current(p)
    current(p) .+= direction(p) .* λ
end
function move_redo!(p::MutableParticle, λ::Real)
    current(p) .= previous(p) .+ direction(p) .* λ
end
function alter_energy!(p::MutableParticle, ΔE::Real)
    previous_energy!(p, energy(p))
    energy!(p, energy(p) + ΔE)
end

struct MElectron{T<:AbstractFloat} <: MutableParticle{T}
    previous::MVector{3, T}
    current::MVector{3, T}
    direction::MVector{3, T}
    energy::MVector{2, T}

    """
        Electron(prev::Position, curr::Position, energy::Float64)
        Electron(el::Electron, 𝜆::Float64, 𝜃::Float64, 𝜑::Float64, ΔE::Float64)::Electron
    
    Create a new `Electron` from this one in which the new `Electron` is a distance `𝜆` from the
    first along a trajectory that is `𝜃` and `𝜑` off the current trajectory.
    """
    function MElectron{T}(prev::AbstractArray, curr::AbstractArray, dir::AbstractArray, prev_energy::Real, energy::Real) where T
        dir = LinearAlgebra.normalize(dir)
        new{T}(MVector{3, T}(prev), MVector{3, T}(curr), MVector{3, T}(dir), MVector{2, T}(prev_energy, energy))
    end
    function MElectron(prev::AbstractArray, curr::AbstractArray, dir::AbstractArray, prev_energy::Real, energy::Real)
        T = promote_type(eltype(prev), eltype(curr), eltype(dir), typeof(prev_energy), typeof(energy))
        MElectron{T}(prev, curr, dir, prev_energy, energy)
    end
    function MElectron(prev::AbstractArray, curr::AbstractArray, prev_energy::Real, energy::Real)
        MElectron(prev, curr, direction(curr, prev), prev_energy, energy)
    end
    function MElectron(prev::AbstractArray, curr::AbstractArray, energy::Real)
        MElectron(prev, curr, energy, energy)
    end
end

function Base.copy(::Type{T}, el::MElectron) where T
    MElectron{T}(previous(el), current(el), direction(el), previous_energy(el), energy(el))
end

Base.copy(el::MElectron{T}) where T = copy(T, el)

function deflect!(el::MElectron, 𝜃::Real, 𝜑::Real)
    (u, v, w) = direction(el)
    cθ, sθ, cϕ, sϕ = cos(𝜃), sin(𝜃), cos(𝜑), sin(𝜑)
    if 1.0 - abs(w) > 1e-8
        @inbounds direction(el)[1] = u * cθ + sθ * (u * w * cϕ - v * sϕ) / sqrt(1.0 - w^2)
        @inbounds direction(el)[2] = v * cθ + sθ * (v * w * cϕ + u * sϕ) / sqrt(1.0 - w^2)
        @inbounds direction(el)[3] = w * cθ - sqrt(1.0 - w^2) * sθ * cϕ
    else
        @inbounds direction(el)[1] = sign(w) * sθ * cϕ
        @inbounds direction(el)[2] = sign(w) * sθ * sϕ
        @inbounds direction(el)[3] = sign(w) * cθ
    end
    LinearAlgebra.normalize!(direction(el))
end

function move!(
    ::Type{S},
    p::MutableParticle,
    mat::AbstractMaterial,
    ::Real,
    ::Integer,
    ::Integer
) where {S<:ElasticScatteringCrossSection}
    𝜆 = randλ(S, mat, energy(p))
    move!(p, 𝜆)
    return 𝜆
end

function move!(
    ::Type{S},
    p::MutableParticle,
    mat::ParametricMaterial,
    rtol::Real,
    maxiter::Integer,
    nquad::Integer
) where {S<:ElasticScatteringCrossSection}
    𝜆 = randλ(S, mat, energy(p), current(p), direction(p), rtol, maxiter, nquad)
    move!(p, 𝜆)
    return 𝜆
end

function transport(
    p::MutableParticle{T},
    reg::AbstractRegion,
    ecs::Type{<:ElasticScatteringCrossSection}=ScreenedRutherford,
    bel::Type{<:BetheEnergyLoss}=JoyLuo,
    mip::Type{<:NeXLMeanIonizationPotential}=Berger1982,
    rtol::Real=0.01,
    maxiter::Integer=10,
    nquad::Integer=5,
) where T
    newreg::Union{AbstractRegion, Nothing} = reg
    𝜆 = move!(ecs, p, material(reg), rtol, maxiter, nquad)
    t = intersection(reg, previous(p), current(p))
    scattered = t > 1
    if scattered
        (𝜃, 𝜙) = scatter(ecs, material(reg), energy(p))
        deflect!(p, 𝜃, 𝜙)
    else
        𝜆 = 𝜆*t + eps(T)
        move_redo!(p, 𝜆)
        newreg = find_region(current(p), reg)
    end
    @logmsg LogLevel(-10000) "is old pos in old reg? $(isinside(reg, previous(p)))"
    @logmsg LogLevel(-10000) "is new pos in new reg? $(isinside(newreg, current(p)))"
    alter_energy!(p, 𝜆, material(reg), bel, mip, nquad)
    return p, newreg
end

function alter_energy!(
    p::MutableParticle, 𝜆::Real, mat::AbstractMaterial, ::Type{BEL}, ::Type{MIP}, ::Int 
) where {BEL<:BetheEnergyLoss, MIP<:NeXLMeanIonizationPotential}
    ΔE = dEds(BEL, energy(p), mat, MIP) * 𝜆
    alter_energy!(p, ΔE)
end
function alter_energy!(
    p::MutableParticle, 𝜆::Real, mat::ParametricMaterial, ::Type{BEL}, ::Type{MIP}, nquad::Int
) where {BEL<:BetheEnergyLoss, MIP<:NeXLMeanIonizationPotential}
    pos, dir, e = current(p), direction(p), energy(p)
    newpos = similar(pos)
    quad = quadrature(nquad)
    ΔE = quad(zero(𝜆), 𝜆) do l
        newpos .= pos .+ l .* dir
        dEds!(BEL, e, mat, newpos, MIP)
    end
    alter_energy!(p, ΔE)
end

function scatter!(p::MutableParticle{T}, mat::AbstractMaterial, ::Type{ECS}) where {T, ECS<:ElasticScatteringCrossSection}
    (𝜃, 𝜙) = scatter(ECS, mat, energy(p), T)
    deflect!(p, 𝜃, 𝜙)
end
function scatter!(p::MutableParticle{T}, mat::ParametricMaterial, ::Type{ECS}) where {T, ECS<:ElasticScatteringCrossSection}
    (𝜃, 𝜙) = scatter!(ECS, mat, energy(p), current(p), T)
    deflect!(p, 𝜃, 𝜙)
end

function transporter(;
    ecs::Type{<:ElasticScatteringCrossSection}=ScreenedRutherford,
    bel::Type{<:BetheEnergyLoss}=JoyLuo,
    mip::Type{<:NeXLMeanIonizationPotential}=Berger1982,
    rtol::Real=0.01,
    maxiter::Integer=10,
    nquad::Integer=5,
)
    return (p::MutableParticle, reg::AbstractRegion) -> transport(p, reg, ecs, bel, mip, rtol, maxiter, nquad) 
end

function trajectory2(
    eval::Function,
    p::MutableParticle,
    reg::AbstractRegion,
    terminate::Function,
    scf::Function=transporter(),
)
    (pc, nextreg) = (p, childmost_region(reg, position(p)))
    while !(terminate(pc, reg) || isnothing(nextreg))
        prevreg = nextreg
        (pc, nextreg) = scf(pc, nextreg)
        eval(pc, prevreg)
    end
end

function trajectory2(
    eval::Function,
    p::MutableParticle,
    reg::AbstractRegion,
    Emin::Real=0.1,
    scf::Function=transporter(),
)
    terminate(p::Particle, ::AbstractRegion) = energy(p) < Emin
    trajectory2(eval, p, reg, terminate, scf)
end

abstract type AbstractVoxelRegion{DIM, P, M} <: AbstractRegion{M} end

struct VoxelisedRegion{DIM, T, P} <: AbstractVoxelRegion{DIM, P, Nothing}
    shape::Rect3{T}
    parent::P
    children::Array{AbstractVoxelRegion, 3}
    nodes::Vector{Vector{T}}
    name::String
    voxel_sizes::NTuple{3, T}

    function VoxelisedRegion(
        sh::Rect3{T},
        parent::Union{Nothing,AbstractRegion},
        num_voxels::Union{NTuple{3, <:Integer}, AbstractVector{<:Integer}},
        name::Union{Nothing,String} = nothing,
    ) where {T<:AbstractFloat}
        maxsize = Float64(Sys.total_memory()) / 200
        if prod(num_voxels) > maxsize
            error("Voxels are too big for memory")
        end
        name = something(
            name,
            isnothing(parent) ? "Root" : "$(parent.name)[$(length(parent.children)+1)]",
        )
        voxel_sizes = (sh.widths[1] / num_voxels[1], sh.widths[2] / num_voxels[2], sh.widths[3] / num_voxels[3])

        #nodes = [(sh.origin[1] + i * voxel_sizes[1], 
        #sh.origin[2] + j * voxel_sizes[2], 
        #sh.origin[3] + k * voxel_sizes[3] ) for i in 0:num_voxels[1], j in 0:num_voxels[2], k in 0:num_voxels[3]]
        nodes = [sh.origin[i] .+ collect(0:num_voxels[i]) .* voxel_sizes[i] for i in 1:3]
        
        (N1, N2, N3) = num_voxels
        voxels = Array{AbstractVoxelRegion}(undef, N1, N2, N3)
        res = new{Tuple{N1, N2, N3}, T, typeof(parent)}(sh, parent, voxels, nodes, name, voxel_sizes)
        
        if !isnothing(parent)
            #= For nested Voxelized regions this could be a problem will have to come up with a more elegant solution
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

            # This cannot be guaranteed
            if !(parent isa VoxelisedRegion)
                @assert all(
                    ch -> all(!isinside(ch.shape, v) for v in vertices),
                    parent.children,
                ) "The child $sh overlaps a child of the parent shape."
            end =#

            push!(parent.children, res)
        end
        return res
    end
end

function VoxelisedRegion(
    sh::Rect3{T},
    mat_func::Function,
    parent::Union{Nothing,AbstractRegion},
    num_voxels::Union{NTuple{3, <:Integer}, AbstractVector{<:Integer}};
    name::Union{Nothing,String} = nothing,
) where {T<:AbstractFloat}
    res = VoxelisedRegion(sh, parent, num_voxels, name)
    for (i, j, k) in eachindex(res)
        re.children[i, j, k] = Voxel((i, j, k), mat_func(centroid(res, i, j, k)), res)
    end
    return res
end

function VoxelisedRegion(
    sh::Rect3{T},
    mat_temp::MaterialTemplate,
    parent::Union{Nothing,AbstractRegion},
    num_voxels::Union{NTuple{3, <:Integer}, AbstractVector{<:Integer}};
    name::Union{Nothing,String}=nothing,
    massfrac_type::Type{<:AbstractFloat}=Float64,
    static::Bool=true,
    kwargs...
) where {T<:AbstractFloat}
    res = VoxelisedRegion(sh, parent, num_voxels, name)
    for (i, j, k) in eachindex(res)
        pos = centroid(res, i, j, k)
        massfracs = MVector{length(mat_temp), massfrac_type} |> zero
        massfracfunc!(mat_temp)(massfracs, pos)
        ρ = densityfunc(mat_temp)(massfracs, pos)
        if static
            mat = STemplateMaterial(mat_temp, massfracs, ρ)
        else
            mat = MTemplateMaterial(mat_temp, massfracs, ρ; kwargs...)
        end
        res.children[i, j, k] = Voxel((i, j, k), mat, res)
    end
    return res
end

# For Future implementation of Octal Regions
const OctalRegion{T, P} = VoxelisedRegion{Tuple{2, 2, 2}, T, P}

Base.size(::VoxelisedRegion{Tuple{N1, N2, N3}}) where {N1, N2, N3} = (N1, N2, N3)
Base.eachindex(::VoxelisedRegion{Tuple{N1, N2, N3}}) where {N1, N2, N3} = begin
    ((i1, i2, i3) for i3 in 1:N3 for i2 in 1:N2 for i1 in 1:N1)
end
Base.length(vr::VoxelisedRegion) = prod(size(vr))

function Base.show(io::IO, ::MIME"text/plain", vr::VoxelisedRegion{<:Any, T, P}) where {T, P}
    print(io, "$(typeof(vr)):")
    print(io, "\n name: ", name(vr))
    print(io, "\n shape: Rect(origin=$(Tuple(origin(shape(vr)))), widths=$(Tuple(widths(shape(vr)))))")
    print(io, "\n voxel_size: $(vr.voxel_sizes)")
    print(io, "\n parent: ", split("$P", "{")[1], "(name=$(name(parent(vr))))")
end
function Base.show(io::IO, vr::VoxelisedRegion)
    print(io, "$(typeof(vr))[", name(vr), ", origin=$(origin(shape(vr))), voxel_sizes=$(vr.voxel_sizes)]")
end

material(::VoxelisedRegion) = error("VoxelisedRegion doesn't have a material")

function node(vr::VoxelisedRegion{<:Any, T, <:Any}, i::Integer, j::Integer, k::Integer)  where T
    SVector{3, T}(vr.nodes[1][i], vr.nodes[2][j], vr.nodes[3][k])
end

function corner(
    vr::VoxelisedRegion{<:Any, T, <:Any}, i::Integer, j::Integer, k::Integer, cornerid::Union{Tuple, AbstractArray}
) where T
    sz = vr.voxel_sizes
    @assert length(cornerid) == 3 "length of cornerid is not 3"
    tmp(a::Integer, b::Integer) = vr.nodes[b][a] + cornerid[b]*sz[b]
    SVector{3, T}(tmp(i ,1), tmp(j, 2), tmp(k, 3))
end

centroid(vr::VoxelisedRegion, i::Integer, j::Integer, k::Integer) = corner(vr, i, j, k, (0.5, 0.5, 0.5))

struct Voxel{P, M<:AbstractMaterial} <: AbstractVoxelRegion{Tuple{1, 1, 1}, P, M}
    index::NTuple
    material::M
    parent::P
end

# Don't have to check the material perhaps
Base.:(==)(vx1::T, vx2::T) where {T<:Voxel} = vx1.index == vx2.index && vx1.parent == vx2.parent

function Base.show(io::IO, ::MIME"text/plain", vx::Voxel{<:Any, M}) where M
    print(io, "$(name(vx)):\n")
    print(io, "mat_type: $M\n")
    print(io, "shape: Rect(origin=$(Tuple(minimum(vx))), widths=$(Tuple(widths(vx))))\n")
    print(io, "parent_size: $(size(parent(vx)))")
end
function Base.show(io::IO, vx::Voxel)
    print(io, "$(typeof(vx))[index= $(Tuple(vx.index)), origin=$(Tuple(minimum(vx))), voxel_sizes=$(Tuple(widths(vx)))]")
end

i1(vx::Voxel) = @inbounds vx.index[1]
i2(vx::Voxel) = @inbounds vx.index[2]
i3(vx::Voxel) = @inbounds vx.index[3]
shape(reg::Voxel) = reg
name(vx::Voxel) = "Voxel[$(i1(vx)), $(i2(vx)), $(i3(vx))] of $(name(parent(vx)))"
children(::Voxel) = []
haschildren(::Voxel) = false

function rect(vx::Voxel)
    RectangularShape(node(vx.parent, i1(vx), i2(vx), i3(vx)), vx.parent.voxel_sizes)
end

Base.maximum(vx::Voxel) = node(vx.parent, i1(vx)+1, i2(vx)+1, i3(vx)+1)
Base.minimum(vx::Voxel) = node(vx.parent, i1(vx), i2(vx), i3(vx))
GeometryBasics.origin(vx::Voxel) = minimum(vx)
GeometryBasics.widths(vx::Voxel) = parent(vx).voxel_sizes

function isinside(vx::Voxel, pos::AbstractVector{<:Real})
    nodes = vx.parent.nodes
    index = vx.index
    tmp(i) = @inbounds nodes[i][index[i]] ≤ pos[i] ≤ nodes[i][index[i]+1]
    return tmp(1) && tmp(2) && tmp(3)
end

function intersection(reg::VoxelisedRegion, pos1::AbstractArray{<:Real}, pos2::AbstractArray{<:Real})
    if isinside(reg, pos1) && haschildren(reg)
        return intersection_inside(childmost_region(reg, pos1), pos1, pos2)
    end
    return intersection(shape(reg), pos1, pos2)
end

function intersection_inside( # how to make this more efficient? 
    vx::Voxel,
    pos1::AbstractArray{T},
    pos2::AbstractArray{T},
) where {T<:AbstractFloat}
    t::T = typemax(T)
    nodes = vx.parent.nodes
    for (i, n) in enumerate(vx.index)
        v = pos2[i] - pos1[i]
        if v != 0
            t1 = (nodes[i][n] - pos1[i]) / v
            t2 = (nodes[i][n+1] - pos1[i]) / v
            #println((t1, t2, v, pos1[i], nodes[i][n], nodes[i][n]))
            if (t1 > 0.0) && (t1 < t)
                t = t1
            end
            if (t2 > 0.0) && (t2 < t)
                t = t2
            end
        end
    end
    return t
end

function intersection( # how to make this more efficient? 
    vx::Voxel,
    pos1::StaticVector{3, T},
    pos2::StaticVector{3, T},
) where T
    if isinside(vx, pos1)
        return intersection_inside(vx, pos1, pos2)
    end
    @debug "$(pos1) not in the voxel with bounds $(minimum(vx)) and $(maximum(vx))"
    if isinside(parent(vx), pos1)
        vx2 = childmost_region(parent(vx), pos1)
        if vx2 == vx
            # This could mean we're running into floating point precision issues.
            return zero(T)
        end
        return intersection(vx2, pos1, pos2)
    end
    # This should typically not happen, but could mean pos1 is on the boundary of the parent.
    @debug pos1, minimum(vx), maximum(vx), minimum(parent(vx)), maximum(parent(vx))
    return intersection(shape(parent(vx)), pos1, pos2)
end

# Note: This function assumes pos is inside region
function childmost_region(reg::VoxelisedRegion{Tuple{Nx, Ny, Nz}}, pos::AbstractArray{<:Real}) where {Nx, Ny, Nz}
    o = origin(shape(reg))
    idx(i) = ceil.(Int, (pos[i] - o[i]) / reg.voxel_sizes[i])
    ix = min(max(1, idx(1)), Nx)
    iy = min(max(1, idx(2)), Ny)
    iz = min(max(1, idx(3)), Nz)
    return childmost_region(reg.children[ix, iy, iz], pos)
end

# Note: This function assumes pos is inside region
childmost_region(reg::Voxel, ::AbstractArray{<:Real}) = reg

function find_region(pos::AbstractArray{<:Real}, reg::AbstractRegion)
    if isinside(shape(reg), pos)
        return childmost_region(reg, pos)
    else
        return find_region(pos, parent(reg))
    end
end

find_region(::AbstractArray, ::Nothing) = nothing

find_region(pos::AbstractArray{<:Real}, reg::Voxel) = find_region(pos, parent(reg))
