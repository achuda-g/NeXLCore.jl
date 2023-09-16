using StaticArrays

# Utils
function volume_conserving_density(elms::AbstractArray{Element})
    N = length(elms)
    ρ_pure = SVector{N, Float64}(density.(elms))
    function _density(c::AbstractArray{<:Real}, ::Any)
        if length(c) != N
            error("Argument c has the wrong size")
        end
        temp = sum(i -> c[i] / ρ_pure[i], eachindex(ρ_pure))
        return 1. / temp
    end
    return _density
end


"""
Abstract type for defining Materials which Hold basic data about a material in Vector-like form for fast arithmetic.

It is a parametric subtype of `AbstractMaterial{U, V}`. `N` is the number of elements. `W` is the element type of atomic fractions.
In addition to `name`, and `properties` as in `AbstractMaterial`, all subtypes must contain the following fields or methods.
    +`elms:AbstractVector{Element}` or `elms_vec(mat)` - vector of elements.
    +`massfracs:AbstractVector{U}` or `massfracs(mat)` - vector of mass fractions in the same order as `elms`.
    +`a:AbstractVector{V<:AbstractFloat}` or `atomicmasses(mat)` - vector of atomic numbers in the same order as `elms`.
    +`atomicfracs::AbstractVectror{W}` or `atomicfracs(mat)` - vector of atomic fractions in the same order as `elms`.
    +`elmdict:Dict{Element, Int}` or `elmdict(mat)` - a mapping from element to integer index of the ement in `elms`.
    +`properties[:Density]` or density(mat) - density of the material
    +`properties[:AtomsPerG] or atoms_per_g(mat) - no oof atoms in 1 g of the material`

Some functionalities only work when `elms`, `massfracs` and `a` have regular indexing between 0 and 1
"""
abstract type VectorizedMaterial{N, U, V, W} <: AbstractMaterial{U, V} end

Base.length(::VectorizedMaterial{N}) where N = N

elms(mat::VectorizedMaterial) = Iterators.Stateful(elms_vector(mat))

"""
    elms_vector(mat::VectorizedMaterial)

Elements in the material as a vector.
"""
elms_vector(mat::VectorizedMaterial) = mat.elms

"""
    massfracs(mat::VectorizedMaterial)

Mass fractions of all elements in the material as a vector.
"""
massfracs(mat::VectorizedMaterial) = mat.massfracs

"""
    atomicmasses(mat::VectorizedMaterial)

Atomic masses of all elements in the material as a vector.
"""
atomicmasses(mat::VectorizedMaterial) = mat.a

"""
    atomicfracs(mat::VectorizedMaterial)

Atomic fractions of all elments in the material as a vector
"""
atomicfracs(mat::VectorizedMaterial) = mat.atomicfracs

density(mat::VectorizedMaterial) = mat[:Density]

elmdict(mat::VectorizedMaterial) = mat.elmdict


Base.getindex(mat::VectorizedMaterial{<:Any, U}, elm::Element) where U =
    get(mat.massfracs, get(elmdict(mat), elm, 0), zero(U))
Base.getindex(mat::VectorizedMaterial{<:Any, U}, z::Int) where U =
    get(mat, elements[z], zero(U))

Base.eachindex(mat::VectorizedMaterial) = eachindex(elms_vector(mat))

a(elm::Element, mat::VectorizedMaterial) = get(mat.a, get(elmdict(mat), elm, 0), a(elm))

# A set of functions that work with indices
elm(mat::VectorizedMaterial, index::Integer) = getindex(elms_vector(mat), index)
# Same as elm but doesn't check if index in bounds. Can be unsafe. Use only with eachindex.
elm_nocheck(mat::VectorizedMaterial, index::Integer) = @inbounds elms_vector(mat)[index]
massfrac(mat::VectorizedMaterial{<:Any, U}, index::Integer) where U = get(massfracs(mat), index, zero(U))
atomicmass(mat::VectorizedMaterial, index::Integer) = getindex(atomicmasses(mat), index)
atomicfrac(mat::VectorizedMaterial{<:Any, <:Any, <:Any, W}, index::Integer) where W = get(atomicfracs(mat), index, zero(W))
atomicfrac(mat::VectorizedMaterial, elm::Element) = atomicfrac(mat, get(elmdict(mat), elm, 0))

_atoms_per_g(mat::VectorizedMaterial, index::Integer) =  massfrac(mat, index) / atomicmass(mat, index) * _AVAGADRO

function _atomicfracs_atoms_per_g!(
    atomicfracs::AbstractArray{<:AbstractFloat}, atomicmasses::AbstractArray{<:Real}, massfracs::AbstractArray{<:Real}
)
    atomicfracs .= massfracs ./ atomicmasses
    tmp = sum(atomicfracs)
    if tmp == zero(tmp)
        @warn "in `_atomicfracs_atoms_per_g!`: massfracs are zero?"
    end
    atomicfracs ./= tmp
    return tmp * _AVAGADRO
end
function _atomicfracs_atoms_per_g(atomicmasses::AbstractArray{T1}, massfracs::AbstractArray{T2}) where {T1, T2}
    T = promote_type(T1, T2)
    atomicfracs = zero(MVector{length(atomicmasses), T})
    atoms_per_g = _atomicfracs_atoms_per_g!(atomicfracs, atomicmasses, massfracs)
    return atomicfracs, atoms_per_g
end

atoms_per_g(mat::VectorizedMaterial) = mat[:AtomsPerG]
atoms_per_g(mat::VectorizedMaterial, elm::Element) = atomicfrac(mat, elm) * atoms_per_g(mat)
atoms_per_g(mat::VectorizedMaterial, index::Integer) = atomicfrac(mat, index) * atoms_per_g(mat)

atoms_per_cm³(mat::VectorizedMaterial, index::Integer) = atoms_per_g(mat, index) * density(mat)

"""
    atoms_per_g_all(mat::VectorizedMaterial)

The number of atoms of each element in 1 g of the material as a vector.
"""
atoms_per_g_all(mat::VectorizedMaterial) = [atoms_per_g_inbounds(mat, i) for i in eachindex(mat)]

"""
    atoms_per_cm³_all(mat::VectorizedMaterial)

The number of atoms of each element in 1 cm³ of the material as a vector.
"""
atoms_per_cm³_all(mat::VectorizedMaterial) = atoms_per_g_all(mat) .* density(mat)

function mac(mat::VectorizedMaterial, energy::Real, alg::Type{<:NeXLAlgorithm}=DefaultAlgorithm)
    return sum(eachindex(mat)) do i
        mac(elm(mat, i), energy, alg) * massfrac(mat, i)
    end
end
function mac(mat::VectorizedMaterial, xray::CharXRay, alg::Type{<:NeXLAlgorithm}=DefaultAlgorithm)
    return sum(eachindex(mat)) do i
        mac(elm(mat, i), xray, alg) * massfrac(mat, i)
    end
end

struct MaterialTemplate{N, V<:AbstractFloat, M, D} <: VectorizedMaterial{N, Nothing, V, Nothing}
    name::String
    elms::SVector{N, Element}
    a::SVector{N, V}
    elmdict::Dict{Element, Int}
    properties::Dict{Symbol, Any}
    massfracfunc!::M
    densityfunc::D
end

"""
    function MaterialTemplate(
        name::AbstractString,
        elms::AbstractArray{Element},
        atomicmasses::Union{AbstractArray, Nothing}=nothing,
        properties::Union{AbstractDict{Symbol,Any}, Nothing}=nothing,
        massfracfunc!::M = nothing,
        densityfunc::D = nothing,
    ) where {M, D}

A template material with the given template, mass fractions and density.
"""
function MaterialTemplate(
    name::AbstractString,
    elms::AbstractArray{Element},
    atomicmasses::Union{AbstractArray, Nothing}=nothing,
    properties::Union{AbstractDict{Symbol,Any}, Nothing}=nothing,
    massfracfunc!::M = nothing,
    densityfunc::D = nothing,
) where {M, D}
    N = length(elms)
    elms = SVector{N, Element}(elms)
    if isnothing(atomicmasses)
        atomicmasses = a.(elms)
    elseif length(atomicmasses) != N
        error("size of atomicmasses doesn't match elms")
    end
    if isnothing(properties)
        properties = Dict{Symbol, Any}()
    end
    V = eltype(atomicmasses)
    atomicmasses = SVector{N, V}(atomicmasses)
    elmdict = Dict((elms[i] => i for i in eachindex(elms)))
    MaterialTemplate(name, elms, atomicmasses, elmdict, properties, massfracfunc!, densityfunc)
end

_mat_temp_call_error() = error("This function shouldn't be called on an instance of `MaterialTemplate`")

massfracs(::MaterialTemplate) = _mat_temp_call_error()
atomicfracs(::MaterialTemplate) = _mat_temp_call_error()
density(::MaterialTemplate) = _mat_temp_call_error()
atoms_per_g(::MaterialTemplate) = _mat_temp_call_error()
Base.getindex(::MaterialTemplate, ::Element) = _mat_temp_call_error()
Base.getindex(::MaterialTemplate, ::Integer) = _mat_temp_call_error()
massfracfunc!(mat::MaterialTemplate) = mat.massfracfunc!
massfracfunc!(::MaterialTemplate{<:Any, <:Any, Nothing}) = error("massfracfunc! not defined in this template")
densityfunc(mat::MaterialTemplate) = mat.densityfunc
densityfunc(::MaterialTemplate{<:Any, <:Any, <:Any, Nothing}) = error("massfracfunc! not defined in this template")
Base.show(io::IO, mat::MaterialTemplate) = print(io, "Material Template")

# Note: this may not be a perfect copy
function Base.copy(mat::MaterialTemplate)
    MaterialTemplate(
        mat.name, mat.elms, mat.atomicmasses, copy(mat.elmdict), copy(properties), massfracfunc!, densityfunc
    )
end

abstract type TemplateMaterial{N, U, V, W} <: VectorizedMaterial{N, U, V, W} end

template(mat::TemplateMaterial) = mat.template
name(mat::TemplateMaterial) = mat |> template |> name
properties(mat::TemplateMaterial) = mat |> template |> properties
elms_vector(mat::TemplateMaterial) = mat |> template |> elms_vector
atomicmasses(mat::TemplateMaterial) = mat |> template |> atomicmasses
elmdict(mat::TemplateMaterial) = mat |> template |> elmdict
massfracfunc!(mat::TemplateMaterial) = mat |> template |> massfracfunc!
densityfunc(mat::TemplateMaterial) = mat |> template |> densityfunc

struct STemplateMaterial{N, U<:AbstractFloat, V, W<:AbstractFloat} <: TemplateMaterial{N, U, V, W}
    template::MaterialTemplate{N, V}
    massfracs::SVector{N, U}
    atomicfracs::SVector{N, W}
    density::AbstractFloat
    atoms_per_g::AbstractFloat
end


"""
    STemplateMaterial(template::MaterialTemplate{N, V}, massfracs::AbstractVector{U}, density::AbstractFloat) where {N, U, V}
    STemplateMaterial(mat::TemplateMaterial, massfracs::AbstractVector, density::AbstractFloat)

A template material with the given template, mass fractions, atomic fractions, density and atoms/g.
"""
function STemplateMaterial(template::MaterialTemplate{N, V}, massfracs::AbstractVector{U}, density::AbstractFloat) where {N, U, V}
    if length(massfracs) != N
        error("massfracs has a wrong length")
    end
    massfracs = SVector{N, U}(massfracs)
    atomicfracs, atoms_per_g = _atomicfracs_atoms_per_g(atomicmasses(template), massfracs)
    STemplateMaterial(template, massfracs, SVector(atomicfracs), density, atoms_per_g)
end
function STemplateMaterial(mat::TemplateMaterial, massfracs::AbstractVector, density::AbstractFloat)
    STemplateMaterial(template(mat), massfracs, density)
end

function Base.copy(mat::STemplateMaterial, copy_template::Bool=false)
    _template = copy_template ? copy(template(mat)) : template(mat)
    STemplateMaterial(_template, mat.massfracs, mat.atomicfracs, mat.density, mat.atoms_per_g)
end

density(mat::STemplateMaterial) = mat.density
atoms_per_g(mat::STemplateMaterial) = mat.atoms_per_g

function Base.getindex(mat::STemplateMaterial, sym::Symbol)
    if sym == :Density
        return density(mat)
    elseif sym == :AtomsPerG
        return atoms_per_g(mat)
    end
    getindex(properties(mat), sym)
end

abstract type MTemplateMaterial{N, U, V, W, A} <: TemplateMaterial{N, U, V, W} end
const ParametricMaterial{N, U, V, W} = MTemplateMaterial{N, U, V, W, true}

# Parametric Material
struct MTemplateMaterialSingle{N, U<:AbstractFloat, V<:AbstractFloat, W<:AbstractFloat, A} <: MTemplateMaterial{N, U, V, W, A}
    template::MaterialTemplate{N, V}
    massfracs::MVector{N, U}
    atomicfracs::MVector{N, W}
    properties::Dict{Symbol, Any}

    function MTemplateMaterialSingle{A}(
        template::MaterialTemplate{N, V}, massfracs::MVector{N, U}, atomicfracs::MVector{N, W}, 
        properties::Dict{Symbol, Any}
    ) where {A, N, U, V, W}
        new{N, U, V, W, A}(template, massfracs, atomicfracs, properties)
    end
end

"""
    MTemplateMaterial(template::MaterialTemplate{N, V}, massfracs::AbstractVector{U}, density::AbstractFloat) where {N, U, V}
    MTemplateMaterial(mat::TemplateMaterial, massfracs::AbstractVector, density::AbstractFloat)

A template material with the given template, mass fractions, atomic fractions, density and atoms/g.
"""
function MTemplateMaterialSingle(
    template::MaterialTemplate{N, V}, massfracs::AbstractArray{U}, density::AbstractFloat, autoupdate::Bool=false,
    lastpos::Union{Nothing, AbstractArray}=nothing,
) where {N, U, V}
    if length(massfracs) != N
        error("massfracs has a wrong length")
    end
    massfracs = MVector{N, U}(massfracs)
    atomicfracs, atoms_per_g = _atomicfracs_atoms_per_g(atomicmasses(template), massfracs)
    properties = Dict{Symbol, AbstractFloat}()
    properties[:Density] = density
    properties[:AtomsPerG] = atoms_per_g
    properties[:LastPos] = lastpos
    MTemplateMaterialSingle{autoupdate}(template, massfracs, atomicfracs, properties)
end
function MTemplateMaterialSingle(
    mat::TemplateMaterial, massfracs::AbstractVector, density::AbstractFloat, autoupdate::Bool=false,
    lastpos::Union{Nothing, AbstractArray}=nothing
)
    MTemplateMaterialSingle(template(mat), massfracs, density, autoupdate, lastpos)
end

_properties(mat::MTemplateMaterialSingle) = mat.properties
function Base.copy(mat::MTemplateMaterialSingle{<:Any, <:Any, <:Any,<:Any, A}, copy_template=false) where A
    _template = copy_template ? copy(template(mat)) : template(mat)
    MTemplateMaterialSingle{A}(_template, copy(mat.massfracs), copy(mat.atomicfracs), copy(mat.properties))
end

struct MTemplateMaterialThreaded{N, U, V, W, A} <: MTemplateMaterial{N, U, V, W, A}
    buffer::DefaultDict{Integer, MTemplateMaterialSingle{N, U, V, W, A}}

    function MTemplateMaterialThreaded(default::MTemplateMaterialSingle{N, U, V, W, A}) where {A, N, U, V, W}
        instances = DefaultDict{Integer, MTemplateMaterialSingle{N, U, V, W, A}}() do
            copy(default, false)
        end
        new{N, U, V, W, A}(instances)
    end
end

"""
    MTemplateMaterial(template::MaterialTemplate{N, V}, massfracs::AbstractVector{U}, density::AbstractFloat) where {N, U, V}
    MTemplateMaterial(mat::TemplateMaterial, massfracs::AbstractVector, density::AbstractFloat)

A template material with the given template, mass fractions, atomic fractions, density and atoms/g.
"""
function MTemplateMaterial(
    template::MaterialTemplate{N, V}, massfracs::AbstractArray{U}, density::AbstractFloat, autoupdate::Bool=false,
    lastpos::Union{Nothing, AbstractArray}=nothing, threaded::Bool=false,
) where {N, U, V}
    res = MTemplateMaterialSingle(template, massfracs, density, autoupdate, lastpos)
    if threaded
        return MTemplateMaterialThreaded(res)
    end
    return res
end

instance(mat::MTemplateMaterialThreaded) = mat.buffer[Threads.threadid()]
template(mat::MTemplateMaterialThreaded) = mat |> instance |> template
massfracs(mat::MTemplateMaterialThreaded) = mat |> instance |> massfracs
atomicfracs(mat::MTemplateMaterialThreaded) = mat |> instance |> atomicfracs
_properties(mat::MTemplateMaterialThreaded) = mat |> instance |> _properties

Base.copy(mat::MTemplateMaterialThreaded, copy_template=true) = MTemplateMaterialThreaded(copy(instance(mat), copy_template))

properties(mat::MTemplateMaterial) = merge(template(mat), _properties(mat))
Base.getindex(mat::MTemplateMaterial, sym::Symbol) = get(_properties(mat), sym, getindex(properties(template(mat)), sym))
Base.get(mat::MTemplateMaterial, sym::Symbol, def) = get(_properties(mat), sym,  get(properties(template(mat)), sym, def))
Base.setindex!(mat::MTemplateMaterial, val, sym::Symbol) = setindex!(_properties(mat), val, sym)

function update!(mat::MTemplateMaterial, x::AbstractArray{<:Real})
    _massfracs = massfracs(mat)
    massfracfunc!(mat)(_massfracs, x)
    mat[:Density] = densityfunc(mat)(mat, _massfracs, x)
    mat[:AtomsPerG] = _atomicfracs_atoms_per_g!(atomicfracs(mat), atomicmasses(mat), _massfracs)
    mat[:LastPos] = x
end

"""
    MTemplateMaterial(
        name::AbstractString,
        elms::AbstractVector{Element},
        massfracfunc!::Any,
        atomicmasses::Union{AbstractArray, Nothing}=nothing,
        densityfunc::Any=nothing,
        properties::Union{AbstractDict{Symbol, Any}, Nothing}=nothing,
        massfractype::Type{<:AbstractFloat}=Float64,
        threaded::Bool=true,
        autoupdate::Bool=true,
    )


A material whose composition and density can be functions of the position in space. Ideally the material must be defined for all points 
in space for correct functionality of all methods. Mass fractions and density are cached in the struct but need to be updated for each 
new position. The last position where material properties were updated is also cached in `properties[:LastPos]`

**Properties**

- `:Density` # Density in g/cm³
- `:Description` # Human friendly
- `:Pedigree` #  Quality indicator for compositional data ("SRM-XXX", "CRM-XXX", "NIST K-Glass", "Stoichiometry", "Wet-chemistry by ???", "WDS by ???", "???")
- `:Conductivity` = :Insulator | :Semiconductor | :Conductor
- `:OtherUserProperties` # Other properties can be defined as needed
"""
function MTemplateMaterial(
    name::AbstractString,
    elms::AbstractVector{Element},
    massfracfunc!::Any,
    atomicmasses::Union{AbstractArray, Nothing}=nothing,
    densityfunc::Any=nothing,
    properties::Union{AbstractDict{Symbol, Any}, Nothing}=nothing,
    massfractype::Type{<:AbstractFloat}=Float64,
    threaded::Bool=true,
    autoupdate::Bool=false,
)
    if isnothing(densityfunc)
        densityfunc = volume_conserving_density(elms)
    end
    template = MaterialTemplate(name, elms, atomicmasses, properties, massfracfunc!, densityfunc)
    N = length(template)
    U = massfractype
    massfracs = zero(MVector{N, U})
    x = zeros(3)
    try
        massfracfunc!(massfracs, x)
    catch ex
        error("masfracfunc! is not a suitable function, Please check the source error: $(ex.msg)")
    end
    if isnothing(densityfunc)
        densityfunc = volume_conserving_density(elms)
    end
    density = nothing
    try
        density = densityfunc(massfracs)
    catch ex
        error("densityfunc is not a suitable function, Please check the source error: $(ex.msg)")
    end
    if ~(density isa Real)
        error("densityfunc must return a scalar")
    end
    result = MTemplateMaterial(template, massfracs, density, autoupdate, x)
    if threaded
        return MTemplateMaterialThreaded(result)
    end
    return result
end

"""
    ParametricMaterial(
        name::AbstractString,
        elms::AbstractVector{Element},
        massfracfunc!::Any,
        atomicmasses::Union{AbstractArray, Nothing}=nothing,
        densityfunc::Any=nothing,
        properties::Union{AbstractDict{Symbol, Any}, Nothing}=nothing,
        massfractype::Type{<:AbstractFloat}=Float64,
        threaded::Bool=true,
    )


A material whose composition and density can be functions of the position in space. Ideally the material must be defined for all points 
in space for correct functionality of all methods. Mass fractions and density are cached in the struct but need to be updated for each 
new position. The last position where material properties were updated is also cached in `properties[:LastPos]`

**Properties**

- `:Density` # Density in g/cm³
- `:Description` # Human friendly
- `:Pedigree` #  Quality indicator for compositional data ("SRM-XXX", "CRM-XXX", "NIST K-Glass", "Stoichiometry", "Wet-chemistry by ???", "WDS by ???", "???")
- `:Conductivity` = :Insulator | :Semiconductor | :Conductor
- `:OtherUserProperties` # Other properties can be defined as needed
"""
function ParametricMaterial(
    name::AbstractString,
    elms::AbstractVector{Element},
    massfracfunc!::Any,
    atomicmasses::Union{AbstractArray, Nothing}=nothing,
    densityfunc::Any=nothing,
    properties::Union{AbstractDict{Symbol, Any}, Nothing}=nothing,
    massfractype::Type{<:AbstractFloat}=Float64,
    threaded::Bool=true,
)
    MTemplateMaterial(name, elms, massfracfunc!, atomicmasses, densityfunc, properties, massfractype, threaded, true)
end

function Base.copy(mat::ParametricMaterial{<:Any, U}) where U
    return ParametricMaterial{U}(mat.name, mat.elms, mat.massfracfunc!, mat.a, mat.densityfunc, mat.properties)
end

function massfractions!(mat::ParametricMaterial, massfracs::AbstractVector{<:Real}, x::AbstractVector{<:Real})
    return mat.massfracfunc!(massfracs, x, mat.elms)
end

function massfractions(mat::ParametricMaterial, x::AbstractVector{<:Real})
    massfracs = similar(mat.massfracs)
    return mat.massfracfunc!(massfracs, x, mat.elms)
end

function density(mat::ParametricMaterial, x::AbstractVector{<:Real})
    return mat.densityfunc(x, mat.elms)
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


