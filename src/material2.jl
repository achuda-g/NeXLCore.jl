using StaticArrays
using GeometryBasics: Point3

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

cpu_current_task_id() = objectid(current_task())

"""
    VectorizedMaterial{N, T} <: AbstractMaterial{T, T}

Abstract type for defining Materials which Hold basic data about a material in Vector-like form for fast arithmetic.
It is a parametric subtype of `AbstractMaterial{T, T}`. `N` is the number of elements. `T` is the type for all floating point fields.
In addition to `name`, and `properties` as in `AbstractMaterial`, all subtypes must contain the following fields or methods.
    +`elms:AbstractVector{Element}` or `elms_vec(mat)` - vector of elements.
    +`massfracs:AbstractVector{T}` or `massfracs(mat)` - vector of mass fractions in the same order as `elms`.
    +`a:AbstractVector{T}` or `atomicmasses(mat)` - vector of atomic numbers in the same order as `elms`.
    +`atomicfracs::AbstractVectror{T}` or `atomicfracs(mat)` - vector of atomic fractions in the same order as `elms`.
    +`elmdict:Dict{Element, Int}` or `elmdict(mat)` - a mapping from element to integer index of the ement in `elms`.
    +`density::T` or `density(mat)` - density of the material
    +`atoms_per_cm³::T` or `atoms_per_cm³(mat)` - no oof atoms in 1 g of the material`

Some functionalities only work when `elms`, `massfracs` and `a` have regular indexing, i.e. starting with `1` and with increments of `1` upto `N`.
"""
abstract type VectorizedMaterial{N, T} <: AbstractMaterial{T, T} end

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

density(mat::VectorizedMaterial) = mat.density

elmdict(mat::VectorizedMaterial) = mat.elmdict


function Base.getindex(mat::VectorizedMaterial{<:Any, T}, elm::Element)::T where T
    get(mat.massfracs, get(elmdict(mat), elm, 0), zero(U))
end
function Base.getindex(mat::VectorizedMaterial{<:Any, T}, z::Int)::T where T
    get(mat, elements[z], zero(U))
end
function Base.getindex(mat::VectorizedMaterial, sym::Symbol)
    if sym == :Density
        return density(mat)
    elseif sym == :AtomsPerG
        return atoms_per_g(mat)
    end
    getindex(properties(mat), sym)
end

Base.eachindex(mat::VectorizedMaterial) = eachindex(elms_vector(mat))

_show(io::IO, m::Any, mat) = print(io, typestr(mat, m), massfracstr(mat, m), densitystr(mat, m))
Base.show(io::IO, m::MIME"text/plain", mat::VectorizedMaterial) = _show(io, m, mat)
Base.show(io::IO, mat::VectorizedMaterial) = _show(io, nothing, mat)

typestr(mat::VectorizedMaterial, ::MIME"text/plain") = "$(length(mat))-element $(typeof(mat)):"
typestr(mat::VectorizedMaterial, ::Nothing) = split("$(typeof(mat))", "{")[1]

_repr(val::AbstractFloat) = "$val"
_repr(val::Union{Float64, Float32}) = @sprintf("%.6g", val)

function massfracstr(mat::VectorizedMaterial, ::MIME"text/plain")
    elms = sort(elms_vector(mat))
    res = "\nMass fractions:"
    for elm in elms
        res *= "\n\t$(symbol(elm))(Z=$(z(elm)), A=$(_repr(a(elm, mat)))) => $(_repr(massfrac(mat, elm)))"
    end
    return res
end
function massfracstr(mat::VectorizedMaterial, ::Nothing)
    elms = sort(elms_vector(mat))
    _str(elm) = "$(symbol(elm)) => $(_repr(massfrac(mat, elm)))"
    return "[" * join((_str(elm) for elm in elms), ", ") * "], "
end

densitystr(mat::VectorizedMaterial, ::MIME"text/plain") = "\nMass density: $(_repr(density(mat))) $(u"g/cm^3")"
densitystr(mat::VectorizedMaterial, ::Nothing) = "ρ = $(_repr(density(mat))) $(u"g/cm^3")}"

a(elm::Element, mat::VectorizedMaterial) = get(atomicmasses(mat), get(elmdict(mat), elm, 0), a(elm))

# A set of functions that work with indices
elm(mat::VectorizedMaterial, index::Integer) = getindex(elms_vector(mat), index)
# Same as elm but doesn't check if index in bounds. Can be unsafe. Use only with eachindex.
elm_nocheck(mat::VectorizedMaterial, index::Integer) = @inbounds elms_vector(mat)[index]
massfrac(mat::VectorizedMaterial{<:Any, T}, index::Integer) where T = get(massfracs(mat), index, zero(T))
massfrac(mat::VectorizedMaterial, elm::Element) = massfrac(mat, get(elmdict(mat), elm, 0))
atomicmass(mat::VectorizedMaterial, index::Integer) = getindex(atomicmasses(mat), index)
atomicfrac(mat::VectorizedMaterial{<:Any, T}, index::Integer) where T = get(atomicfracs(mat), index, zero(T))
atomicfrac(mat::VectorizedMaterial, elm::Element) = atomicfrac(mat, get(elmdict(mat), elm, 0))

function _atoms_per_g(mat::VectorizedMaterial{<:Any, T}, index::Integer)::T where T
    massfrac(mat, index) / atomicmass(mat, index) * _AVAGADRO(T)
end

function _atomicfracs_atoms_per_g!(
    atomicfracs::AbstractArray{T}, atomicmasses::AbstractArray{T}, massfracs::AbstractArray{T}
)::T where T
    atomicfracs .= massfracs ./ atomicmasses
    tmp = sum(atomicfracs)
    if tmp == zero(tmp)
        @warn "encountered divide by zero in `_atomicfracs_atoms_per_g!`"
    end
    atomicfracs ./= tmp
    return tmp * _AVAGADRO(T)
end
function _atomicfracs_atoms_per_g(atomicmasses::AbstractArray{T}, massfracs::AbstractArray{T}) where T
    atomicfracs = massfracs ./ atomicmasses
    tmp = sum(atomicfracs)
    if tmp == zero(tmp)
        @warn "encountered divide by zero in `_atomicfracs_atoms_per_g`"
    end
    return atomicfracs ./ tmp, tmp * _AVAGADRO(T)
end

atoms_per_g(mat::VectorizedMaterial) = mat.atoms_per_cm³ / density(mat)
atoms_per_g(mat::VectorizedMaterial, index::Union{Element, Integer}) = atomicfrac(mat, index) * atoms_per_g(mat)
atoms_per_cm³(mat::VectorizedMaterial) = mat.atoms_per_cm³
atoms_per_cm³(mat::VectorizedMaterial, index::Union{Element, Integer}) = atomicfrac(mat, index) * atoms_per_cm³(mat)

"""
    atoms_per_g_all(mat::VectorizedMaterial)

The number of atoms of each element in 1 g of the material as a vector.
"""
atoms_per_g_all(mat::VectorizedMaterial) = atomicfracs(mat) .* atoms_per_g(mat)

"""
    atoms_per_cm³_all(mat::VectorizedMaterial)

The number of atoms of each element in 1 cm³ of the material as a vector.
"""
atoms_per_cm³_all(mat::VectorizedMaterial) = atomicfracs(mat) .* atoms_per_cm³(mat)

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

struct MaterialTemplate{N, T<:AbstractFloat, M, D} <: VectorizedMaterial{N, T}
    name::String
    elms::SVector{N, Element}
    a::SVector{N, T}
    elmdict::Dict{Element, Int}
    properties::Dict{Symbol, Any}
    massfracfunc!::M
    densityfunc::D
end

"""
    function MaterialTemplate(
        ::Type{<:AbstractFloat},
        name::AbstractString,
        elms::AbstractArray{Element},
        atomicmasses::Union{AbstractArray, Nothing}=nothing,
        properties::Union{AbstractDict{Symbol,Any}, Nothing}=nothing,
        massfracfunc!::Any = nothing,
        densityfunc::Any = nothing,
    )
    function MaterialTemplate(
        name::AbstractString,
        elms::AbstractArray{Element},
        massfracfunc!::Any = nothing,
        densityfunc::Any = nothing,
        atomicmasses::Union{AbstractArray, Nothing}=nothing,
        properties::Union{AbstractDict{Symbol,Any}, Nothing}=nothing
    )

A template material with the given template, mass fractions and density.
"""
function MaterialTemplate(
    ::Type{T},
    name::AbstractString,
    elms::AbstractArray{Element},
    massfracfunc!::Any = nothing,
    densityfunc::Any = nothing,
    atomicmasses::Union{AbstractArray, Nothing}=nothing,
    properties::Union{AbstractDict{Symbol,Any}, Nothing}=nothing
) where {T<:AbstractFloat}
    N = length(elms)
    elms = SVector{N, Element}(elms)
    atomicmasses = something(atomicmasses, a.(elms))
    if length(atomicmasses) != N
        error("size of atomicmasses doesn't match elms")
    end
    properties = something(properties, Dict{Symbol, Any}())
    atomicmasses = SVector{N, T}(T.(atomicmasses))
    elmdict = Dict((elms[i] => i for i in eachindex(elms)))
    MaterialTemplate(String(name), elms, atomicmasses, elmdict, properties, massfracfunc!, densityfunc)
end
function MaterialTemplate(
    name::AbstractString,
    elms::AbstractArray{Element},
    massfracfunc!::Any = nothing,
    densityfunc::Any = nothing,
    atomicmasses::Union{AbstractArray, Nothing}=nothing,
    properties::Union{AbstractDict{Symbol,Any}, Nothing}=nothing
)
    MaterialTemplate(Float64,name, elms, massfracfunc!, densityfunc, atomicmasses, properties)
end

_mat_temp_call_error() = error("This function shouldn't be called on an instance of `MaterialTemplate`")

massfracs(::MaterialTemplate) = _mat_temp_call_error()
atomicfracs(::MaterialTemplate) = _mat_temp_call_error()
density(::MaterialTemplate) = _mat_temp_call_error()
atoms_per_cm³(::MaterialTemplate) = _mat_temp_call_error()
Base.getindex(::MaterialTemplate, ::Element) = _mat_temp_call_error()
Base.getindex(::MaterialTemplate, ::Integer) = _mat_temp_call_error()
massfracfunc!(mat::MaterialTemplate) = mat.massfracfunc!
densityfunc(mat::MaterialTemplate) = mat.densityfunc
function massfracstr(mat::MaterialTemplate, ::MIME"text/plain")
    elms = sort(elms_vector(mat))
    res = "\nElements:\n"
    for elm in elms
        res *= "\t$(symbol(elm))[Z=$(z(elm)), A=$(_repr(a(elm, mat)))]\n"
    end
    return res
end
function massfracstr(mat::MaterialTemplate, ::Any)
    elms = sort(elms_vector(mat))
    _str(elm) = "$(symbol(elm))"
    return "[" * join((_str(elm) for elm in elms), ", ") * "], "
end

densitystr(mat::MaterialTemplate, ::MIME"text/plain") = ""
densitystr(mat::MaterialTemplate, ::Any) = "}"

# Note: this may not be a perfect copy
function Base.copy(mat::MaterialTemplate)
    MaterialTemplate(
        mat.name, mat.elms, mat.atomicmasses, copy(mat.elmdict), copy(properties), massfracfunc!, densityfunc
    )
end

abstract type TemplateMaterial{N, T, AUTO} <: VectorizedMaterial{N, T} end
const ParametricMaterial{N, T} = TemplateMaterial{N, T, true}

template(mat::TemplateMaterial) = mat.template
name(mat::TemplateMaterial) = mat |> template |> name
properties(mat::TemplateMaterial) = mat |> template |> properties
elms_vector(mat::TemplateMaterial) = mat |> template |> elms_vector
atomicmasses(mat::TemplateMaterial) = mat |> template |> atomicmasses
elmdict(mat::TemplateMaterial) = mat |> template |> elmdict
massfracfunc!(mat::TemplateMaterial) = mat |> template |> massfracfunc!
densityfunc(mat::TemplateMaterial) = mat |> template |> densityfunc

evaluateat(eval, mat::TemplateMaterial, pos::AbstractVector) = eval(instance(mat, pos))

function Base.setindex!(mat::TemplateMaterial, val, sym::Symbol)
    setindex!(properties(mat), val, sym)
    @warn "Altered `properties` of template of $mat. This propogates to all materials using the same template"
end

struct STemplateMaterial{N, T<:AbstractFloat, M , D} <: TemplateMaterial{N, T, false}
    template::MaterialTemplate{N, T, M, D}
    massfracs::SVector{N, T}
    atomicfracs::SVector{N, T}
    density::T
    atoms_per_cm³::T
end


"""
    STemplateMaterial(template::MaterialTemplate{N, T}, massfracs::AbstractVector{T}, density::AbstractFloat) where {N, T}
    STemplateMaterial(mat::TemplateMaterial, massfracs::AbstractVector{T}, density::T) where {N, T}

A template material with the given template, mass fractions and density.
"""
function STemplateMaterial(template::MaterialTemplate{N, T}, massfracs::AbstractVector{T}, density::T) where {N, T}
    massfracs = SVector{N, T}(massfracs)
    atomicfracs, atoms_per_g = _atomicfracs_atoms_per_g(atomicmasses(template), massfracs)
    return STemplateMaterial(template, massfracs, SVector(atomicfracs), density, atoms_per_g * density)
end
function STemplateMaterial(mat::TemplateMaterial{N, T}, massfracs::AbstractVector{T}, density::T) where {N, T}
    return STemplateMaterial(template(mat), massfracs, density)
end
function STemplateMaterial(template, pos::AbstractVector)
    return instance(SParametricMaterial(template), pos)
end


function Base.copy(mat::STemplateMaterial, copy_template::Bool=false)
    _template = copy_template ? copy(template(mat)) : template(mat)
    return STemplateMaterial(_template, mat.massfracs, mat.atomicfracs, mat.density, mat.atoms_per_cm³)
end

struct SParametricMaterial{N, T<:AbstractFloat, M , D} <: TemplateMaterial{N, T, true}
    template::MaterialTemplate{N, T, M, D}

    """
        SParametricMaterial(temp::MaterialTemplate)

    A template material that doesn't cache any values but returns a `STemplateMaterial` when called with `instance`.
    """
    function SParametricMaterial(temp::MaterialTemplate{N, T, M, D}) where {N, T, M, D}
        testpos = zero(Point3{T})
        mfracs = massfracfunc!(temp)(T, testpos)
        if !(mfracs isa AbstractVector{T} && length(mfracs) == N)
            error("template must have a massfracfunc! that returns a subtype of AbstractVector of length $N")
        end
        if !(typeof(densityfunc(temp)(mfracs, testpos)) == T)
            error("template must have a densityfunc that returns a scalar $T")
        end
        return new{N, T, M, D}(temp)
    end
end

function instance(mat::SParametricMaterial{N, T}, pos::AbstractVector{<:Real}) where {N, T}
    massfracs = massfracfunc!(mat)(T, pos)
    density = densityfunc(mat)(massfracs, pos)
    return STemplateMaterial(mat, massfracs, density)
end

function massfracstr(mat::SParametricMaterial, ::MIME"text/plain")
    elms = sort(elms_vector(mat))
    res = "\nElements:"
    for elm in elms
        res *= "\n\t$(symbol(elm))(Z=$(z(elm)), A=$(_repr(a(elm, mat))))"
    end
    return res
end
function massfracstr(mat::SParametricMaterial, ::Nothing)
    elms = sort(elms_vector(mat))
    _str(elm) = "$(symbol(elm))"
    return "[" * join((_str(elm) for elm in elms), ", ") * "], "
end
densitystr(::SParametricMaterial, ::MIME{Symbol("text/plain")}) = ""
densitystr(::SParametricMaterial, ::Nothing) = ""

abstract type MTemplateMaterial{N, T, AUTO} <: TemplateMaterial{N, T, AUTO} end
# Parametric Material
mutable struct MTemplateMaterialSingle{N, T, AUTO, L, M, D} <: MTemplateMaterial{N, T, AUTO}
    const template::MaterialTemplate{N, T, M, D}
    const massfracs::SizedVector{N, T, Vector{T}}
    const atomicfracs::SizedVector{N, T, Vector{T}}
    density::T
    atoms_per_cm³::T
    const lock::L
    const poscache::SizedVector{3, T, Vector{T}}

    function MTemplateMaterialSingle{AUTO}(
        template::MaterialTemplate{N, T, M, D}, massfracs::SizedVector{N, T, Vector{T}}, atomicfracs::SizedVector{N, T, Vector{T}}, 
        density::T, atoms_per_cm³::T, lock::L, pos::SizedVector{3, T, Vector{T}}
    ) where {AUTO, N, T, M, D, L<:Union{Nothing, Base.AbstractLock}}
        new{N, T, AUTO, L, M, D}(template, massfracs, atomicfracs, density, atoms_per_cm³, lock, pos)
    end
end

position_cache(mat::MTemplateMaterialSingle) = mat.poscache

function Base.copy(mat::MTemplateMaterialSingle, copy_template=false)
    lock = nothing
    if !isnothing(mat.lock)
        try
            lock = typeof(mat.lock)()
            @warn "`mat.lock` could have been copied eroneously"
        catch ex
            println("lock $(mat.lock) could not be copied")
            throw(ex)
        end 
    end
    copy(mat, lock, copy_template)
end
function Base.copy(
    mat::MTemplateMaterialSingle{<:Any, <:Any, AUTO}, lock::Union{Base.AbstractLock, Nothing}, copy_template=false
    ) where AUTO
    _template = copy_template ? copy(template(mat)) : template(mat)
    MTemplateMaterialSingle{AUTO}(
        _template, copy(mat.massfracs), copy(mat.atomicfracs), mat.density, mat.atoms_per_cm³, lock, copy(mat.poscache)
    )
end

# Parametric Material
const MTemplateMaterialLocked{N, T, AUTO} = MTemplateMaterialSingle{N, T, AUTO, <:Base.AbstractLock}
const MTemplateMaterialUnLocked{N, T, AUTO} = MTemplateMaterialSingle{N, T, AUTO, Nothing}

function locked(eval, mat::MTemplateMaterialLocked)
    lock(mat.lock)
    try
        return eval(mat)
    finally
        unlock(mat.lock)
    end
end
function evaluateat(eval, mat::MTemplateMaterialLocked, pos::AbstractVector)
    locked(mat) do mat_
        eval(instance(mat_, pos))
    end
end

struct MTemplateMaterialThreaded{N, T, AUTO, ID, IDF, M, D} <: MTemplateMaterial{N, T, AUTO}
    buffer::DefaultDict{ID, MTemplateMaterialUnLocked{N, T, AUTO, M, D}}
    identifier::IDF

    function MTemplateMaterialThreaded{ID}(
        default::MTemplateMaterialUnLocked{N, T, AUTO, M, D}, identifier::IDF
    ) where {N, T, M, D, AUTO, ID, IDF}
        buffer = DefaultDict{ID, MTemplateMaterialUnLocked{N, T, AUTO, M, D}}() do
            if length(buffer) > 100
                @warn "maybe too many instances"
            end
            copy(default, false)
        end
        try
            id = identifier()
            if typeof(id) != ID
                @warn "the type returned by `identifier()` is not $ID. This could be different during runtime"
            end
        catch
            @warn "could not validate `identifier`"
        end
        new{N, T, AUTO, ID, IDF, M, D}(buffer, identifier)
    end
end

instance(mat::MTemplateMaterialThreaded) = mat.buffer[mat.identifier()]
template(mat::MTemplateMaterialThreaded) = mat |> instance |> template
massfracs(mat::MTemplateMaterialThreaded) = mat |> instance |> massfracs
atomicfracs(mat::MTemplateMaterialThreaded) = mat |> instance |> atomicfracs
density(mat::MTemplateMaterialThreaded) = mat |> instance |> density
atoms_per_cm³(mat::MTemplateMaterialThreaded) = mat |> instance |> atoms_per_cm³

Base.copy(mat::MTemplateMaterialThreaded, copy_template=true) = MTemplateMaterialThreaded(copy(instance(mat), copy_template))

function instance(mat::MTemplateMaterialSingle, x::AbstractArray{<:Real})
    _massfracs = massfracs(mat)
    massfracfunc!(mat)(_massfracs, x)
    mat.density = densityfunc(mat)(_massfracs, x)
    mat.atoms_per_cm³ = _atomicfracs_atoms_per_g!(atomicfracs(mat), atomicmasses(mat), _massfracs) * mat.density
    mat.poscache .= x
    return mat
end
function instance(mat::MTemplateMaterialThreaded, x::AbstractArray{<:Real})
    return instance(instance(mat), x)
    #return mat
end

function update(mat::Union{MTemplateMaterialThreaded, MTemplateMaterialSingle}, x::AbstractArray{<:Real})
    instance(mat, x)
    return mat
end

"""
    MTemplateMaterial(
        template::Union{MaterialTemplate, MTemplateMaterialSingle},
        massfracs::AbstractVector{<:AbstractFloat},
        density::AbstractFloat;
        autoupdate::Bool=false,
        lock::Union{Base.AbstractLock, Nothing}=nothing, 
        lastpos::Union{Nothing, AbstractArray}=nothing,
        properties::Union{Nothing, Dict{Symbol, <:Any}}=nothing,
    )

A template material with the given template, mass fractions, atomic fractions, density and atoms/g.
"""
function MTemplateMaterialSingle(
    template::MaterialTemplate{N, T}, massfracs::AbstractArray{T}, density::AbstractFloat; autoupdate::Bool=false,
    lock::Union{Base.AbstractLock, Nothing}=nothing, lastpos::Union{Nothing, AbstractArray}=nothing, kwargs...
) where {N, T}
    if length(massfracs) != N
        error("massfracs has a wrong length")
    end
    if length(kwargs) != 0
        @warn "These keyword arguments do nothing: $(join(kwargs, ", "))"
    end
    massfracs = SizedVector{N, T, Vector{T}}(massfracs)
    atomicfracs, atoms_per_g = _atomicfracs_atoms_per_g(atomicmasses(template), massfracs)
    lastpos = SizedVector{3, T}(Vector{T}(something(lastpos, zeros(T, 3))))
    MTemplateMaterialSingle{autoupdate}(template, massfracs, atomicfracs, T(density), T(atoms_per_g * density), lock, lastpos)
end
function MTemplateMaterialSingle(
    mat::MTemplateMaterialSingle, massfracs::AbstractVector, density::AbstractFloat; lock::Union{Base.AbstractLock, Nothing}=nothing,
    kwargs...
)
    if isnothing(lock)
        try
            lock = typeof(mat.lock)()
            @warn "`mat.lock` could have been copied eroneously"
        catch
            @warn "tried to copy `mat.lk` but failed, leaving it as `nothing`"
            nothing
        end
    end
    MTemplateMaterialSingle(template(mat), massfracs, density; lock=lock, kwargs...)
end

"""
    MTemplateMaterialThreaded(default::MTemplateMaterialUnLocked; id_type::Type=Nothing, identifier::Any=nothing)

A mutable template material with buffer for each thread/task to prevent dataraces.
"""
function MTemplateMaterialThreaded(default::MTemplateMaterialUnLocked; id_type::Type=Nothing, identifier::Any=nothing)
    identifier = something(identifier, ()->objectid(current_task()))
    if id_type == Nothing
        try
            id_type = typeof(identifier())
        catch
            error("could not infer the type returned by `identifier()`")
        end
    end
    return MTemplateMaterialThreaded{id_type}(default, identifier)
end


"""
    MTemplateMaterial(
        template::MaterialTemplate,
        massfracs::AbstractVector{<:AbstractFloat},
        density::AbstractFloat;
        lock::Union{Base.AbstractLock, Nothing}=nothing, 
        lastpos::Union{Nothing, AbstractArray}=nothing,
        threaded::Bool=true,
        id_type::Type=Nothing,
        identifier::Any=nothing
        autoupdate::Bool=true,
    )

A mutable template material with the given template, mass fractions, atomic fractions, density and atoms/g.
"""
function MTemplateMaterial(
    template::MaterialTemplate, massfracs::AbstractArray, density::AbstractFloat; lock::Union{Base.AbstractLock, Nothing}=nothing, 
    threaded::Bool=false, id_type::Type=Nothing, identifier::Any=nothing, kwargs...
)
    if threaded && !(isnothing(lock))
        @warn "lock `lk` must be nothing for threaded material, setting it to `nothing`"
        lock = nothing
    end
    res = MTemplateMaterialSingle(template, massfracs, density; lock=lock, kwargs...)
    if threaded
        return MTemplateMaterialThreaded(res; id_type=id_type, identifier=identifier)
    end
    return res
end

"""
    MTemplateMaterial(
        name::AbstractString,
        elms::AbstractVector{Element},
        massfracfunc!::Any;
        static::Bool=false,
        atomicmasses::Union{AbstractArray, Nothing}=nothing,
        properties::Union{Nothing, Dict{Symbol, <:Any}}=nothing,
        densityfunc::Any=nothing,
        massfractype::Type{<:AbstractFloat}=Float64,
        threaded::Bool=true,
        id_type::Type=Nothing,
        identifier::Any=nothing
        autoupdate::Bool=true,
        pos::Union{AbstractVector{<:AbstractFloat}, Nothing}=nothing
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
function TemplateMaterial(
    template::MaterialTemplate{N, T};
    static::Bool=true,
    pos::Union{AbstractVector{<:AbstractFloat}, Nothing}=nothing,
    kwargs...
) where {N, T}
    if static
        if get(kwargs, :autoupdate, false)
            return SParametricMaterial(template)
        elseif !isnothing(pos)
            return STemplateMaterial(template, pos)
        else
            error("Cannot create static template material without `autoupdate=true` or a `pos` being specified")
        end
    end
    massfracs = SizedVector{N, T}(zeros(T, N))
    x = something(pos, zeros(3))
    try
        massfracs = massfracfunc!(template)(massfracs, x)
    catch ex
        error("masfracfunc! is not a suitable function, Please check the source error: $(ex.msg)")
    end
    density = nothing
    try
        density = densityfunc(template)(massfracs, x)
    catch ex
        error("densityfunc is not a suitable function, Please check the source error: $(ex.msg)")
    end
    if ~(density isa Real)
        error("densityfunc must return a scalar")
    end
    return MTemplateMaterial(template, massfracs, density; lastpos=x, kwargs...)
end

function TemplateMaterial(
    name::AbstractString,
    elms::AbstractVector{Element},
    massfracfunc!::Any;
    atomicmasses::Union{AbstractArray, Nothing}=nothing,
    properties::Union{Dict{Symbol, <:Any}, Nothing}=nothing,
    densityfunc::Any=nothing,
    massfractype::Type{<:AbstractFloat}=Float64,
    kwargs...
)
    if !isnothing(massfracfunc!) && isnothing(densityfunc)
        densityfunc = volume_conserving_density(elms)
    end
    template = MaterialTemplate(massfractype, name, elms, massfracfunc!, densityfunc, atomicmasses, properties)
    return TemplateMaterial(template; kwargs...)
end

"""
    ParametricMaterial(
        name::AbstractString,
        elms::AbstractVector{Element},
        massfracfunc!::Any;
        static::Bool=false,
        atomicmasses::Union{AbstractArray, Nothing}=nothing,
        densityfunc::Any=nothing,
        massfractype::Type{<:AbstractFloat}=Float64,
        threaded::Bool=true,
        id_type::Type=Nothing,
        identifier::Any=nothing
        pos::Union{AbstractVector{<:AbstractFloat}, Nothing}=nothing
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
    args...;
    kwargs...
)
    if haskey(kwargs, :autoupdate)
        error("keyword argument `autoupdate` not allowed for `ParametricMaterial`")
    end
    TemplateMaterial(args...; autoupdate=true, kwargs...)
end

function massfractions!(mat::ParametricMaterial, massfracs::AbstractVector{<:Real}, x::AbstractVector{<:Real})
    return massfracfunc!(mat)(massfracs, x)
end

function massfractions(mat::ParametricMaterial, x::AbstractVector{<:Real})
    massfracs = similar(mat.massfracs)
    return massfracfunc!(mat)(massfracs, x)
end

function density(mat::ParametricMaterial, x::AbstractVector{<:Real})
    return densityfunc(mat)(massfracs(mat), x)
end

function STemplateMaterial(::Type{T}, mat::Material) where T
    ρ = density(mat)
    elms_ = collect(elms(mat))
    a_ = [T(a(elm, mat)) for elm in elms_]
    template = MaterialTemplate(T, mat.name, elms_, nothing, nothing, a_, mat.properties)
    massfracs = [T(mat[elm]) for elm in elms_]
    return STemplateMaterial(template, massfracs, ρ)
end
STemplateMaterial(mat::Material) = STemplateMaterial(Float64, mat)

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


