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

cpu_current_task_id() = objectid(current_task())

"""
Abstract type for defining Materials which Hold basic data about a material in Vector-like form for fast arithmetic.

It is a parametric subtype of `AbstractMaterial{U, V}`. `N` is the number of elements. `W` is the element type of atomic fractions.
In addition to `name`, and `properties` as in `AbstractMaterial`, all subtypes must contain the following fields or methods.
    +`elms:AbstractVector{Element}` or `elms_vec(mat)` - vector of elements.
    +`massfracs:AbstractVector{U}` or `massfracs(mat)` - vector of mass fractions in the same order as `elms`.
    +`a:AbstractVector{V}` or `atomicmasses(mat)` - vector of atomic numbers in the same order as `elms`.
    +`atomicfracs::AbstractVectror{W}` or `atomicfracs(mat)` - vector of atomic fractions in the same order as `elms`.
    +`elmdict:Dict{Element, Int}` or `elmdict(mat)` - a mapping from element to integer index of the ement in `elms`.
    +`density::D` or `density(mat)` - density of the material
    +`atoms_per_g::A` or `atoms_per_g(mat)` - no oof atoms in 1 g of the material`

`U`, `V`, `W`, `D` and `A` are typically but not necessarily all `AbstractFloat`s.
Some functionalities only work when `elms`, `massfracs` and `a` have regular indexin, i.e. starting with 1 and with increments of 1.
"""
abstract type VectorizedMaterial{N, U, V, W, D, A} <: AbstractMaterial{U, V} end

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


Base.getindex(mat::VectorizedMaterial{<:Any, U}, elm::Element) where U =
    get(mat.massfracs, get(elmdict(mat), elm, 0), zero(U))
Base.getindex(mat::VectorizedMaterial{<:Any, U}, z::Int) where U =
    get(mat, elements[z], zero(U))
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
typestr(mat::VectorizedMaterial, ::Any) = split("$(typeof(mat))", "{")[1]

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
function massfracstr(mat::VectorizedMaterial, ::Any)
    elms = sort(elms_vector(mat))
    _str(elm) = "$(symbol(elm)) => $(_repr(massfrac(mat, elm)))"
    return "[" * join((_str(elm) for elm in elms), ", ") * "], "
end

densitystr(mat::VectorizedMaterial, ::MIME"text/plain") = "\nMass density: $(_repr(density(mat))) $(u"g/cm^3")"
densitystr(mat::VectorizedMaterial, ::Any) = "ρ = $(_repr(density(mat))) $(u"g/cm^3")}"

a(elm::Element, mat::VectorizedMaterial) = get(atomicmasses(mat), get(elmdict(mat), elm, 0), a(elm))

# A set of functions that work with indices
elm(mat::VectorizedMaterial, index::Integer) = getindex(elms_vector(mat), index)
# Same as elm but doesn't check if index in bounds. Can be unsafe. Use only with eachindex.
elm_nocheck(mat::VectorizedMaterial, index::Integer) = @inbounds elms_vector(mat)[index]
massfrac(mat::VectorizedMaterial{<:Any, U}, index::Integer) where U = get(massfracs(mat), index, zero(U))
massfrac(mat::VectorizedMaterial, elm::Element) = massfrac(mat, get(elmdict(mat), elm, 0))
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

atoms_per_g(mat::VectorizedMaterial) = mat.atoms_per_g
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

struct MaterialTemplate{N, V<:AbstractFloat, M, D} <: VectorizedMaterial{N, Nothing, V, Nothing, Nothing, Nothing}
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

abstract type TemplateMaterial{N, U, V, W, D, A} <: VectorizedMaterial{N, U, V, W, D, A} end

template(mat::TemplateMaterial) = mat.template
name(mat::TemplateMaterial) = mat |> template |> name
properties(mat::TemplateMaterial) = mat |> template |> properties
elms_vector(mat::TemplateMaterial) = mat |> template |> elms_vector
atomicmasses(mat::TemplateMaterial) = mat |> template |> atomicmasses
elmdict(mat::TemplateMaterial) = mat |> template |> elmdict
massfracfunc!(mat::TemplateMaterial) = mat |> template |> massfracfunc!
densityfunc(mat::TemplateMaterial) = mat |> template |> densityfunc

struct STemplateMaterial{N, U<:AbstractFloat, V, W<:AbstractFloat, D<:AbstractFloat, A<:AbstractFloat} <: TemplateMaterial{N, U, V, W, D, A}
    template::MaterialTemplate{N, V}
    massfracs::SVector{N, U}
    atomicfracs::SVector{N, W}
    density::D
    atoms_per_g::A
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

abstract type MTemplateMaterial{N, U, V, W, D, A, AUTO} <: TemplateMaterial{N, U, V, W, D, A} end
const ParametricMaterial{N, U, V, W, D, A} = MTemplateMaterial{N, U, V, W, D, A, true}

# Parametric Material
struct MTemplateMaterialSingle{N, U<:AbstractFloat, V<:AbstractFloat, W<:AbstractFloat, D, A, AUTO, L} <: MTemplateMaterial{N, U, V, W, D, A, AUTO}
    template::MaterialTemplate{N, V}
    massfracs::MVector{N, U}
    atomicfracs::MVector{N, W}
    density::MVector{1, D}
    atoms_per_g::MVector{1, A}
    properties::Dict{Symbol, Any}
    lk::L

    function MTemplateMaterialSingle{AUTO}(
        template::MaterialTemplate{N, V}, massfracs::MVector{N, U}, atomicfracs::MVector{N, W}, 
        density::D, atoms_per_g::A, properties::Dict, lk::L
    ) where {AUTO, N, U, V, W, D, A, L}
        new{N, U, V, W, D, A, AUTO, L}(template, massfracs, atomicfracs, MVector{1, D}(density), MVector{1, A}(atoms_per_g), properties, lk)
    end
end

_properties(mat::MTemplateMaterialSingle) = mat.properties
function Base.copy(mat::MTemplateMaterialSingle, copy_template=false)
    lk = nothing
    if !isnothing(mat.lk)
        try
            lk = typeof(mat.lk)()
        catch ex
            println("lock $(mat.lk) could not be copied")
            throw(ex)
        end 
    end
    copy(mat, lk, copy_template)
end
function Base.copy(
    mat::MTemplateMaterialSingle{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, AUTO}, lk::Union{Base.AbstractLock, Nothing}, copy_template=false
    ) where AUTO
    _template = copy_template ? copy(template(mat)) : template(mat)
    MTemplateMaterialSingle{A}(
        _template, copy(mat.massfracs), copy(mat.atomicfracs), copy(mat.density), copy(mat.atoms_per_g), copy(mat.properties), lk
    )
end
density(mat::MTemplateMaterialSingle) = @inbounds mat.density[1]
atoms_per_g(mat::MTemplateMaterialSingle) = @inbounds mat.atoms_per_g[1]

# Parametric Material
const MTemplateMaterialLocked{N, U, V, W, D, A, AUTO} = MTemplateMaterialSingle{N, U, V, W, D, A, AUTO, <:Base.AbstractLock}
const MTemplateMaterialUnLocked{N, U, V, W, D, A, AUTO} = MTemplateMaterialSingle{N, U, V, W, D, A, AUTO, Nothing}

function locked(eval, mat::MTemplateMaterialLocked)
    lock(mat.lk)
    try
        return eval(mat)
    finally
        unlock(mat)
    end
end

struct MTemplateMaterialThreaded{N, U, V, W, D, A, AUTO, ID, IDF} <: MTemplateMaterial{N, U, V, W, D, A, AUTO}
    buffer::DefaultDict{ID, MTemplateMaterialUnLocked{N, U, V, W, D, A, AUTO}}
    identifier::IDF

    function MTemplateMaterialThreaded{ID}(
        default::MTemplateMaterialUnLocked{N, U, V, W, D, A, AUTO}, identifier::IDF
    ) where {N, U, V, W, D, A, AUTO, ID, IDF}
        buffer = DefaultDict{I, MTemplateMaterialUnLocked{N, U, V, W, D, A, AUTO}}() do
            if length(buffer) > 1000
                @warn "too many instances"
            end
            copy(default, false)
        end
        try
            id = identifier()
            if typeof(id) != ID
                @warn "the type returned by `identifier()` is not the same as `ID`, this can be different during runtime"
            end
        catch
            @warn "could not validate `identifier`"
        end
        new{N, U, V, W, D, A, AUTO, ID, IDF}(buffer, identifier)
    end
end

instance(mat::MTemplateMaterialThreaded) = mat.buffer[mat.identifier()]
template(mat::MTemplateMaterialThreaded) = mat |> instance |> template
massfracs(mat::MTemplateMaterialThreaded) = mat |> instance |> massfracs
atomicfracs(mat::MTemplateMaterialThreaded) = mat |> instance |> atomicfracs
_properties(mat::MTemplateMaterialThreaded) = mat |> instance |> _properties

Base.copy(mat::MTemplateMaterialThreaded, copy_template=true) = MTemplateMaterialThreaded(copy(instance(mat), copy_template))

properties(mat::MTemplateMaterial) = merge(template(mat), _properties(mat))
Base.getindex(mat::MTemplateMaterial, sym::Symbol) = get(_properties(mat), sym, getindex(properties(template(mat)), sym))
Base.get(mat::MTemplateMaterial, sym::Symbol, def) = get(_properties(mat), sym,  get(properties(template(mat)), sym, def))
Base.setindex!(mat::MTemplateMaterial, val, sym::Symbol) = setindex!(_properties(mat), val, sym)

function update!(mat::MTemplateMaterialSingle, x::AbstractArray{<:Real})
    _massfracs = massfracs(mat)
    massfracfunc!(mat)(_massfracs, x)
    mat.density[1] = densityfunc(mat)(_massfracs, x)
    mat.atoms_per_g[1] = _atomicfracs_atoms_per_g!(atomicfracs(mat), atomicmasses(mat), _massfracs)
    mat[:LastPos] = x
end
function update!(mat::MTemplateMaterialThreaded, x::AbstractArray{<:Real})
    update!(instance(mat), x)
end

"""
    MTemplateMaterial(
        template::Union{MaterialTemplate, MTemplateMaterialSingle},
        massfracs::AbstractVector{<:AbstractFloat},
        density::AbstractFloat;
        autoupdate::Bool=false,
        lk::Union{Base.AbstractLock, Nothing}=nothing, 
        lastpos::Union{Nothing, AbstractArray}=nothing,
        properties::Union{Nothing, Dict{Symbol, <:Any}}=nothing,
    )

A template material with the given template, mass fractions, atomic fractions, density and atoms/g.
"""
function MTemplateMaterialSingle(
    template::MaterialTemplate{N, V}, massfracs::AbstractArray{U}, density::AbstractFloat; autoupdate::Bool=false,
    lk::Union{Base.AbstractLock, Nothing}=nothing, lastpos::Union{Nothing, AbstractArray}=nothing,
    properties::Union{Nothing, Dict{Symbol, <:Any}}=nothing, kwargs...
) where {N, U, V}
    if length(massfracs) != N
        error("massfracs has a wrong length")
    end
    if length(kwargs) != 0
        @warn "These keyword arguments do nothing: $(join(kwargs, ", "))"
    end
    massfracs = MVector{N, U}(massfracs)
    atomicfracs, atoms_per_g = _atomicfracs_atoms_per_g(atomicmasses(template), massfracs)
    properties = something(properties, Dict{Symbol, AbstractFloat}())
    properties[:LastPos] = lastpos
    MTemplateMaterialSingle{autoupdate}(template, massfracs, atomicfracs, density, atoms_per_g, properties, lk)
end
function MTemplateMaterialSingle(
    mat::MTemplateMaterialSingle, massfracs::AbstractVector, density::AbstractFloat; lk::Union{Base.AbstractLock, Nothing}=nothing,
    kwargs...
)
    if isnothing(lk)
        try
            lk = typeof(mat.lk)()
        catch
            @warn "tried to copy `mat.lk` but failed, leaving it as `nothing`"
            nothing
        end
    end
    MTemplateMaterialSingle(template(mat), massfracs, density; lk=lk, kwargs...)
end

"""
    MTemplateMaterialThreaded(default::MTemplateMaterialUnLocked; id_type::Type=Nothing, identifier::Any=nothing)

A material with buffer for each thread/task to prevent dataraces.
"""
function MTemplateMaterialThreaded(default::MTemplateMaterialUnLocked; id_type::Type=Nothing, identifier::Any=nothing)
    identifier = something(identifier, cpu_current_task_id)
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
        autoupdate::Bool=false,
        lk::Union{Base.AbstractLock, Nothing}=nothing, 
        lastpos::Union{Nothing, AbstractArray}=nothing,
        properties::Union{Nothing, Dict{Symbol, <:Any}}=nothing,
        threaded::Bool=true,
        id_type::Type=Nothing,
        identifier::Any=nothing
        autoupdate::Bool=true,
    )

A template material with the given template, mass fractions, atomic fractions, density and atoms/g.

**Properties**

- `:Density` # Density in g/cm³
- `:Description` # Human friendly
- `:Pedigree` #  Quality indicator for compositional data ("SRM-XXX", "CRM-XXX", "NIST K-Glass", "Stoichiometry", "Wet-chemistry by ???", "WDS by ???", "???")
- `:Conductivity` = :Insulator | :Semiconductor | :Conductor
- `:OtherUserProperties` # Other properties can be defined as needed
"""
function MTemplateMaterial(
    template::MaterialTemplate, massfracs::AbstractArray, density::AbstractFloat; lk::Union{Base.AbstractLock, Nothing}=nothing, 
    threaded::Bool=false, id_type::Type=Nothing, identifier::Any=nothing, kwargs...
)
    if threaded && !(isnothing(lk))
        @warn "lock `lk` must be nothing for threaded material, setting it to `nothing`"
        lk = nothing
    end
    res = MTemplateMaterialSingle(template, massfracs, density; lk=lk, kwargs...)
    if threaded
        return MTemplateMaterialThreaded(res; id_type=id_type, identifier=identifier)
    end
    return res
end

"""
    MTemplateMaterial(
        name::AbstractString,
        elms::AbstractVector{Element},
        massfracfunc!::Any,
        atomicmasses::Union{AbstractArray, Nothing}=nothing,
        densityfunc::Any=nothing,
        massfractype::Type{<:AbstractFloat}=Float64,
        threaded::Bool=true,
        id_type::Type=Nothing,
        identifier::Any=nothing
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
    massfracfunc!::Any;
    atomicmasses::Union{AbstractArray, Nothing}=nothing,
    densityfunc::Any=nothing,
    massfractype::Type{<:AbstractFloat}=Float64,
    kwargs...
)
    if isnothing(densityfunc)
        densityfunc = volume_conserving_density(elms)
    end
    template = MaterialTemplate(name, elms, atomicmasses, properties, massfracfunc!, densityfunc)
    massfracs = zero(MVector{length(template), massfractype})
    x = get(kwargs, :lastpos, zeros(3))
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
        density = densityfunc(massfracs, x)
    catch ex
        error("densityfunc is not a suitable function, Please check the source error: $(ex.msg)")
    end
    if ~(density isa Real)
        error("densityfunc must return a scalar")
    end
    if haskey(kwargs, :lastpos)
        return MTemplateMaterial(template, massfracs, density; kwargs...)
    end
    return MTemplateMaterial(template, massfracs, density; lastpos=x, kwargs...)
end

"""
    ParametricMaterial(
        name::AbstractString,
        elms::AbstractVector{Element},
        massfracfunc!::Any,
        atomicmasses::Union{AbstractArray, Nothing}=nothing,
        densityfunc::Any=nothing,
        massfractype::Type{<:AbstractFloat}=Float64,
        threaded::Bool=true,
        id_type::Type=Nothing,
        identifier::Any=nothing
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
    massfracfunc!::Any;
    kwargs...
)
    if haskey(kwargs, :autoupdate)
        error("keyword argument `autoupdate` not allowed for `ParametricMaterial`")
    end
    MTemplateMaterial(name, elms, massfracfunc!; autoupdate=true, kwargs...)
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


