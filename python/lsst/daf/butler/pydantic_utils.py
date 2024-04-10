# This file is part of daf_butler.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (http://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This software is dual licensed under the GNU General Public License and also
# under a 3-clause BSD license. Recipients may choose which of these licenses
# to use; please see the files gpl-3.0.txt and/or bsd_license.txt,
# respectively.  If you choose the GPL option then the following text applies
# (but note that there is still no warranty even if you opt for BSD instead):
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import annotations

__all__ = ("DeferredValidation", "get_universe_from_context", "SerializableRegion", "SerializableTime")

from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Generic, Self, TypeAlias, TypeVar, get_args

import pydantic
from astropy.time import Time
from lsst.sphgeom import Region
from pydantic_core import core_schema

from .time_utils import TimeConverter

if TYPE_CHECKING:
    from .dimensions import DimensionUniverse

_T = TypeVar("_T")


def get_universe_from_context(context: dict[str, Any] | None) -> DimensionUniverse:
    """Extract the dimension universe from a Pydantic validation context
    dictionary.

    Parameters
    ----------
    context : `dict`
        Dictionary obtained from `pydantic.ValidationInfo.context`.

    Returns
    -------
    universe : `DimensionUniverse`
        Definitions for all dimensions.

    Notes
    -----
    This function just provides consistent error handling around::

        context["universe"]
    """
    if context is None:
        raise ValueError("This object requires Pydantic validation context to be deserialized.")
    try:
        return context["universe"]
    except KeyError:
        raise ValueError(
            "This object requires the DimensionUniverse to be provided in the Pydantic validation "
            "context to be deserialized."
        ) from None


class DeferredValidation(Generic[_T]):
    """A base class whose subclasses define a wrapper for a Pydantic-aware type
    that defers validation but declares the same JSON schema.

    Parameters
    ----------
    data : `object`
        Unvalidated data representing an instance of the wrapped type.  This
        may be the serialized form of the wrapped type, an instance of the
        wrapped type, or anything else - but the in the latter case, calls to
        `validated` will fail with a Pydantic validation error, and if the
        object is known to be an instance of the wrapped type, `from_validated`
        should be preferred.

    Notes
    -----
    This class must be subclassed to be used, but subclasses are always
    trivial::

        class SerializableThing(DeferredValidation[Thing]):
            pass

    The type parameter for `DeferredValidation` may be a special typing object
    like `typing.Union` or `typing.Annotated` instead of an actual `type`
    object.  The only requirement is that it must be a type Pydantic
    recognizes, like a `pydantic.BaseModel` subclass, a dataclass, or a
    primitive built-in.

    A wrapper subclass (e.g. ``SerializableThing``) can be used with Pydantic
    via `pydantic.TypeAdapter` or as a field in `pydantic.BaseModel`.  The JSON
    schema of the wrapper will be consistent with the JSON schema of the
    wrapped type (though it may not use JSON pointer references the same way),
    and Pydantic serialization will work regardless of whether the wrapper
    instance was initialized with the raw type or the wrapped type. Pydantic
    validation of the wrapper will effectively do nothing, however; instead,
    the `validated` method must be called to return a fully-validated instance
    of the wrapped type, which is then cached within the wrapper for subsequent
    calls to `validated`.

    Indirect subclasses of `DeferredValidation` are not permitted.

    A major use case for `DeferredValidation` is types whose validation
    requires additional runtime context (via the Pydantic "validation context"
    dictionary that can custom validator hooks can access).  These types are
    often first deserialized (e.g. by FastAPI) in a way that does not permit
    that context to be provided.
    """

    def __init__(self, data: Any):
        self._data = data
        self._is_validated = False

    @classmethod
    def from_validated(cls, wrapped: _T) -> Self:
        """Construct from an instance of the wrapped type.

        Unlike invoking the constructor with an instance of the wrapped type,
        this factory marks the held instance as already validated (since that
        is expected to be guaranteed by the caller, possibly with the help of
        static analysis), which sidesteps Pydantic validation in later calls
        to `validated`.

        Parameters
        ----------
        wrapped : `object`
            Instance of the wrapped type.

        Returns
        -------
        wrapper : `DeferredValidation`
            Instance of the wrapper.
        """
        result = cls(wrapped)
        result._is_validated = True
        return result

    def validated(self, **kwargs: Any) -> _T:
        """Validate (if necessary) and return the validated object.

        Parameters
        ----------
        **kwargs
            Additional keywords arguments are passed as the Pydantic
            "validation context" `dict`.

        Returns
        -------
        wrapped
            An instance of the wrapped type.  This is also cached for the next
            call to `validated`, *which will ignore ``**kwargs``*.
        """
        if not self._is_validated:
            self._data = self._get_wrapped_type_adapter().validate_python(self._data, context=kwargs)
            self._is_validated = True
        return self._data

    _WRAPPED_TYPE: ClassVar[Any | None] = None
    _WRAPPED_TYPE_ADAPTER: ClassVar[pydantic.TypeAdapter[Any] | None] = None

    def __init_subclass__(cls) -> None:
        # We override __init_subclass__ to grab the type argument to the
        # DeferredValidation base class, since that's the wrapped type.
        assert (
            cls.__base__ is DeferredValidation
        ), "Indirect subclasses of DeferredValidation are not allowed."
        try:
            # This uses some typing internals that are not as stable as the
            # rest of Python, so it's the messiest aspect of this class, but
            # even if it breaks on (say) some Python minor releases, it should
            # be easy to detect and fix and I think that makes it better than
            # requiring the wrapped type to be declared twice when subclassing.
            # Since the type-checking ecosystem depends on this sort of thing
            # to work it's not exactly private, either.
            cls._WRAPPED_TYPE = get_args(cls.__orig_bases__[0])[0]  # type: ignore
        except Exception as err:
            raise TypeError("DeferredValidation must be subclassed with a single type parameter.") from err
        return super().__init_subclass__()

    @classmethod
    def _get_wrapped_type_adapter(cls) -> pydantic.TypeAdapter[_T]:
        """Return the Pydantic adapter for the wrapped type, constructing and
        caching it if necessary.
        """
        if cls._WRAPPED_TYPE_ADAPTER is None:
            if cls._WRAPPED_TYPE is None:
                raise TypeError("DeferredValidation must be subclassed to be used.")
            cls._WRAPPED_TYPE_ADAPTER = pydantic.TypeAdapter(cls._WRAPPED_TYPE)
        return cls._WRAPPED_TYPE_ADAPTER

    def _serialize(self) -> Any:
        """Serialize this object."""
        if self._is_validated:
            return self._get_wrapped_type_adapter().dump_python(self._data)
        else:
            return self._data

    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type: Any, _handler: pydantic.GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        # This is the Pydantic hook for overriding serialization and
        # validation.  It's also normally the hook for defining the JSON
        # schema, but we throw that JSON schema away and define our own in
        # __get_pydantic_json_schema__.
        return core_schema.json_or_python_schema(
            # When deserializing from JSON, invoke the constructor with the
            # result of parsing the JSON into Python primitives.
            json_schema=core_schema.no_info_plain_validator_function(cls),
            # When validating a Python dict...
            python_schema=core_schema.union_schema(
                [
                    # ...first see if we already have an instance of the
                    # wrapper...
                    core_schema.is_instance_schema(cls),
                    # ...and otherwise just call the constructor on whatever
                    # we were given.
                    core_schema.no_info_plain_validator_function(cls),
                ]
            ),
            # When serializing to JSON, just call the _serialize method.
            serialization=core_schema.plain_serializer_function_ser_schema(cls._serialize),
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls, _core_schema: core_schema.CoreSchema, handler: pydantic.json_schema.GetJsonSchemaHandler
    ) -> pydantic.json_schema.JsonSchemaValue:
        # This is the Pydantic hook for customizing JSON schema.  We ignore
        # the schema generated for this class, and just return the JSON schema
        # of the wrapped type.
        json_schema = handler(cls._get_wrapped_type_adapter().core_schema)
        return handler.resolve_ref_schema(json_schema)


class _SerializedRegionAnnotation:
    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type: Any, _handler: pydantic.GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        # This is Pydantic's way of declaring a "schema" that expects a `str`
        # and then calls _deserialize.
        from_str_schema = core_schema.chain_schema(
            [
                core_schema.str_schema(),
                core_schema.no_info_plain_validator_function(cls._deserialize),
            ]
        )
        return core_schema.json_or_python_schema(
            # When reading JSON, do just that.
            json_schema=from_str_schema,
            # When validating Python, do that only if we don't already have a
            # Region instance.
            python_schema=core_schema.union_schema([core_schema.is_instance_schema(Region), from_str_schema]),
            # When serializing to JSON, just call the _serialize method.
            serialization=core_schema.plain_serializer_function_ser_schema(cls._serialize),
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls, _core_schema: core_schema.CoreSchema, handler: pydantic.json_schema.GetJsonSchemaHandler
    ) -> pydantic.json_schema.JsonSchemaValue:
        # JSON schema is the usual one for `str` fields...
        result = handler.resolve_ref_schema(handler(core_schema.str_schema()))
        # ...with a custom description and a note that it's base64-encoded.
        result["description"] = "A region on the sphere from the lsst.sphgeom package."
        result["media"] = {"binaryEncoding": "base16", "type": "application/lsst.sphgeom"}
        return result

    @staticmethod
    def _deserialize(value: str) -> Region:
        return Region.decode(bytes.fromhex(value))

    @staticmethod
    def _serialize(region: Region) -> str:
        return region.encode().hex()


SerializableRegion: TypeAlias = Annotated[Region, _SerializedRegionAnnotation]
"""A Pydantic-annotated version of `lsst.sphgeom.Region`.

An object annotated with this type is always an `lsst.sphgeom.Region` instance
in Python, but unlike `lsst.sphgeom.Region` itself it can be used as a type
in Pydantic models and type adapters, resulting in the field being saved as
a hex encoding of the sphgeom-encoded bytes.
"""


class _SerializedTimeAnnotation:
    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type: Any, _handler: pydantic.GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        # This is Pydantic's way of declaring a "schema" that expects an `int`
        # and then calls _deserialize.
        from_int_schema = core_schema.chain_schema(
            [
                core_schema.int_schema(),
                core_schema.no_info_plain_validator_function(cls._deserialize),
            ]
        )
        return core_schema.json_or_python_schema(
            # When reading JSON, do just that.
            json_schema=from_int_schema,
            # When validating Python, do that only if we don't already have a
            # Time instance.
            python_schema=core_schema.union_schema([core_schema.is_instance_schema(Time), from_int_schema]),
            # When serializing to JSON, just call the _serialize method.
            serialization=core_schema.plain_serializer_function_ser_schema(cls._serialize),
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls, _core_schema: core_schema.CoreSchema, handler: pydantic.json_schema.GetJsonSchemaHandler
    ) -> pydantic.json_schema.JsonSchemaValue:
        # JSON schema is the usual one for `int` fields...
        result = handler.resolve_ref_schema(handler(core_schema.int_schema()))
        # ...with a custom description.
        result["description"] = "A TAI time represented as integer nanoseconds since 1970-01-01 00:00:00."
        return result

    @staticmethod
    def _deserialize(value: int) -> Time:
        return TimeConverter().nsec_to_astropy(value)

    @staticmethod
    def _serialize(time: Time) -> int:
        return TimeConverter().astropy_to_nsec(time)


SerializableTime: TypeAlias = Annotated[Time, _SerializedTimeAnnotation]
"""A Pydantic-annotated version of `astropy.time.Time`.

An object annotated with this type is always an `astropy.time.Time` instance
in Python, but unlike `astropy.time.Time` itself it can be used as a type
in Pydantic models and type adapters, resulting in the field being saved as
integer nanoseconds since 1970-01-01 00:00:00.
"""
