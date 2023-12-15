"""
Convert a data model to a typed CLI argument parser
"""

from typing import Any, Sequence, Type
import dataclasses

import pydantic
from tap import Tap


@dataclasses.dataclass(frozen=True)
class _FieldData:
    """
    Data about a field which is sufficient to inform a Tap variable/argument.
    """

    name: str
    annotation: Type
    is_required: bool
    default: Any
    description: str | None = ""


def _field_data_from_pydantic_model(
    pydantic_model: Type[pydantic.BaseModel],
) -> list[_FieldData]:
    return [
        _FieldData(
            name,
            field.annotation,
            field.is_required(),
            field.default,
            field.description,
        )
        for name, field in pydantic_model.model_fields.items()
    ]


def _field_data_from_dataclass(dataclass: Type) -> list[_FieldData]:
    fields = dataclasses.fields(dataclass)

    def is_required(field: dataclasses.Field) -> bool:
        return (
            field.default is dataclasses.MISSING
            and field.default_factory is dataclasses.MISSING
        )

    return [
        _FieldData(
            field.name,
            field.type,
            is_required(field),
            field.default,
            field.metadata.get("description"),
        )
        for field in fields
    ]


def _field_data(data_model: Any) -> list[_FieldData]:
    if dataclasses.is_dataclass(data_model):
        return _field_data_from_dataclass(data_model)
    elif issubclass(data_model, pydantic.BaseModel):
        return _field_data_from_pydantic_model(data_model)
    else:
        raise TypeError(
            "data_model must be a dataclass or a Pydantic BaseModel. "
            f"Got {type(data_model)}"
        )


def _tap_class(field_data: Sequence[_FieldData]) -> Type[Tap]:
    class ArgParser(Tap):
        def configure(self):
            for field in field_data:
                name = field.name
                self._annotations[name] = field.annotation
                self.class_variables[name] = {"comment": field.description or ""}
                if field.is_required:
                    kwargs = {}
                else:
                    kwargs = dict(required=False, default=field.default)
                self.add_argument(f"--{name}", **kwargs)

    return ArgParser


def tap_from_data_model(data_model: Any) -> Type[Tap]:
    """
    Convert a data model to a typed CLI argument parser.

    Parameters
    ----------
    data_model : Any
        a dataclass (class or instance) or Pydantic model

    Returns
    -------
    Type[Tap]
        a typed CLI argument parser class

    Note
    ----
    For a dataclass, argument descriptions are set to the field's
    `metadata["description"]`.

    For example::

        @dataclass
        class Data:
            my_field: str = field(metadata={"description": "field description})
    """
    field_data = _field_data(data_model)
    return _tap_class(field_data)
