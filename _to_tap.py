"""
Convert a data model to a typed CLI argument parser
"""

import dataclasses
from typing import Any, Sequence, Type

import pydantic
from tap import Tap


_PydanticField = pydantic.fields.FieldInfo | pydantic.dataclasses.FieldInfo


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


def _field_data_from_dataclass_field(name: str, field: dataclasses.Field) -> _FieldData:
    def is_required(field: dataclasses.Field) -> bool:
        return (
            field.default is dataclasses.MISSING
            and field.default_factory is dataclasses.MISSING
        )

    return _FieldData(
        name,
        field.type,
        is_required(field),
        field.default,
        field.metadata.get("description"),
    )


def _field_data_from_pydantic_field(name: str, field: _PydanticField) -> _FieldData:
    return _FieldData(
        name, field.annotation, field.is_required(), field.default, field.description
    )


def _fields_data(data_model: Any) -> list[_FieldData]:
    # Iterate through fields to handle:
    #   1. mixing fields w/ classes, e.g., using pydantic Fields in a (builtin)
    #      dataclass, or using (builtin) dataclass fields in a pydantic BaseModel
    #   2. using dataclasses.field and pydantic.Field in the same data model
    if dataclasses.is_dataclass(data_model):
        name_to_field = {field.name: field for field in dataclasses.fields(data_model)}
    elif issubclass(data_model, pydantic.BaseModel):
        name_to_field = data_model.model_fields
    else:
        raise TypeError(
            "data_model must be a dataclass or a Pydantic BaseModel. "
            f"Got {type(data_model)}"
        )
    fields_data = []
    for name, field in name_to_field.items():
        if isinstance(field, dataclasses.Field):
            # Idiosyncrasy: if a pydantic Field is used in a pydantic dataclass, then
            # field.default is a FieldInfo object instead of the field's default value.
            # And more importantly, field.annotation is NoneType
            if isinstance(field.default, _PydanticField):
                field.default.annotation = field.type
                field_data = _field_data_from_pydantic_field(name, field.default)
            else:
                field_data = _field_data_from_dataclass_field(name, field)
        elif isinstance(field, _PydanticField):
            field_data = _field_data_from_pydantic_field(name, field)
        else:
            raise TypeError(
                f"Each field must be a dataclass or Pydantic field. Got {type(field)}"
            )
        fields_data.append(field_data)
    return fields_data


def _tap_class(fields_data: Sequence[_FieldData]) -> Type[Tap]:
    class ArgParser(Tap):
        def configure(self):
            for field_data in fields_data:
                variable = field_data.name
                self._annotations[variable] = field_data.annotation
                self.class_variables[variable] = {
                    "comment": field_data.description or ""
                }
                if field_data.is_required:
                    kwargs = {}
                else:
                    kwargs = dict(required=False, default=field_data.default)
                self.add_argument(f"--{variable}", **kwargs)

    return ArgParser


def tap_class_from_data_model(data_model: Any) -> Type[Tap]:
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
    fields_data = _fields_data(data_model)
    return _tap_class(fields_data)
