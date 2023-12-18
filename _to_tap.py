"""
Convert a data model to a typed CLI argument parser.

TODO: get this to work for Python 3.8, 3.9, and Pydantic v1.
"""

import dataclasses
from typing import Any, Sequence

import pydantic
from tap import Tap


_PydanticField = pydantic.fields.FieldInfo | pydantic.dataclasses.FieldInfo


@dataclasses.dataclass(frozen=True)
class _FieldData:
    """
    Data about a field which is sufficient to inform a Tap variable/argument.
    """

    name: str
    annotation: type
    is_required: bool
    default: Any
    description: str | None = ""


def _field_data_from_dataclass(name: str, field: dataclasses.Field) -> _FieldData:
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


def _field_data_from_pydantic(
    name: str, field: _PydanticField, annotation: type | None = None
) -> _FieldData:
    annotation = field.annotation if annotation is None else annotation
    return _FieldData(
        name, annotation, field.is_required(), field.default, field.description
    )


def _fields_data(data_model: Any) -> list[_FieldData]:
    if dataclasses.is_dataclass(data_model):
        # This condition also holds for a Pydantic dataclass instance or model
        name_to_field = {field.name: field for field in dataclasses.fields(data_model)}
    elif isinstance(data_model, pydantic.BaseModel) or issubclass(
        data_model, pydantic.BaseModel
    ):  # Check isinstance before issubclass. issubclass requires data_model is a class
        name_to_field = data_model.model_fields
    else:
        raise TypeError(
            "data_model must be a builtin or Pydantic dataclass (instance or class) or "
            f"a Pydantic BaseModel (instance or class). Got {type(data_model)}"
        )
    # It's possible to mix fields w/ classes, e.g., use pydantic Fields in a (builtin)
    # dataclass, or use (builtin) dataclass fields in a pydantic BaseModel. It's also
    # possible to use (builtin) dataclass fields and pydantic Fields in the same data
    # model. Therefore, the type of the data model doesn't determine the type of each
    # field. The solution is to iterate through the fields and check each type.
    fields_data: list[_FieldData] = []
    for name, field in name_to_field.items():
        if isinstance(field, dataclasses.Field):
            # Idiosyncrasy: if a pydantic Field is used in a pydantic dataclass, then
            # field.default is a FieldInfo object instead of the field's default value.
            # Furthermore, field.annotation is always NoneType. Luckily, the actual type
            # of the field is stored in field.type
            if isinstance(field.default, _PydanticField):
                field_data = _field_data_from_pydantic(
                    name, field.default, annotation=field.type
                )
            else:
                field_data = _field_data_from_dataclass(name, field)
        elif isinstance(field, _PydanticField):
            field_data = _field_data_from_pydantic(name, field)
        else:
            raise TypeError(
                f"Each field must be a dataclass or Pydantic field. Got {type(field)}"
            )
        fields_data.append(field_data)
    return fields_data


def _tap_class(fields_data: Sequence[_FieldData]) -> type[Tap]:
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


def tap_class_from_data_model(data_model: Any) -> type[Tap]:
    """
    Convert a data model to a typed CLI argument parser.

    Parameters
    ----------
    data_model : Any
        a builtin or Pydantic dataclass (class or instance) or Pydantic `BaseModel`
        (class or instance)

    Returns
    -------
    type[Tap]
        a typed CLI argument parser class

    Note
    ----
    For a `data_model` containing builtin dataclass `field`s, argument descriptions are
    set to the `field`'s `metadata["description"]`.

    For example::

        @dataclass
        class Data:
            my_field: str = field(metadata={"description": "field description})
    """
    fields_data = _fields_data(data_model)
    return _tap_class(fields_data)
