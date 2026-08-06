from dataclasses import dataclass, fields, is_dataclass, asdict
from collections.abc import Mapping as ABCMapping
import inspect
from enum import Enum
import json
from typing import (
    Any,
    Optional,
    Tuple,
    Union,
    Type,
    Mapping,
    List,
    Dict,
    get_origin,
    get_args,
)


MDL_PROMPT = """
You are a JSON generator.
You will always receive input data, an instruction and a target schema in the form of MDL (Model Data Language).
Your task is to produce a JSON object that strictly follows the MDL schema.

Rules:
- Respond ONLY with a valid JSON object.
- Do not include explanations, notes, or formatting such as code fences.
- Do not include any text outside the JSON object.
- Ensure the output JSON matches the schema exactly.
- The values must be well-formed according to their type (e.g., str, int).
- When the schema defines string fields (e.g., code: str), the entire content must be returned as a string value inside JSON.
- All keys defined in the schema must always be included in the output JSON, even if their value is empty


INSTRUCTION:
{instruction}

INPUT_SCHEMA:
{input_schema}

OUTPUT_SCHEMA:
{output_schema}
"""


def build_mdl_system_prompt(
    instruction: str, input_generic: Any, output_generic: Any
) -> str:

    return MDL_PROMPT.format(
        instruction=instruction,
        input_schema="\n".join(compile_mdl(dataclass_generic=input_generic)),
        output_schema="\n".join(compile_mdl(dataclass_generic=output_generic)),
    )


_NONE = type(None)
_PRIMITIVES = {str, int, float, bool, _NONE}


def _is_union_origin(origin: Any) -> bool:
    return origin is Union


def _is_enum_type(target_type: Any) -> bool:
    return isinstance(target_type, type) and issubclass(target_type, Enum)


def _is_supported_string_enum(target_type: Any) -> bool:
    """
    MDL only supports enums declared in the following form:

        class SomeEnum(str, Enum):
            VALUE = "value"

    A plain Enum containing string values is intentionally not supported:

        class SomeEnum(Enum):
            VALUE = "value"
    """
    if not _is_enum_type(target_type):
        return False

    if not issubclass(target_type, str):
        return False

    return all(isinstance(member.value, str) for member in target_type)


def _enum_mdl(enum_type: Type[Enum]) -> str:
    """
    Render a string-backed enum using its allowed values.

    Example:

        class Status(str, Enum):
            ACTIVE = "active"
            PAUSED = "paused"

    Produces:

        Enum["active", "paused"]
    """
    if not _is_supported_string_enum(enum_type):
        raise TypeError(
            "MDL only supports string enums declared as `class SomeEnum(str, Enum)`"
        )

    enum_values = ", ".join(
        json.dumps(member.value, ensure_ascii=False) for member in enum_type
    )

    return f"Enum[{enum_values}]"


def compile_mdl(dataclass_generic: Any) -> List[str]:
    """
    Convert a Python dataclass into Model Data Language.

    Supported types:
      - str
      - int
      - float
      - bool
      - Optional
      - Union
      - List
      - Dict / Mapping
      - Tuple
      - Nested dataclasses
      - String-backed enums declared as `class SomeEnum(str, Enum)`

    Enum example:

        class Status(str, Enum):
            ACTIVE = "active"
            PAUSED = "paused"

        @dataclass
        class Result:
            status: Status

        compile_mdl(Result)

    Produces:

        ['status: Enum["active", "paused"]']

    Args:
        dataclass_generic:
            The dataclass type to compile.

    Returns:
        A list containing one MDL declaration per dataclass field.

    Raises:
        AssertionError:
            If the supplied object is not a dataclass.

        TypeError:
            If the schema contains an unsupported type.
    """
    assert is_dataclass(dataclass_generic), "dataclass_generic must be a dataclass type"

    def primitive_name(target_type: Any) -> str:
        if target_type is _NONE:
            raise TypeError("MDL does not support None as an explicit field type")

        return target_type.__name__

    def reduce_type(target_type: Any) -> str:
        origin = get_origin(target_type)
        args = get_args(target_type)

        # Union and Optional
        if _is_union_origin(origin):
            non_none_types = [argument for argument in args if argument is not _NONE]

            has_none = len(non_none_types) != len(args)

            reduced_parts = [reduce_type(argument) for argument in non_none_types]

            if has_none:
                if len(reduced_parts) == 1:
                    inner = reduced_parts[0]
                else:
                    inner = "Union[" + ", ".join(reduced_parts) + "]"

                return f"Optional[{inner}]"

            return "Union[" + ", ".join(reduced_parts) + "]"

        # List
        if origin in (list, List):
            (inner_type,) = args or (Any,)
            return f"List[{reduce_type(inner_type)}]"

        # Dict and Mapping
        if origin in (
            dict,
            Dict,
            Mapping,
            ABCMapping,
        ):
            key_type, value_type = (args + (Any, Any))[:2]

            return f"Dict[{reduce_type(key_type)}, {reduce_type(value_type)}]"

        # Tuple
        if origin in (tuple, Tuple):
            if args and args[-1] is Ellipsis:
                return f"Tuple[{reduce_type(args[0])}, ...]"

            if args:
                return (
                    "Tuple["
                    + ", ".join(reduce_type(argument) for argument in args)
                    + "]"
                )

            return "Tuple[Any, ...]"

        # Enum
        if _is_enum_type(target_type):
            return _enum_mdl(target_type)

        # Nested dataclass
        if isinstance(target_type, type) and is_dataclass(target_type):
            inner_fields = [
                f"{field.name}: {reduce_type(field.type)}"
                for field in fields(target_type)
            ]

            inner = ", ".join(inner_fields) if inner_fields else "/* empty */"

            return f"Dict{{{inner}}}"

        # Primitive leaves
        if target_type in _PRIMITIVES:
            return primitive_name(target_type)

        # Explicit Any remains unsupported.
        if target_type is Any:
            raise TypeError(
                "MDL does not support Any as a datatype; choose a concrete datatype"
            )

        raise TypeError(f"MDL does not support datatype {target_type!r}")

    return [
        f"{field.name}: {reduce_type(field.type)}"
        for field in fields(dataclass_generic)
    ]
