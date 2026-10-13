from dataclasses import dataclass, fields, is_dataclass, asdict
from collections.abc import Mapping as ABCMapping
import inspect
from enum import Enum
import json
import types
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
    get_type_hints,
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
    # UnionType is present on Python 3.10+, for annotations such as str | None.
    return origin is Union or origin is getattr(types, "UnionType", None)


def _is_enum_type(target_type: Any) -> bool:
    return isinstance(target_type, type) and issubclass(target_type, Enum)


def _is_supported_string_enum(target_type: Any) -> bool:
    """
    MDL only supports enums declared in the following form:

        class SomeEnum(str, Enum):
            VALUE = "value"

    A plain Enum containing string values is intentionally not supported.
    """
    if not _is_enum_type(target_type):
        return False
    if not issubclass(target_type, str):
        return False
    return all(isinstance(member.value, str) for member in target_type)


def _enum_mdl(enum_type: Type[Enum]) -> str:
    """Render a string-backed enum as Enum["value1", "value2"]."""
    if not _is_supported_string_enum(enum_type):
        raise TypeError(
            "MDL only supports string enums declared as `class SomeEnum(str, Enum)`"
        )

    enum_values = ", ".join(
        json.dumps(member.value, ensure_ascii=False) for member in enum_type
    )
    return f"Enum[{enum_values}]"


def _get_resolved_hints(dataclass_type: Any):
    """Resolve postponed annotations, even for some dynamically loaded modules.

    Python normally looks up forward references in the defining module via
    sys.modules. Dynamically loaded code may have a live globals dictionary
    without a registered module. In that case, try the call-site namespace
    belonging to the exact dataclass being compiled.

    An unavailable namespace cannot always be reconstructed from a class;
    unresolvable forward references still raise NameError.
    """
    try:
        return get_type_hints(dataclass_type)
    except NameError:
        pass

    frame = inspect.currentframe()
    try:
        frame = frame.f_back
        while frame is not None:
            for namespace in (frame.f_globals, frame.f_locals):
                if namespace.get(dataclass_type.__name__) is not dataclass_type:
                    continue
                # Local names can also contain forward-referenced types.
                resolved_namespace = dict(frame.f_globals)
                resolved_namespace.update(frame.f_locals)
                try:
                    return get_type_hints(
                        dataclass_type,
                        globalns=resolved_namespace,
                        localns=resolved_namespace,
                    )
                except NameError:
                    continue
            frame = frame.f_back
    finally:
        del frame

    # Preserve the original exception and message for missing references.
    return get_type_hints(dataclass_type)


def _resolved_dataclass_fields(dataclass_type: Any):
    hints = _get_resolved_hints(dataclass_type)
    for field in fields(dataclass_type):
        yield field.name, hints[field.name]


def compile_mdl(dataclass_generic: Any) -> List[str]:
    """
    Convert a Python dataclass into Model Data Language.

    Supported types:
      - str, int, float, bool
      - Optional / Union (including T | None and T | U on Python 3.10+)
      - List, Dict / Mapping, Tuple
      - Nested dataclasses
      - String-backed enums declared as class SomeEnum(str, Enum)

    Resolves postponed annotations created by
    ``from __future__ import annotations`` using get_type_hints.

    Returns a list containing one MDL declaration per dataclass field.

    Raises:
        AssertionError: If the supplied object is not a dataclass.
        TypeError: If a schema contains an unsupported type.
        NameError: If an annotation refers to a name that cannot be resolved.
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
        if origin in (dict, Dict, Mapping, ABCMapping):
            key_type, value_type = (args + (Any, Any))[:2]
            return f"Dict[{reduce_type(key_type)}, {reduce_type(value_type)}]"

        # Tuple
        if origin in (tuple, Tuple):
            if args and args[-1] is Ellipsis:
                return f"Tuple[{reduce_type(args[0])}, ...]"
            if args:
                return "Tuple[" + ", ".join(reduce_type(a) for a in args) + "]"
            return "Tuple[Any, ...]"

        # Enum
        if _is_enum_type(target_type):
            return _enum_mdl(target_type)

        # Nested dataclass
        if isinstance(target_type, type) and is_dataclass(target_type):
            inner_fields = [
                f"{name}: {reduce_type(field_type)}"
                for name, field_type in _resolved_dataclass_fields(target_type)
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
        f"{name}: {reduce_type(field_type)}"
        for name, field_type in _resolved_dataclass_fields(dataclass_generic)
    ]
