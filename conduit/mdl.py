from dataclasses import dataclass, fields, is_dataclass, asdict

from typing import (
    Any,
    Optional,
    Tuple,
    Union,
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

{input_schema}

OUTPUT:
{schema}
"""


def build_mdl_input(input_generic: Any):
    output = "\n".join(compile_mdl(dataclass_generic=input_generic))
    return "INPUT_SCHEMA: \n\n" + output


def build_mdl_system_prompt(
    instruction: str, input_generic: Any, output_generic: Any
) -> str:

    input = build_mdl_input(input_generic=input_generic)

    return MDL_PROMPT.format(
        instruction=instruction,
        input_schema=input,
        schema="\n".join(compile_mdl(dataclass_generic=output_generic)),
    )


_NONE = type(None)
_PRIMITIVES = {str, int, float, bool, _NONE}


def compile_mdl(dataclass_generic: Any) -> List[str]:
    """
    Convert a Python dataclass into Model Data Language (MDL).

    Model Data Language (MDL) is a schema-like representation of the
    shape and types of structured data. It provides a human-readable,
    standardized way of describing Python data models based on type
    hints and dataclass fields.

    Key ideas:
      - **Primitives** like str, int, float, bool are preserved directly.
      - **Composites** (List, Dict, Tuple, Union/Optional) are expanded
        into a consistent textual form.
      - **Nested dataclasses** are unfolded into dictionary-like
        structures that expose their inner fields.
      - **Unknowns (Any)** are represented as `Any`.

    This makes MDL useful for:
      - Documenting data structures clearly.
      - Sharing schema definitions outside of Python.
      - Ensuring consistent type-safe mappings for configs, APIs, or ML models.

    Example:
        >>> from dataclasses import dataclass
        >>> from typing import List, Optional

        >>> @dataclass
        ... class User:
        ...     id: int
        ...     name: str
        ...     tags: Optional[List[str]]

        >>> compile_mdl(User)
        ['id: int', 'name: str', 'tags: Optional[List[str]]']

    Returns:
        List[str]: A list of strings, one per dataclass field,
                   with its MDL type description.
    """
    assert is_dataclass(dataclass_generic), "output_generic must be a dataclass type"

    def is_prim(t) -> bool:
        return t in _PRIMITIVES

    def prim_name(t) -> str:
        if t is _NONE:
            raise TypeError("MDL does not support None as a explict type")
        else:
            return t.__name__

    def reduce_type(t) -> str:
        origin = get_origin(t)
        args = get_args(t)

        if origin is Union:
            non_none = [a for a in args if a is not _NONE]
            parts = [reduce_type(a) for a in non_none]
            has_none = len(non_none) != len(args)

            if has_none:
                inner = (
                    parts[0] if len(parts) == 1 else "Union[" + ", ".join(parts) + "]"
                )
                return f"Optional[{inner}]"

            return "Union[" + ", ".join(parts) + "]"

        # List
        if origin in (list, List):
            (inner,) = args or (Any,)
            return f"List[{reduce_type(inner)}]"

        # Dict / Mapping
        if origin in (dict, Dict, Mapping):
            k, v = (args + (Any, Any))[:2]
            return f"Dict[{reduce_type(k)}, {reduce_type(v)}]"

        # Tuple
        if origin in (tuple, Tuple):
            if args and args[-1] is Ellipsis:
                return f"Tuple[{reduce_type(args[0])}, ...]"
            if args:
                return "Tuple[" + ", ".join(reduce_type(a) for a in args) + "]"
            return "Tuple[Any, ...]"

        # Dataclass -> Dict{field: type, ...}
        if isinstance(t, type) and is_dataclass(t):
            # render each field with reduced type
            items = []
            for f in fields(t):
                items.append(f"{f.name}: {reduce_type(f.type)}")
            inner = ", ".join(items) if items else "/* empty */"
            return f"Dict{{{inner}}}"

        # Primitive leaves
        if is_prim(t):
            return prim_name(t)

        # Any / unknown -> Any
        if t is Any:
            raise TypeError(
                "MDL does not supprt Any as a datatype please choose a valid datatype"
            )
        return ""

    lines = []
    for f in fields(dataclass_generic):
        lines.append(f"{f.name}: {reduce_type(f.type)}")

    return lines


### TESTS ###


@dataclass
class Inner:
    x: int
    y: Optional[str]


def test_basic_minimal_fields():
    @dataclass
    class A:
        a: int
        b: str
        c: float
        d: bool

    lines = compile_mdl(A)
    assert lines == ["a: int", "b: str", "c: float", "d: bool"]


def test_optionals_and_unions_flat():
    @dataclass
    class B:
        e: Optional[int]
        f: Union[str, int]
        g: Union[int, None]  # equivalent to Optional[int]
        h: Optional[str]

    lines = compile_mdl(B)
    assert lines == [
        "e: Optional[int]",
        "f: Union[str, int]",
        "g: Optional[int]",
        "h: Optional[str]",
    ]


def test_lists_of_primitives_and_inner():
    @dataclass
    class C:
        h: List[str]
        i: List[int]
        j: List[Inner]

    lines = compile_mdl(C)
    assert lines == [
        "h: List[str]",
        "i: List[int]",
        "j: List[Dict{x: int, y: Optional[str]}]",
    ]


def test_dicts_of_primitive_and_inner():
    @dataclass
    class D:
        k: Dict[str, int]
        l: Dict[str, Inner]
        m: Dict[str, Optional[Inner]]

    lines = compile_mdl(D)
    assert lines == [
        "k: Dict[str, int]",
        "l: Dict[str, Dict{x: int, y: Optional[str]}]",
        "m: Dict[str, Optional[Dict{x: int, y: Optional[str]}]]",
    ]


def test_nested_list_dict_mixes_with_allowed_types():
    @dataclass
    class E:
        a: List[Dict[str, int]]
        b: Dict[str, List[int]]
        c: List[Dict[str, Inner]]
        d: Dict[str, List[Inner]]
        e: List[Dict[str, Optional[Inner]]]
        f: Dict[str, List[Optional[Inner]]]

    lines = compile_mdl(E)
    assert lines == [
        "a: List[Dict[str, int]]",
        "b: Dict[str, List[int]]",
        "c: List[Dict[str, Dict{x: int, y: Optional[str]}]]",
        "d: Dict[str, List[Dict{x: int, y: Optional[str]}]]",
        "e: List[Dict[str, Optional[Dict{x: int, y: Optional[str]}]]]",
        "f: Dict[str, List[Optional[Dict{x: int, y: Optional[str]}]]]",
    ]


def test_deeper_nesting_still_within_allowed_palette():
    @dataclass
    class F:
        a: List[List[int]]
        b: Dict[str, Dict[str, int]]
        c: Dict[str, Dict[str, Inner]]
        d: List[Dict[str, List[int]]]
        e: List[Dict[str, List[Inner]]]

    lines = compile_mdl(F)
    assert lines == [
        "a: List[List[int]]",
        "b: Dict[str, Dict[str, int]]",
        "c: Dict[str, Dict[str, Dict{x: int, y: Optional[str]}]]",
        "d: List[Dict[str, List[int]]]",
        "e: List[Dict[str, List[Dict{x: int, y: Optional[str]}]]]",
    ]


def test_unions_and_optionals_in_collections_but_only_allowed_variants():
    @dataclass
    class G:
        a: List[Union[str, int]]
        b: Dict[str, Union[str, int]]
        c: Optional[Union[str, int]]
        d: List[Optional[int]]
        e: Dict[str, Optional[int]]
        f: List[Optional[Inner]]
        g: Dict[str, List[Optional[int]]]

    lines = compile_mdl(G)
    assert lines == [
        "a: List[Union[str, int]]",
        "b: Dict[str, Union[str, int]]",
        "c: Optional[Union[str, int]]",
        "d: List[Optional[int]]",
        "e: Dict[str, Optional[int]]",
        "f: List[Optional[Dict{x: int, y: Optional[str]}]]",
        "g: Dict[str, List[Optional[int]]]",
    ]


def test_original_outer_shape_from_prompt():
    @dataclass
    class Outer:
        a: int
        b: str
        c: float
        d: bool
        e: Optional[int]
        f: Union[str, int]
        g: Union[int, None]
        h: List[str]
        i: List[int]
        j: List[Inner]
        k: Dict[str, int]
        l: Dict[str, Inner]
        m: Dict[str, Optional[Inner]]

    lines = compile_mdl(Outer)
    assert lines == [
        "a: int",
        "b: str",
        "c: float",
        "d: bool",
        "e: Optional[int]",
        "f: Union[str, int]",
        "g: Optional[int]",
        "h: List[str]",
        "i: List[int]",
        "j: List[Dict{x: int, y: Optional[str]}]",
        "k: Dict[str, int]",
        "l: Dict[str, Dict{x: int, y: Optional[str]}]",
        "m: Dict[str, Optional[Dict{x: int, y: Optional[str]}]]",
    ]


def test_explicit_none_field_raises():
    import pytest

    @dataclass
    class H:
        bad: type(None)  # explicit None type is not allowed

    with pytest.raises(TypeError):
        compile_mdl(H)


def test_explicit_any_field_raises():
    import pytest

    @dataclass
    class I:
        bad: Any  # explicit Any type should be rejected

    with pytest.raises(TypeError):
        compile_mdl(I)
