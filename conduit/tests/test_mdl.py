import pytest
from enum import Enum
from dataclasses import dataclass, fields, is_dataclass, asdict
from conduit.mdl import compile_mdl
from conduit.utils.deployment import parse_llm_json

from typing import (
    Any,
    Optional,
    Union,
    List,
    Dict,
)


class Status(str, Enum):
    """Allowed workflow states."""

    ACTIVE = "active"
    PAUSED = "paused"


class Priority(str, Enum):
    LOW = "low"
    HIGH = "high"


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
    @dataclass
    class H:
        bad: type(None)  # explicit None type is not allowed

    with pytest.raises(TypeError):
        compile_mdl(H)


def test_explicit_any_field_raises():

    @dataclass
    class I:
        bad: Any  # explicit Any type should be rejected

    with pytest.raises(TypeError):
        compile_mdl(I)


import pytest

from dataclasses import dataclass
from enum import Enum
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Union,
)

from conduit.mdl import compile_mdl


@dataclass
class Inner:
    x: int
    y: Optional[str]


class Status(str, Enum):
    """Allowed workflow states."""

    ACTIVE = "active"
    PAUSED = "paused"


class Priority(str, Enum):
    LOW = "low"
    HIGH = "high"


def test_basic_minimal_fields():
    @dataclass
    class A:
        a: int
        b: str
        c: float
        d: bool

    lines = compile_mdl(A)

    assert lines == [
        "a: int",
        "b: str",
        "c: float",
        "d: bool",
    ]


def test_optionals_and_unions_flat():
    @dataclass
    class B:
        e: Optional[int]
        f: Union[str, int]
        g: Union[int, None]
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


def test_unions_and_optionals_in_collections():
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
    @dataclass
    class H:
        bad: type(None)

    with pytest.raises(TypeError):
        compile_mdl(H)


def test_explicit_any_field_raises():
    @dataclass
    class I:
        bad: Any

    with pytest.raises(TypeError):
        compile_mdl(I)


def test_string_enum_is_rendered_with_metadata():
    @dataclass
    class EnumResult:
        status: Status

    lines = compile_mdl(EnumResult)

    assert lines == [
        "status: Enum{"
        'name: "Status", '
        'docstring: "Allowed workflow states.", '
        'value: ["active", "paused"], '
        'decorators: ["str", "Enum"]'
        "}"
    ]


def test_enum_without_docstring_uses_empty_string():
    @dataclass
    class EnumResult:
        priority: Priority

    lines = compile_mdl(EnumResult)

    assert lines == [
        "priority: Enum{"
        'name: "Priority", '
        'docstring: "", '
        'value: ["low", "high"], '
        'decorators: ["str", "Enum"]'
        "}"
    ]


def test_optional_enum():
    @dataclass
    class EnumResult:
        status: Optional[Status]

    lines = compile_mdl(EnumResult)

    enum_schema = (
        "Enum{"
        'name: "Status", '
        'docstring: "Allowed workflow states.", '
        'value: ["active", "paused"], '
        'decorators: ["str", "Enum"]'
        "}"
    )

    assert lines == [f"status: Optional[{enum_schema}]"]


def test_list_of_enums():
    @dataclass
    class EnumResult:
        statuses: List[Status]

    lines = compile_mdl(EnumResult)

    enum_schema = (
        "Enum{"
        'name: "Status", '
        'docstring: "Allowed workflow states.", '
        'value: ["active", "paused"], '
        'decorators: ["str", "Enum"]'
        "}"
    )

    assert lines == [f"statuses: List[{enum_schema}]"]


def test_dict_of_enums():
    @dataclass
    class EnumResult:
        statuses: Dict[str, Status]

    lines = compile_mdl(EnumResult)

    enum_schema = (
        "Enum{"
        'name: "Status", '
        'docstring: "Allowed workflow states.", '
        'value: ["active", "paused"], '
        'decorators: ["str", "Enum"]'
        "}"
    )

    assert lines == [f"statuses: Dict[str, {enum_schema}]"]


def test_enum_inside_nested_dataclass():
    @dataclass
    class EnumInner:
        status: Status

    @dataclass
    class EnumOuter:
        item: EnumInner

    lines = compile_mdl(EnumOuter)

    enum_schema = (
        "Enum{"
        'name: "Status", '
        'docstring: "Allowed workflow states.", '
        'value: ["active", "paused"], '
        'decorators: ["str", "Enum"]'
        "}"
    )

    assert lines == [f"item: Dict{{status: {enum_schema}}}"]


def test_plain_enum_is_rejected_even_with_string_values():
    class PlainEnum(Enum):
        ACTIVE = "active"

    @dataclass
    class InvalidResult:
        status: PlainEnum

    with pytest.raises(
        TypeError,
        match="class SomeEnum\\(str, Enum\\)",
    ):
        compile_mdl(InvalidResult)


def test_integer_enum_is_rejected():
    class NumberEnum(int, Enum):
        ONE = 1

    @dataclass
    class InvalidResult:
        number: NumberEnum

    with pytest.raises(
        TypeError,
        match="class SomeEnum\\(str, Enum\\)",
    ):
        compile_mdl(InvalidResult)


import pytest

from dataclasses import dataclass
from enum import Enum
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Union,
)

from conduit.mdl import compile_mdl


@dataclass
class Inner:
    x: int
    y: Optional[str]


class Status(str, Enum):
    """Allowed workflow states."""

    ACTIVE = "active"
    PAUSED = "paused"


class Priority(str, Enum):
    LOW = "low"
    HIGH = "high"


def test_basic_minimal_fields():
    @dataclass
    class A:
        a: int
        b: str
        c: float
        d: bool

    lines = compile_mdl(A)

    assert lines == [
        "a: int",
        "b: str",
        "c: float",
        "d: bool",
    ]


def test_optionals_and_unions_flat():
    @dataclass
    class B:
        e: Optional[int]
        f: Union[str, int]
        g: Union[int, None]
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


def test_unions_and_optionals_in_collections():
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
    @dataclass
    class H:
        bad: type(None)

    with pytest.raises(TypeError):
        compile_mdl(H)


def test_explicit_any_field_raises():
    @dataclass
    class I:
        bad: Any

    with pytest.raises(TypeError):
        compile_mdl(I)


def test_string_enum_is_rendered_with_metadata():
    @dataclass
    class EnumResult:
        status: Status

    lines = compile_mdl(EnumResult)

    assert lines == ['status: Enum["active", "paused"]']


def test_enum_without_docstring_uses_empty_string():
    @dataclass
    class EnumResult:
        priority: Priority

    lines = compile_mdl(EnumResult)

    assert lines == ['priority: Enum["low", "high"]']


def test_optional_enum():
    @dataclass
    class EnumResult:
        status: Optional[Status]

    lines = compile_mdl(EnumResult)

    assert lines == ['status: Optional[Enum["active", "paused"]]']


def test_list_of_enums():
    @dataclass
    class EnumResult:
        statuses: List[Status]

    lines = compile_mdl(EnumResult)

    assert lines == ['statuses: List[Enum["active", "paused"]]']


def test_dict_of_enums():
    @dataclass
    class EnumResult:
        statuses: Dict[str, Status]

    lines = compile_mdl(EnumResult)

    assert lines == ['statuses: Dict[str, Enum["active", "paused"]]']


def test_enum_inside_nested_dataclass():
    @dataclass
    class EnumInner:
        status: Status

    @dataclass
    class EnumOuter:
        item: EnumInner

    lines = compile_mdl(EnumOuter)

    assert lines == ['item: Dict{status: Enum["active", "paused"]}']


def test_plain_enum_is_rejected_even_with_string_values():
    class PlainEnum(Enum):
        ACTIVE = "active"

    @dataclass
    class InvalidResult:
        status: PlainEnum

    with pytest.raises(
        TypeError,
        match="class SomeEnum\\(str, Enum\\)",
    ):
        compile_mdl(InvalidResult)


def test_integer_enum_is_rejected():
    class NumberEnum(int, Enum):
        ONE = 1

    @dataclass
    class InvalidResult:
        number: NumberEnum

    with pytest.raises(
        TypeError,
        match="class SomeEnum\\(str, Enum\\)",
    ):
        compile_mdl(InvalidResult)


def test_string_enum_mdl():
    @dataclass
    class Result:
        status: Status

    lines = compile_mdl(Result)

    assert lines == ['status: Enum["active", "paused"]']


def test_optional_string_enum_mdl():
    @dataclass
    class Result:
        status: Optional[Status]

    lines = compile_mdl(Result)

    assert lines == ['status: Optional[Enum["active", "paused"]]']


def test_list_of_string_enums_mdl():
    @dataclass
    class Result:
        statuses: List[Status]

    lines = compile_mdl(Result)

    assert lines == ['statuses: List[Enum["active", "paused"]]']


def test_dict_of_string_enums_mdl():
    @dataclass
    class Result:
        statuses: Dict[str, Status]

    lines = compile_mdl(Result)

    assert lines == ['statuses: Dict[str, Enum["active", "paused"]]']


def test_nested_dataclass_with_string_enum_mdl():
    @dataclass
    class EnumInner:
        status: Status

    @dataclass
    class Result:
        inner: EnumInner

    lines = compile_mdl(Result)

    assert lines == ['inner: Dict{status: Enum["active", "paused"]}']


def test_plain_enum_is_not_supported():
    class PlainStatus(Enum):
        ACTIVE = "active"
        PAUSED = "paused"

    @dataclass
    class Result:
        status: PlainStatus

    with pytest.raises(TypeError):
        compile_mdl(Result)


def test_non_string_enum_is_not_supported():
    class NumericStatus(int, Enum):
        ACTIVE = 1
        PAUSED = 2

    @dataclass
    class Result:
        status: NumericStatus

    with pytest.raises(TypeError):
        compile_mdl(Result)


def test_parse_string_enum_value():
    @dataclass
    class Result:
        status: Status

    result = parse_llm_json(
        '{"status": "active"}',
        Result,
    )

    assert result.status is Status.ACTIVE
    assert result.status.value == "active"


def test_parse_optional_string_enum_value():
    @dataclass
    class Result:
        status: Optional[Status]

    result = parse_llm_json(
        '{"status": "paused"}',
        Result,
    )

    assert result.status is Status.PAUSED


def test_parse_optional_string_enum_null():
    @dataclass
    class Result:
        status: Optional[Status]

    result = parse_llm_json(
        '{"status": null}',
        Result,
    )

    assert result.status is None


def test_parse_list_of_string_enum_values():
    @dataclass
    class Result:
        statuses: List[Status]

    result = parse_llm_json(
        '{"statuses": ["active", "paused"]}',
        Result,
    )

    assert result.statuses == [
        Status.ACTIVE,
        Status.PAUSED,
    ]


def test_parse_dict_of_string_enum_values():
    @dataclass
    class Result:
        statuses: Dict[str, Status]

    result = parse_llm_json(
        """
        {
            "statuses": {
                "first": "active",
                "second": "paused"
            }
        }
        """,
        Result,
    )

    assert result.statuses == {
        "first": Status.ACTIVE,
        "second": Status.PAUSED,
    }


def test_parse_nested_string_enum_value():
    @dataclass
    class EnumInner:
        status: Status
        priority: Priority

    @dataclass
    class Result:
        inner: EnumInner

    result = parse_llm_json(
        """
        {
            "inner": {
                "status": "active",
                "priority": "high"
            }
        }
        """,
        Result,
    )

    assert result.inner.status is Status.ACTIVE
    assert result.inner.priority is Priority.HIGH


def test_parse_bare_enum_value_for_single_field_dataclass():
    @dataclass
    class Result:
        status: Status

    result = parse_llm_json(
        '"active"',
        Result,
    )

    assert result.status is Status.ACTIVE


def test_parse_invalid_enum_value_raises():
    @dataclass
    class Result:
        status: Status

    with pytest.raises(ValueError):
        parse_llm_json(
            '{"status": "unknown"}',
            Result,
        )


def test_parse_enum_member_name_does_not_coerce():
    @dataclass
    class Result:
        status: Status

    with pytest.raises(ValueError):
        parse_llm_json(
            '{"status": "ACTIVE"}',
            Result,
        )


def test_parse_non_string_enum_value_raises():
    @dataclass
    class Result:
        status: Status

    with pytest.raises(ValueError):
        parse_llm_json(
            '{"status": 1}',
            Result,
        )
