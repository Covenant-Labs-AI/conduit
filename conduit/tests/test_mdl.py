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


# --- SVM Conduit typed-boundary regression tests ---
"""Regression tests for the 12 `request` / 32 `instruction` SVM failures.

Run from a checkout/environment with the REAL Conduit package installed:
    python -m pytest -q test_conduit_svm_boundary.py

These tests do not contact a model server. The typed Conduit call is exercised
up to and including JSON decoding with `_completion` replaced by a fake.

Optionally set SVM_SOURCE_PATH=/path/to/svm_upgrade_d.py to also test the exact
real input/output classes from SVM, not just minimal reproductions.
"""

import ast
import importlib.util
import os
import pathlib
from dataclasses import asdict, dataclass, fields
from typing import Dict, List, Optional
from unittest.mock import patch

import pytest


# These tests run even when Conduit is not installed, and establish that Python
# dataclass inheritance alone does not delete inherited fields.
@dataclass
class IntentBase:
    request: str
    candidate_edit_files: List[str]


@dataclass
class IdentityIntent(IntentBase):
    identity_candidates: List[str]


@dataclass
class EditorBase:
    instruction: str
    file_path: str


@dataclass
class RealizationEditor(EditorBase):
    realization_identities: List[str]


@dataclass
class IntentResult:
    objective: str
    constraints: List[str]


@dataclass
class FileEdit:
    path: str
    edits: List[str]


@dataclass
class EditorResult:
    task_analysis: str
    files: List[FileEdit]


@pytest.mark.parametrize(
    ("input_type", "field_name"),
    [(IdentityIntent, "request"), (RealizationEditor, "instruction")],
)
def test_stdlib_inherited_fields_survive_asdict(input_type, field_name):
    if input_type is IdentityIntent:
        obj = IdentityIntent("change target", ["main.py"], ["GRE-001"])
    else:
        obj = RealizationEditor("replace method", "main.py", ["GRO-01"])
    assert field_name in {f.name for f in fields(input_type)}
    assert asdict(obj)[field_name] == getattr(obj, field_name)


def _conduit():
    # Skip only the real-runtime-dependent tests when Conduit isn't installed;
    # don't manufacture a passing substitute for the actual implementation.
    pytest.importorskip("conduit.mdl", reason="Requires the real Conduit checkout")
    from conduit.mdl import build_mdl_system_prompt, compile_mdl
    from conduit.utils import dataclass_to_dict
    from conduit.blocks import OpenAICompatibleRuntimeBlock

    return (
        build_mdl_system_prompt,
        compile_mdl,
        dataclass_to_dict,
        OpenAICompatibleRuntimeBlock,
    )


@pytest.mark.parametrize(
    ("obj", "field_name"),
    [
        (IdentityIntent("change target", ["main.py"], ["GRE-001"]), "request"),
        (RealizationEditor("replace method", "main.py", ["GRO-01"]), "instruction"),
    ],
)
def test_actual_conduit_serialization_preserves_inherited_key(obj, field_name):
    _, compile_mdl, dataclass_to_dict, _ = _conduit()
    schema = compile_mdl(type(obj))
    serialized = dataclass_to_dict(obj)
    assert field_name in [line.split(":", 1)[0] for line in schema]
    assert isinstance(serialized, dict), type(serialized)
    assert field_name in serialized, (field_name, serialized)
    assert serialized[field_name] == getattr(obj, field_name)


@pytest.mark.parametrize(
    ("obj", "result_type", "field_name"),
    [
        (
            IdentityIntent("change target", ["main.py"], ["GRE-001"]),
            IntentResult,
            "request",
        ),
        (
            RealizationEditor("replace method", "main.py", ["GRO-01"]),
            EditorResult,
            "instruction",
        ),
    ],
)
def test_mdl_prompt_contains_inherited_input_field(obj, result_type, field_name):
    build_prompt, _, _, _ = _conduit()
    prompt = build_prompt("test typed call", obj, result_type)
    assert f"{field_name}: str" in prompt
    assert "OUTPUT_SCHEMA:" in prompt
    assert "INPUT_SCHEMA:" in prompt


class _FakeCompletion:
    def __init__(self, content: str):
        self.content = content
        self.reasoning = None
        self.finish_reason = "stop"

    def require_content(self) -> str:
        return self.content


def _run_no_network_typed_call(obj, result_type, raw_json: str, expected_field: str):
    _, _, _, Block = _conduit()
    block = Block(
        api_key="test", host="127.0.0.1", scheme="http", api_mode="chat_completions"
    )
    captured = []

    def fake_completion(*, model_id, messages, system_message):
        assert model_id == "test-model"
        assert expected_field in messages[0]["content"]
        assert f"{expected_field}: str" in system_message
        captured.append((messages, system_message))
        return _FakeCompletion(raw_json)

    with patch.object(block, "_completion", side_effect=fake_completion):
        result = block("test-model", input=obj, output=result_type, guidance="test")

    assert len(captured) == 1, "Must reach the model transport boundary exactly once"
    assert isinstance(result, result_type)
    return result


@pytest.mark.parametrize(
    ("obj", "result_type", "raw_json", "expected_field"),
    [
        (
            IdentityIntent("change target", ["main.py"], ["GRE-001"]),
            IntentResult,
            '{"objective":"change target","constraints":[]}',
            "request",
        ),
        (
            RealizationEditor("replace method", "main.py", ["GRO-01"]),
            EditorResult,
            '{"task_analysis":"done","files":[]}',
            "instruction",
        ),
    ],
)
def test_full_typed_conduit_call_without_http(
    obj, result_type, raw_json, expected_field
):
    _run_no_network_typed_call(obj, result_type, raw_json, expected_field)


def test_alternating_intent_and_editor_many_times_no_hidden_state():
    # Late failures (variants 7/8) warrant checking mixed calls through one
    # long-lived runtime object, in addition to isolated first-call success.
    _, _, _, Block = _conduit()
    block = Block(
        api_key="test", host="127.0.0.1", scheme="http", api_mode="chat_completions"
    )
    count = 0

    def fake_completion(*, model_id, messages, system_message):
        nonlocal count
        count += 1
        if "request':" in messages[0]["content"]:
            assert "request: str" in system_message
            return _FakeCompletion('{"objective":"ok","constraints":[]}')
        assert "instruction':" in messages[0]["content"]
        assert "instruction: str" in system_message
        return _FakeCompletion('{"task_analysis":"ok","files":[]}')

    with patch.object(block, "_completion", side_effect=fake_completion):
        for i in range(128):
            if i % 2 == 0:
                result = block(
                    "test-model",
                    input=IdentityIntent(f"r{i}", [], []),
                    output=IntentResult,
                    guidance="test",
                )
                assert isinstance(result, IntentResult)
            else:
                result = block(
                    "test-model",
                    input=RealizationEditor(f"e{i}", "main.py", []),
                    output=EditorResult,
                    guidance="test",
                )
                assert isinstance(result, EditorResult)
    assert count == 128


# Optional integration using SVM's exact production dataclasses, including its
# nested authority, anchors, policies, and enum types. This is more decisive
# than testing only a small mirror of those types.
@pytest.fixture(scope="module")
def actual_svm():
    _conduit()
    source = os.environ.get("SVM_SOURCE_PATH")
    if not source:
        pytest.skip("Set SVM_SOURCE_PATH to run the exact-SVM-schema tests")
    path = pathlib.Path(source)
    if not path.is_file():
        pytest.fail(f"SVM_SOURCE_PATH missing: {path}")
    import sys

    module_name = "svm_conduit_boundary_under_test"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _actual_intent(svm):
    authority = svm.ChangeAuthority(
        request="Rename A to B",
        explicitly_named_files=["main.py"],
        allowed_write_files=["main.py"],
        prohibited_write_files=[],
        write_scope_mode=svm.WriteScopeMode.EXPLICIT_EXCLUSIVE,
        exclusive_file_scope=True,
        explicitly_named_symbols=["A"],
        referenced_symbols=[],
        introduced_symbols=[],
        allowed_target_symbols=["A"],
        prohibited_target_symbols=[],
        target_scope_mode=svm.TargetScopeMode.EXPLICIT_EXCLUSIVE,
        exclusive_target_scope=True,
    )
    return svm.IdentityChangeIntentInput(
        request="Rename A to B",
        authority=authority,
        candidate_edit_files=["main.py"],
        advisory_context_files=[],
        semantic_anchors=[],
        identity_candidates=[],
    )


def _actual_editor(svm):
    return svm.RealizationEditorModelInput(
        instruction="Implement the change",
        file_path="main.py",
        acceptance_criteria=[],
        expected_source_units=[],
        unresolved_source_units=[],
        local_context=[],
        cross_file_context=[],
        dependency_evidence=[],
        allow_file_no_change=False,
        units=[],
        symbols=[],
        anchors=[],
        introduced_targets=[],
        transaction_guarantees=[],
        recommended_operations=[],
        identity_transitions=[],
    )


@pytest.mark.parametrize(
    ("factory", "field"),
    [(_actual_intent, "request"), (_actual_editor, "instruction")],
)
def test_exact_svm_model_input_schema_and_serialization(actual_svm, factory, field):
    build_prompt, compile_mdl, dataclass_to_dict, _ = _conduit()
    obj = factory(actual_svm)
    schema = compile_mdl(type(obj))
    serialized = dataclass_to_dict(obj)
    assert field in serialized
    assert serialized[field] == getattr(obj, field)
    assert f"{field}: str" in schema
    assert f"{field}: str" in build_prompt("test", obj, IntentResult)


def test_exact_svm_change_intent_typed_call(actual_svm):
    obj = _actual_intent(actual_svm)
    # Use the production output class, including all required arrays.
    raw = (
        '{"objective":"x","constraints":[],"completion_condition":"done",'
        '"transition_requirement_modes":[],"transition_requirement_applicability_selectors":[],'
        '"transition_requirement_target_selectors":[],"transition_requirement_effects":[],'
        '"transition_requirement_source_anchor_ids":[],"identity_transitions":[]}'
    )
    _run_no_network_typed_call(
        obj, actual_svm.IdentityChangeIntentProposal, raw, "request"
    )


def test_exact_svm_editor_typed_call(actual_svm):
    obj = _actual_editor(actual_svm)
    raw = '{"task_analysis":"x","files":[],"outcome":"apply","evidence":""}'
    _run_no_network_typed_call(
        obj, actual_svm.GenericRealizationUpdateResult, raw, "instruction"
    )


def test_exact_svm_schemas_in_alternating_cycle(actual_svm):
    build_prompt, compile_mdl, dataclass_to_dict, _ = _conduit()
    intent = _actual_intent(actual_svm)
    editor = _actual_editor(actual_svm)
    for i in range(100):
        obj = intent if i % 2 == 0 else editor
        field = "request" if i % 2 == 0 else "instruction"
        assert field in dataclass_to_dict(obj)
        assert field in {line.split(":", 1)[0] for line in compile_mdl(type(obj))}
        assert f"{field}: str" in build_prompt("test", obj, IntentResult)
