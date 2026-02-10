import pytest
from dataclasses import dataclass, fields, is_dataclass, asdict
from conduit.mdl import compile_mdl

from typing import (
    Any,
    Optional,
    Union,
    List,
    Dict,
)


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
