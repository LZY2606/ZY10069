#
# Copyright 2021-2025 WhiteMech
#
# ------------------------------
#
# This file is part of pddl.
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
#

"""Internal machinery shared by the typed-list parsing entry points.

This module is the single place where the PDDL typed-list syntax is grouped
into (names, type tags) pairs and where those pairs are normalized/validated
against shared boundary rules:

- untyped names belong to the default (untyped) group;
- one or more consecutive names followed by ``- type`` share that type;
- keywords keep their current case sensitivity ("object" is the only keyword
  allowed as a type);
- input order is preserved;
- duplicates follow the same policy for every entry point.

The grouping is purely syntactic. Semantic checks (unknown types, cycles in
the type hierarchy) are collected with source locations and reported later,
sorted by source position, so that more specific errors are never masked.
"""

import itertools
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    FrozenSet,
    Generic,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
)

from lark import Token

from pddl.custom_types import name, parse_name, parse_type
from pddl.helpers.base import check, safe_index
from pddl.logic.functions import NumericFunction
from pddl.logic.terms import _print_tag_set
from pddl.parser.symbols import Symbols

T = TypeVar("T", name, NumericFunction)


@dataclass(frozen=True)
class SourceLocation:
    """A position of a token in the user-provided PDDL text."""

    line: int
    column: int
    snippet: str

    def format(self) -> str:
        """Format the location for diagnostics."""
        return f"at line {self.line}, column {self.column}: '{self.snippet}'"


@dataclass(frozen=True)
class TypedListItem(Generic[T]):
    """A single name/function grouped with the type tags that follow it."""

    value: T
    type_tags: FrozenSet[name]
    location: Optional[SourceLocation]
    type_location: Optional[SourceLocation]


@dataclass
class TypedListDiagnostic(ValueError):
    """
    A single typed-list failure, anchored to the user input.

    The message is exactly the one that used to be raised by the old code
    paths, so the public exception text is unchanged.
    """

    message: str
    location: Optional[SourceLocation] = None

    def __str__(self) -> str:
        """Get the diagnostic string."""
        if self.location is None:
            return self.message
        return f"{self.message} ({self.location.format()})"


def _location_of(token: Any) -> Optional[SourceLocation]:
    """Extract a source location from a Lark token, if available."""
    if not isinstance(token, Token) or token.line is None or token.column is None:
        return None
    return SourceLocation(
        line=token.line,
        column=token.column,
        snippet=str(token),
    )


def _type_location_of(parent_names: Any) -> Optional[SourceLocation]:
    """Extract the location of the first token of a type group."""
    candidates = parent_names if isinstance(parent_names, list) else [parent_names]
    for candidate in candidates:
        location = _location_of(candidate)
        if location is not None:
            return location
    return None


def group_typed_list(
    tokens: Sequence[Union[Any, List[Any]]],
) -> List[
    Tuple[Any, FrozenSet[Any], Optional[SourceLocation], Optional[SourceLocation]]
]:
    """
    Group typed-list tokens into (item, raw_type_tags, location, type_location).

    The grouping rules are the same for names, variables, constants, objects and
    function skeletons:

    - a list without the ``-`` separator is one untyped group;
    - otherwise, every ``names - types`` block consumes the names before it;
    - a trailing block without a type is untyped.

    No name normalization happens here, only the shared syntactic grouping.
    """
    groups: List[
        Tuple[Any, FrozenSet[Any], Optional[SourceLocation], Optional[SourceLocation]]
    ] = []

    sep_value = Symbols.TYPE_SEP.value
    type_sep_index = safe_index(tokens, sep_value)
    if type_sep_index is None:
        # simple list of names, all untyped
        groups.extend((item, frozenset(), _location_of(item), None) for item in tokens)
        return groups

    start_index = 0
    end_index = len(tokens)
    while type_sep_index is not None:
        parent_names = tokens[type_sep_index + 1]
        if not isinstance(parent_names, list):
            parent_names = [parent_names]

        type_location = _type_location_of(parent_names)
        raw_tags = frozenset(parent_names)
        for item_name in itertools.islice(tokens, start_index, type_sep_index):
            groups.append((item_name, raw_tags, _location_of(item_name), type_location))

        start_index = type_sep_index + 2
        type_sep_index = safe_index(tokens, sep_value, start_index, end_index)

    # the last sublist might have no parent type
    if start_index != end_index:
        for item_name in itertools.islice(tokens, start_index, end_index):
            groups.append((item_name, frozenset(), _location_of(item_name), None))

    return groups


def normalize_typed_item(
    item_name: Any, raw_type_tags: FrozenSet[Any]
) -> Tuple[T, FrozenSet[name]]:
    """
    Normalize one grouped item and its type tags.

    This performs the keyword/regular-expression checks shared by every entry
    point (via :func:`parse_name`/:func:`parse_type`); it does no semantic
    check against the declared types of a domain.
    """
    check(
        isinstance(item_name, str) or isinstance(item_name, NumericFunction),
        f"invalid item '{item_name}' in typed list",
    )
    value: Any = parse_name(item_name) if isinstance(item_name, str) else item_name
    type_tags = frozenset(map(parse_type, raw_type_tags))
    return value, type_tags


def iter_typed_items(
    tokens: Sequence[Union[Any, List[Any]]],
) -> List[TypedListItem]:
    """Group typed-list tokens and normalize every item (shared entry point)."""
    items: List[TypedListItem] = []
    for raw_item, raw_tags, location, type_location in group_typed_list(tokens):
        value, type_tags = normalize_typed_item(raw_item, raw_tags)
        items.append(
            TypedListItem(
                value=value,
                type_tags=type_tags,
                location=location,
                type_location=type_location,
            )
        )
    return items


def make_duplicate_name_diagnostic(
    item_name: T, known_tags: FrozenSet[name], location: Optional[SourceLocation]
) -> TypedListDiagnostic:
    """Build the historical 'duplicate name' diagnostic."""
    if len(known_tags) > 0:
        message = (
            f"duplicate name '{item_name}' in typed list already inherits from "
            f"types {sorted(map(str, known_tags))}"
        )
    else:
        message = f"duplicate name '{item_name}' in typed list already present"
    return TypedListDiagnostic(message, location)


def make_duplicate_tag_diagnostic(
    item_name: T, type_tag: name, location: Optional[SourceLocation]
) -> TypedListDiagnostic:
    """Build the historical 'duplicate type tag' diagnostic."""
    return TypedListDiagnostic(
        f"duplicate type tag '{type_tag}' in typed list: type already specified "
        f"for item {item_name}",
        location,
    )


def make_inconsistent_tags_diagnostic(
    item_name: T,
    previous_tags: FrozenSet[name],
    new_tags: FrozenSet[name],
    location: Optional[SourceLocation],
) -> TypedListDiagnostic:
    """Build the historical 'invalid types for item' diagnostic."""
    return TypedListDiagnostic(
        f"invalid types for item '{item_name}': previous known tags were "
        f"{_print_tag_set(previous_tags)}, got {_print_tag_set(new_tags)}",
        location,
    )


def make_multiple_types_diagnostic(
    item_name: T, type_tags: FrozenSet[name]
) -> TypedListDiagnostic:
    """Build the historical 'more than one type' diagnostic for names."""
    return TypedListDiagnostic(
        f"typed list names should not have more than one type, got '{item_name}' "
        f"with types {_print_tag_set(type_tags)}"
    )


class TypedListIndex(Generic[T]):
    """
    Accumulate typed items following the historical duplicate policies.

    The index keeps input order both for names and for the variable sequence.
    Duplicates are rejected for names/constants/objects (unless they carry the
    very same set of tags), while variables may repeat the same tags.
    """

    def __init__(self, allow_duplicates: bool) -> None:
        """Initialize the index."""
        self._allow_duplicates = allow_duplicates
        self._types_to_items: Dict[name, Set[T]] = {}
        self._item_to_types: Dict[T, Set[name]] = {}
        self._item_to_types_sequence: List[
            Tuple[T, Set[name], Optional[SourceLocation]]
        ] = []

    def add(self, item: TypedListItem) -> None:
        """Add a normalized item, raising on duplicate/inconsistent tags."""
        item_name = item.value
        type_tags = set(item.type_tags)
        if not self._allow_duplicates:
            if item_name in self._item_to_types:
                raise make_duplicate_name_diagnostic(
                    item_name,
                    frozenset(self._item_to_types[item_name]),
                    item.location,
                )
            for type_tag in type_tags:
                if type_tag in self._item_to_types.get(item_name, set()):
                    raise make_duplicate_tag_diagnostic(
                        item_name, type_tag, item.type_location
                    )
        if item_name in self._item_to_types:
            previous = self._item_to_types[item_name]
            if previous != type_tags:
                raise make_inconsistent_tags_diagnostic(
                    item_name,
                    frozenset(previous),
                    frozenset(type_tags),
                    item.location,
                )
        for type_tag in type_tags:
            self._types_to_items.setdefault(type_tag, set()).add(item_name)
        self._item_to_types.setdefault(item_name, set()).update(type_tags)
        self._item_to_types_sequence.append((item_name, type_tags, item.location))

    def names(self) -> Dict[T, Optional[name]]:
        """Get the name -> (single) type mapping."""
        result: Dict[T, Optional[name]] = {}
        for item_name, type_tags in self._item_to_types.items():
            if len(type_tags) > 1:
                raise make_multiple_types_diagnostic(item_name, frozenset(type_tags))
            result[item_name] = next(iter(type_tags)) if type_tags else None
        return result

    def variables(self) -> Tuple[Tuple[T, Set[name], Optional[SourceLocation]], ...]:
        """Get the ordered ((variable, tags, location), ...) sequence."""
        return tuple(self._item_to_types_sequence)
