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

"""Public, backwards-compatible façade over the shared typed-list machinery."""

from typing import Any, Dict, List, Set, Tuple, TypeVar

from pddl.custom_types import name
from pddl.logic.functions import NumericFunction
from pddl.parser.typed_lists import (
    TypedListDiagnostic,
    TypedListIndex,
    iter_typed_items,
)

T = TypeVar("T", name, NumericFunction)


class TypedListParser:
    """
    An index for PDDL types and PDDL names/variables/functions.

    This class is a thin façade over the shared internal machinery in
    :mod:`pddl.parser.typed_lists`. The domain, problem and action parsing
    entry points all build their typed lists through that machinery; this
    façade only preserves the historical class-based API and its exceptions
    (plain :class:`ValueError`).

    OrderedDict-like insertion order is preserved, e.g. for predicate
    variables.
    """

    def __init__(self, allow_duplicates: bool) -> None:
        """Initialize the types index."""
        self._index: TypedListIndex = TypedListIndex(allow_duplicates)

    def add_item(self, item_name: Any, type_tags: Set) -> None:
        """
        Add an item to the types index with the given type tags.

        Both the item name and the type tags are validated according to the
        name type regular expression.

        :param item_name: the item name
        :param type_tags: the types for the item
        """
        from pddl.custom_types import parse_name, parse_type
        from pddl.parser.typed_lists import TypedListItem

        value = parse_name(item_name) if isinstance(item_name, str) else item_name
        normalized_tags = frozenset(map(parse_type, type_tags))
        try:
            self._index.add(
                TypedListItem(
                    value=value,
                    type_tags=normalized_tags,
                    location=None,
                    type_location=None,
                )
            )
        except TypedListDiagnostic as diagnostic:
            raise ValueError(diagnostic.message) from diagnostic

    def get_typed_list_of_names(self) -> Dict:
        """Get the typed list of names in form of dictionary."""
        try:
            return self._index.names()
        except TypedListDiagnostic as diagnostic:
            raise ValueError(diagnostic.message) from diagnostic

    def get_typed_list_of_variables(self) -> Tuple[Tuple, ...]:
        """Get the typed list of variables in form of a tuple of pairs."""
        return self._index.variables()

    @classmethod
    def parse_typed_list(
        cls, tokens: List[Any], allow_duplicates: bool = False
    ) -> "TypedListParser":
        """
        Parse typed list.

        This method takes in input a list of tokens as returned by the domain
        or problem parsers, and returns a TypesIndex object.

        The input list of tokens must have the following format:
        - if the list is not typed, it is simply a list of names
        - if the list is typed, the format is:
          [name_1, ..., name_n], "-", [type_1, ..., type_m], ...

        >>> index = TypedListParser.parse_typed_list(["a", "b", "c"])
        >>> index.get_typed_list_of_names()
        {'a': None, 'b': None, 'c': None}

        >>> index = TypedListParser.parse_typed_list(["a", "b", "c", "-", ["t1"]])
        >>> index.get_typed_list_of_names()
        {'a': 't1', 'b': 't1', 'c': 't1'}

        >>> index = TypedListParser.parse_typed_list(["a", "b", "c", "-", ["t1", "t2"]])
        >>> index.get_typed_list_of_names()
        Traceback (most recent call last):
        ...
        ValueError: typed list names should not have more than one type, got 'a' with types ['t1', 't2']

        :param tokens: the list of tokens
        :param allow_duplicates: whether duplicates are allowed
        :return: the TypesIndex object
        """
        result = TypedListParser(allow_duplicates=allow_duplicates)
        try:
            for item in iter_typed_items(tokens):
                result._index.add(item)
        except TypedListDiagnostic as diagnostic:
            raise ValueError(diagnostic.message) from diagnostic
        return result
