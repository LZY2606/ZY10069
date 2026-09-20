#
# Copyright 2021-2025 WhiteMech
#
# ------------------------------
#
# This file is part of pddl.
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
#

"""Internal mechanism to handle typed lists and type resolution.

This module is the single place where the PDDL parsers:

- group the tokens of a typed list (names, variables, constants, objects,
  function skeletons) into (item, type tags) pairs - see `TypedListParser`;
- resolve the type tags of the grouped items against the declared types
  hierarchy - see `TypeResolver`.

Both components attach source positions (line/column) and, when the input
text is available, the offending input snippet to their diagnostics, so that
failures can be traced back to the user input.  When several problems are
detected, the diagnostics are sorted by source position (line, column, then
token index) so the output is stable.
"""

from typing import (
    Any,
    Dict,
    Generic,
    Iterable,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

from pddl.custom_types import name, parse_name, parse_type
from pddl.helpers.base import check, find_cycle, safe_index
from pddl.logic.functions import NumericFunction
from pddl.logic.terms import _print_tag_set
from pddl.parser.symbols import Symbols

T = TypeVar("T", name, NumericFunction)


class SourcePosition:
    """The position of a token in the input text (best effort)."""

    __slots__ = ("line", "column", "index")

    def __init__(
        self,
        line: Optional[int] = None,
        column: Optional[int] = None,
        index: int = 0,
    ) -> None:
        """Initialize the source position."""
        self.line = line
        self.column = column
        self.index = index

    @classmethod
    def of(cls, token: Any, index: int) -> "SourcePosition":
        """Build the position of a token, if the token carries one."""
        return cls(
            line=getattr(token, "line", None),
            column=getattr(token, "column", None),
            index=index,
        )

    def sort_key(self) -> Tuple[bool, int, int, int]:
        """Return a key for stable sorting by source position."""
        return (self.line is None, self.line or 0, self.column or 0, self.index)

    def __str__(self) -> str:
        """Get the string representation."""
        if self.line is None:
            return ""
        return f" (line {self.line}, column {self.column})"


class TypedListDiagnostic:
    """A diagnostic about a typed list, anchored to the user input."""

    def __init__(
        self,
        message: str,
        position: Optional[SourcePosition] = None,
        source: Optional[str] = None,
    ) -> None:
        """Initialize the diagnostic."""
        self.message = message
        self.position = position if position is not None else SourcePosition()
        self.source = source

    def sort_key(self) -> Tuple[bool, int, int, int]:
        """Return a key for stable sorting by source position."""
        return self.position.sort_key()

    @property
    def snippet(self) -> Optional[str]:
        """Return the input line the diagnostic points to, if available."""
        if self.source is None or self.position.line is None:
            return None
        lines = self.source.splitlines()
        if not 1 <= self.position.line <= len(lines):
            return None
        return lines[self.position.line - 1].strip()

    def __str__(self) -> str:
        """Get the string representation."""
        result = self.message + str(self.position)
        snippet = self.snippet
        if snippet:
            result += f": {snippet}"
        return result


def format_diagnostics(diagnostics: List[TypedListDiagnostic], what: str) -> str:
    """
    Format a non-empty list of diagnostics, sorted by source position.

    :param diagnostics: the diagnostics (at least one).
    :param what: what the diagnostics are about, used in the header.
    :return: the formatted message.
    """
    sorted_diagnostics = sorted(diagnostics, key=lambda d: d.sort_key())
    if len(sorted_diagnostics) == 1:
        return str(sorted_diagnostics[0])
    enumerated = "\n".join(
        f"{index}) {diagnostic}"
        for index, diagnostic in enumerate(sorted_diagnostics, start=1)
    )
    return (
        f"found {len(sorted_diagnostics)} errors in {what}, "
        f"sorted by source position:\n{enumerated}"
    )


class TypedListParser(Generic[T]):
    """
    An index for PDDL types and PDDL names/variables/functions.

    This class is used to index PDDL names, variables, and functions by their types.
    OrderedDict is used to preserve the order of the types and the names, e.g. for predicate variables.

    Other types of validations are performed to ensure that the types index is consistent.
    """

    def __init__(
        self, allow_duplicates: bool, source: Optional[str] = None
    ) -> None:
        """
        Initialize the types index.

        :param allow_duplicates: whether duplicate items are allowed
        :param source: the input text the tokens come from (used for snippets)
        """
        self._allow_duplicates = allow_duplicates
        self._source = source

        self._types_to_items: Dict[T, Set[T]] = {}
        self._item_to_types: Dict[T, Set[T]] = {}
        self._item_to_types_sequence: List[Tuple[T, Set[T]]] = []

        self._diagnostics: List[TypedListDiagnostic] = []
        self._item_positions: Dict[T, SourcePosition] = {}
        self._type_tag_positions: Dict[T, SourcePosition] = {}

    @property
    def diagnostics(self) -> Tuple[TypedListDiagnostic, ...]:
        """Get the diagnostics collected so far."""
        return tuple(self._diagnostics)

    def item_position(self, item_name: T) -> SourcePosition:
        """Get the position of the first occurrence of an item."""
        return self._item_positions.get(item_name, SourcePosition())

    def type_tag_position(self, type_tag: T) -> SourcePosition:
        """Get the position of the first occurrence of a type tag."""
        return self._type_tag_positions.get(type_tag, SourcePosition())

    def add_item(
        self,
        item_name: T,
        type_tags: Set[T],
        position: Optional[SourcePosition] = None,
        type_tag_positions: Optional[Mapping[Any, SourcePosition]] = None,
    ) -> None:
        """
        Add an item to the types index with the given type tags.

        Both the item name and the type tags are validated according to the name type regular expression.
        Structural problems (duplicates, inconsistent tags) do not raise immediately:
        they are recorded as diagnostics, so that several of them can be reported
        together, sorted by source position (see `raise_if_diagnostics`).

        :param item_name: the item name
        :param type_tags: the types for the item
        :param position: the source position of the item token
        :param type_tag_positions: the source positions of the type tag tokens
        """
        position = position if position is not None else SourcePosition()
        if not self._allow_duplicates:
            is_duplicate = self._check_item_name_already_present(item_name, position)
            if not is_duplicate:
                self._check_tags_already_present(item_name, type_tags, position)
                self._check_item_types(item_name, type_tags, position)
        else:
            self._check_item_types(item_name, type_tags, position)
        self._add_item(item_name, type_tags)
        self._item_positions.setdefault(item_name, position)
        for type_tag in type_tags:
            tag_position = (type_tag_positions or {}).get(type_tag, position)
            self._type_tag_positions.setdefault(type_tag, tag_position)

    def raise_if_diagnostics(self) -> None:
        """Raise a ValueError with all the collected diagnostics, if any."""
        if not self._diagnostics:
            return
        raise ValueError(format_diagnostics(self._diagnostics, "typed list"))

    def get_typed_list_of_names(self) -> Dict[T, Optional[T]]:
        """Get the typed list of names in form of dictionary."""
        result: Dict[T, Optional[T]] = {}
        for item, types_tags in self._item_to_types.items():
            if len(types_tags) > 1:
                self._raise_multiple_types_error(item, types_tags)
            type_tag = next(iter(types_tags)) if len(types_tags) == 1 else None
            result[item] = type_tag
        return result

    def get_typed_list_of_variables(self) -> Tuple[Tuple[T, Set[T]], ...]:
        """Get the typed list of variables in form of a tuple of pairs."""
        return tuple(self._item_to_types_sequence)

    @classmethod
    def parse_typed_list(
        cls,
        tokens: List[Union[T, List[T]]],
        allow_duplicates: bool = False,
        source: Optional[str] = None,
    ) -> "TypedListParser":
        """
        Parse typed list.

        This method takes in input a list of tokens as returned by the domain or problem parsers,
        and returns a TypesIndex object.

        The tokens may carry source positions (e.g. Lark tokens); when they do,
        and optionally when the input `source` text is provided, the diagnostics
        produced by the returned object point back to the user input.

        The input list of tokens must have the following format:
        - if the list is not typed, it is simply a list of names
        - if the list is typed, the format is: [name_1, ..., name_n], "-", [type_1, ..., type_m], ...

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
        :param allow_duplicates: whether allowed_duplicates are allowed
        :param source: the input text the tokens come from (used for snippets)
        :return: the TypesIndex object
        """
        result = TypedListParser(allow_duplicates=allow_duplicates, source=source)

        type_sep_index = safe_index(tokens, Symbols.TYPE_SEP.value)

        if type_sep_index is None:
            # simple list of names
            cls._add_typed_lists(result, 0, len(tokens), tokens, set())
            result.raise_if_diagnostics()
            return result

        # if we are here, the matched pattern is: [name_1, ..., name_n], "-", parent_name, ...
        # Consume typed sublists iteratively. Caveat: the last typed list *might* have no parent type.

        # the index of the separator symbol for the current typed sublist being processed
        type_sep_index = safe_index(tokens, Symbols.TYPE_SEP.value)
        # the index of the first element of the current typed sublist
        start_index = 0
        # the index of the last element of the current typed sublist
        end_index = len(tokens)
        while type_sep_index is not None:
            # the name of the parent type is the element after the separator
            parent_names = tokens[type_sep_index + 1]
            # handle the case of a single parent type
            if not isinstance(parent_names, list):
                parent_names = [parent_names]

            # record the positions of the type tag tokens
            type_tag_positions = {
                parent_name: SourcePosition.of(parent_name, type_sep_index + 1)
                for parent_name in parent_names
            }

            # parse the typed list
            cls._add_typed_lists(
                result,
                start_index,
                type_sep_index,
                tokens,
                set(parent_names),
                type_tag_positions,
            )

            # go to next typed list (if any)
            start_index = type_sep_index + 2
            type_sep_index = safe_index(
                tokens, Symbols.TYPE_SEP.value, start_index, end_index
            )

        # this is to handle the case the last sublist is not typed
        if start_index != end_index:
            # parse the last typed list, with no type.
            cls._add_typed_lists(result, start_index, end_index, tokens, set())

        result.raise_if_diagnostics()
        return result

    @classmethod
    def _add_typed_lists(
        cls,
        result: "TypedListParser",
        start_index: int,
        end_index: int,
        tokens: List[Union[T, List[T]]],
        type_tags: Set[str],
        type_tag_positions: Optional[Mapping[Any, SourcePosition]] = None,
    ) -> None:
        """
        Merge typed lists.

        Side-effect on the 'result' dictionary. The start_index and end_index are needed to avoid useless
        sublist copies.
        """
        for index in range(start_index, end_index):
            item_name = tokens[index]
            check(
                isinstance(item_name, str) or isinstance(item_name, NumericFunction),
                f"invalid item '{item_name}' in typed list",
            )
            # these lines implicitly perform name validation
            cast_item_name: Any = (
                parse_name(item_name) if isinstance(item_name, str) else item_name
            )
            type_tags_names: Set[Any] = set(map(parse_type, type_tags))
            result.add_item(
                cast_item_name,
                type_tags_names,
                position=SourcePosition.of(item_name, index),
                type_tag_positions=type_tag_positions,
            )

    def _check_item_name_already_present(
        self, item_name: T, position: SourcePosition
    ) -> bool:
        """
        Check if the item name/function is already present in the index.

        :param item_name: the item name
        :param position: the source position of the offending occurrence
        :return: True if the item name is already present, False otherwise
        """
        if item_name in self._item_to_types:
            types_list = sorted(map(str, self._item_to_types[item_name]))
            if len(types_list) > 0:
                message = (
                    f"duplicate name '{item_name}' in typed list already inherits from types {types_list}"
                )
            else:
                message = f"duplicate name '{item_name}' in typed list already present"
            self._diagnostics.append(
                TypedListDiagnostic(message, position, self._source)
            )
            return True
        return False

    def _check_tags_already_present(
        self, item_name: T, type_tags: Set[T], position: SourcePosition
    ) -> None:
        """
        Check if the type tags are already present for the given item name.

        :param item_name: the item name
        :param type_tags: the type tags
        :param position: the source position of the offending occurrence
        """
        existing_tags = self._item_to_types.get(item_name, set())
        for type_tag in sorted(map(str, type_tags)):
            if type_tag in existing_tags:
                self._diagnostics.append(
                    TypedListDiagnostic(
                        f"duplicate type tag '{type_tag}' in typed list: type already specified for item {item_name}",
                        position,
                        self._source,
                    )
                )

    def _check_item_types(
        self, item_name: T, type_tags: Set[T], position: SourcePosition
    ) -> None:
        """Check if the types of the item are valid."""
        if item_name in self._item_to_types:
            previous_type_tags = self._item_to_types[item_name]
            if previous_type_tags != type_tags:
                self._diagnostics.append(
                    TypedListDiagnostic(
                        f"invalid types for item '{item_name}': previous known tags were "
                        f"{_print_tag_set(previous_type_tags)}, got {_print_tag_set(type_tags)}",
                        position,
                        self._source,
                    )
                )

    def _add_item(self, item_name: T, type_tags: Set[T]) -> None:
        """Add an item (no validation)."""
        for type_tag in type_tags:
            self._types_to_items.setdefault(type_tag, set()).add(item_name)
        self._item_to_types.setdefault(item_name, set()).update(type_tags)
        self._item_to_types_sequence.append((item_name, type_tags))

    def _raise_multiple_types_error(self, item_name: T, type_tags: Set[T]) -> None:
        """Raise an error if the item has multiple types."""
        raise ValueError(
            f"typed list names should not have more than one type, got '{item_name}' with "
            f"types {_print_tag_set(type_tags)}"
        )


class TypeResolver:
    """
    Resolve the type tags of typed lists against a types hierarchy.

    This is the type-resolution half of the typed-list mechanism: it mirrors the
    checks performed when a PDDL domain object is constructed (see
    `pddl._validation.Types` and `pddl._validation.TypeChecker`), but works on the
    parser-side representation so that the produced diagnostics can be anchored to
    the user input (line/column and input snippet).

    The resolver never raises: it returns diagnostics sorted by source position,
    and it is up to the caller to decide whether and how to surface them. This
    guarantees that no validation is anticipated in a way that could mask more
    specific errors raised elsewhere.
    """

    def __init__(
        self,
        types_hierarchy: Optional[Mapping[Any, Optional[Any]]] = None,
        types_list: Optional[TypedListParser] = None,
        source: Optional[str] = None,
    ) -> None:
        """
        Initialize the type resolver.

        :param types_hierarchy: the declared types, as a name-to-parent mapping
        :param types_list: the typed list the types were declared in (for positions)
        :param source: the input text (used for snippets)
        """
        self._types_hierarchy = dict(types_hierarchy) if types_hierarchy else {}
        self._types_list = types_list
        self._source = source

    @property
    def all_types(self) -> Set[Any]:
        """Get all the available types (declared types and their ancestors)."""
        result = set(self._types_hierarchy.keys()) | set(
            self._types_hierarchy.values()
        )
        result.discard(None)
        return result

    def cycle_diagnostics(self) -> List[TypedListDiagnostic]:
        """
        Detect inheritance cycles in the types hierarchy.

        :return: one diagnostic per type declaration involved in the cycle,
            sorted by source position.
        """
        cycle = find_cycle(
            {
                key: {value} if value is not None else set()
                for key, value in self._types_hierarchy.items()
            }
        )
        if cycle is None:
            return []
        cycle_str = " -> ".join(map(str, cycle))
        diagnostics = [
            TypedListDiagnostic(
                f"type '{type_name}' is part of the inheritance cycle {cycle_str}",
                self._type_declaration_position(type_name),
                self._source,
            )
            for type_name in cycle
        ]
        return sorted(diagnostics, key=lambda d: d.sort_key())

    def unknown_type_diagnostics(
        self, typed_lists: Iterable[Tuple[TypedListParser, str]]
    ) -> List[TypedListDiagnostic]:
        """
        Detect type tags that are not in the available types.

        :param typed_lists: pairs of (typed list, item kind) to check
        :return: one diagnostic per unknown type tag occurrence,
            sorted by source position.
        """
        all_types = self.all_types
        diagnostics = []
        for typed_list, kind in typed_lists:
            for item, type_tags in typed_list.get_typed_list_of_variables():
                for type_tag in sorted(map(str, type_tags)):
                    if type_tag not in all_types:
                        position = typed_list.type_tag_position(type_tag)
                        if position.line is None:
                            position = typed_list.item_position(item)
                        diagnostics.append(
                            TypedListDiagnostic(
                                f"type '{type_tag}' of {kind} '{item}' is not in "
                                f"available types {sorted(map(str, all_types))}",
                                position,
                                self._source,
                            )
                        )
        return sorted(diagnostics, key=lambda d: d.sort_key())

    def _type_declaration_position(self, type_name: Any) -> SourcePosition:
        """Get the position of the declaration of a type, if available."""
        if self._types_list is None:
            return SourcePosition()
        return self._types_list.item_position(type_name)
