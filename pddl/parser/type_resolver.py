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

"""
Shared type resolution for the domain/problem/action parsers.

While :mod:`pddl.parser.typed_lists` groups tokens syntactically, this module
resolves the resulting type tags against the type hierarchy of the domain
being parsed:

- it records the ``(:types ...)`` hierarchy as it is encountered;
- it records typed term references (constants, action parameters, quantified
  variables, predicate/function arguments);
- it reports unknown types and inheritance cycles *after* the whole document
  has been grouped, aggregating every error and sorting it by source position.

Deferring the semantic checks guarantees that more specific errors (e.g. a
keyword used as a name, a duplicate constant) keep surfacing first.
"""

from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Tuple

from pddl.custom_types import name
from pddl.exceptions import PDDLValidationError
from pddl.helpers.base import find_cycle
from pddl.logic.terms import Term
from pddl.parser.symbols import Symbols
from pddl.parser.typed_lists import SourceLocation


@dataclass(frozen=True)
class _TypeReference:
    """A use of one or more type tags on a term."""

    what: str
    type_tags: FrozenSet[name]
    location: Optional[SourceLocation]


class TypeResolver:
    """Resolve typed-list outputs against the domain type hierarchy."""

    def __init__(self) -> None:
        """Initialize the resolver."""
        self._hierarchy: Dict[name, Optional[name]] = {}
        self._type_locations: Dict[name, SourceLocation] = {}
        self._references: List[_TypeReference] = []

    def record_hierarchy(
        self,
        hierarchy: Dict[name, Optional[name]],
        locations: Dict[name, SourceLocation],
    ) -> None:
        """Record the type hierarchy parsed from the ``:types`` section."""
        self._hierarchy.update(hierarchy)
        for type_name, location in locations.items():
            self._type_locations.setdefault(type_name, location)

    def record_term(self, term: Term, location: Optional[SourceLocation]) -> None:
        """Record a typed term occurrence for later unknown-type checking."""
        tags = frozenset(term.type_tags)
        if len(tags) == 0:
            return
        self._references.append(
            _TypeReference(what=repr(term), type_tags=tags, location=location)
        )

    def _all_known_types(self) -> FrozenSet[name]:
        # mirror the historical Types.all_types semantics: declared types and
        # their parents. The built-in 'object' tag is always legal when typing
        # is in use (it is the implicit root), but - to keep the historical
        # error text stable - it is included in the "available types" set only
        # when it already appears in the declared hierarchy.
        result = set(self._hierarchy.keys())
        result.update(
            parent for parent in self._hierarchy.values() if parent is not None
        )
        return frozenset(result)

    def _unknown_type_diagnostics(self) -> List[Tuple[SourceLocation, str]]:
        known_types = self._all_known_types()
        diagnostics: List[Tuple[SourceLocation, str]] = []
        seen: set = set()
        for reference in self._references:
            object_tag = name(Symbols.OBJECT.value)
            unknown = sorted(
                tag
                for tag in reference.type_tags
                if tag not in known_types and tag != object_tag
            )
            if not unknown:
                continue
            message = (
                f"types {[str(t) for t in unknown]} of term {reference.what} "
                f"are not in available types {set(sorted(known_types, key=str))!r}"
            )
            key = (message, reference.location)
            if key in seen:
                continue
            seen.add(key)
            location = reference.location or SourceLocation(0, 0, "")
            diagnostics.append((location, message))
        diagnostics.sort(key=lambda entry: (entry[0].line, entry[0].column, entry[1]))
        return diagnostics

    def finalize(self) -> None:
        """
        Run the deferred semantic checks.

        Inheritance cycles are reported before unknown types because the
        hierarchy is read top-to-bottom and a cycle makes parent lookups
        meaningless. Unknown-type diagnostics are aggregated and sorted by
        source position.
        """
        self._check_cycle()
        diagnostics = self._unknown_type_diagnostics()
        if diagnostics:
            messages = "\n\n".join(
                self._render(message, location) for location, message in diagnostics
            ).lstrip()
            raise LocatedTypeError(
                messages,
                [location for location, _ in diagnostics],
                [message for _, message in diagnostics],
            )

    @staticmethod
    def _render(message: str, location: SourceLocation) -> str:
        """Render a diagnostic, appending the location when available."""
        if location.line == 0:
            return message
        return f"{message} ({location.format()})"

    def _check_cycle(self) -> None:
        graph = {
            key: {value} if value is not None else set()
            for key, value in self._hierarchy.items()
        }
        cycle = find_cycle(graph)  # type: ignore
        if cycle is None:
            return
        location = self._type_locations.get(name(cycle[0]))
        message = "cycle detected in the type hierarchy: " + " -> ".join(cycle)
        raise LocatedTypeError(
            self._render(message, location) if location is not None else message,
            [location] if location is not None else [],
            [message],
        )


class LocatedTypeError(PDDLValidationError):
    """A type error carrying source locations and the un-suffixed messages."""

    def __init__(self, message, locations, base_messages):
        """Initialize the error."""
        super().__init__(message)
        self.locations = list(locations)
        self.base_messages = list(base_messages)
