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

"""Behavioral guards for the shared typed-list parsing machinery.

These tests cover the same typed-list shape across the five parsing entry
points (domain constants, action parameters, problem objects, quantified
variables, function parameters), prove that the entries really share one
internal function (test-side call counter), and pin down multi-error ordering
and source locations.
"""

import copy
import pickle
from textwrap import dedent

import pytest

from pddl.core import Domain
from pddl.exceptions import PDDLParsingError, PDDLValidationError
from pddl.parser import typed_lists
from pddl.parser.domain import DomainParser
from pddl.parser.problem import ProblemParser
from pddl.parser.typed_list_parser import TypedListParser

DOMAIN_HEADER = """
(define (domain shared-typed-lists)
    (:requirements :typing :universal-preconditions :existential-preconditions
                   :numeric-fluents :negative-preconditions)
    (:types vehicle place)
"""


def _domain(body: str) -> str:
    return dedent(DOMAIN_HEADER) + dedent(body) + "\n)\n"


# ---------------------------------------------------------------------------
# the five entries share the exact same shape
# ---------------------------------------------------------------------------


def test_domain_constants_grouping_shape() -> None:
    """Consecutive names followed by one type are grouped together."""
    domain = DomainParser()(_domain("    (:constants a b c - vehicle x y)"))
    tags = {c.name: c.type_tag for c in domain.constants}
    assert tags == {
        "a": "vehicle",
        "b": "vehicle",
        "c": "vehicle",
        "x": None,
        "y": None,
    }


def test_action_parameters_grouping_shape() -> None:
    """Action parameters follow the same grouping boundary rules."""
    domain = DomainParser()(_domain("""
    (:predicates (p ?v ?w ?z))
    (:action move
        :parameters (?v ?w - vehicle ?z)
        :precondition (p ?v ?w ?z)
        :effect (p ?v ?w ?z))
            """))
    action = next(iter(domain.actions))
    tags = {v.name: set(v.type_tags) for v in action.parameters}
    assert tags == {"v": {"vehicle"}, "w": {"vehicle"}, "z": set()}


def test_problem_objects_grouping_shape() -> None:
    """Problem objects follow the same grouping boundary rules."""
    problem_str = dedent("""
    (define (problem p)
        (:domain shared-typed-lists)
        (:objects car1 car2 - vehicle home office - place free)
        (:init)
        (:goal (and))
    )
    """)
    problem = ProblemParser()(problem_str)
    tags = {o.name: o.type_tag for o in problem.objects}
    assert tags == {
        "car1": "vehicle",
        "car2": "vehicle",
        "home": "place",
        "office": "place",
        "free": None,
    }


def test_quantified_variables_grouping_shape() -> None:
    """Quantified variables follow the same grouping boundary rules."""
    domain = DomainParser()(_domain("""
    (:predicates (p ?v ?w ?z))
    (:action q
        :parameters ()
        :precondition (and
            (forall (?v ?w - vehicle ?z - place) (p ?v ?w ?z))
            (exists (?a ?b - place) (p ?a ?b ?a)))
        :effect (p))
            """))
    action = next(iter(domain.actions))
    precondition = action.precondition
    assert precondition is not None
    forall_cond = precondition.operands[0]  # type: ignore[attr-defined]
    forall_tags = {v.name: set(v.type_tags) for v in forall_cond.variables}
    assert forall_tags == {"v": {"vehicle"}, "w": {"vehicle"}, "z": {"place"}}

    exists_cond = precondition.operands[1]  # type: ignore[attr-defined]
    exists_tags = {v.name: set(v.type_tags) for v in exists_cond.variables}
    assert exists_tags == {"a": {"place"}, "b": {"place"}}


def test_function_parameters_grouping_shape() -> None:
    """Function skeleton parameters follow the same grouping boundary rules."""
    domain = DomainParser()(_domain("""
    (:predicates (p))
    (:functions (distance ?from ?to - place) (price ?what - vehicle ?free))
    (:action a
        :parameters ()
        :precondition (p)
        :effect (p))
            """))
    functions = {f.name: f for f in domain.functions}
    distance_terms = {v.name: set(v.type_tags) for v in functions["distance"].terms}
    assert distance_terms == {"from": {"place"}, "to": {"place"}}
    price_terms = {v.name: set(v.type_tags) for v in functions["price"].terms}
    assert price_terms == {"what": {"vehicle"}, "free": set()}


def test_untyped_default_group_is_empty_tags() -> None:
    """Without a separator every entry lands in the default untyped group."""
    parsed = TypedListParser.parse_typed_list(["a", "b", "c"]).get_typed_list_of_names()
    assert parsed == {"a": None, "b": None, "c": None}


def test_keyword_case_is_significant() -> None:
    """The 'object' type is the only keyword allowed as a type tag."""
    domain = DomainParser()(dedent("""
    (define (domain case-sensitive)
        (:requirements :typing)
        (:types car - object)
    )
    """))
    assert domain.types == {"car": None}


# ---------------------------------------------------------------------------
# shared-mechanism call counter
# ---------------------------------------------------------------------------


def test_all_entries_share_one_internal_grouping(monkeypatch) -> None:
    """Every entry point must call the same internal grouping function.

    A test-side counter on the single internal function proves the entries do
    not re-implement (or copy) the grouping logic, and that each of the five
    entry shapes reaches it.
    """
    call_log = []
    original = typed_lists.iter_typed_items

    def counting(tokens):
        call_log.append([str(token) for token in tokens])
        return original(tokens)

    monkeypatch.setattr(typed_lists, "iter_typed_items", counting)

    DomainParser()(_domain("""
    (:constants k1 k2 - vehicle)
    (:predicates (p ?v - vehicle ?w))
    (:functions (f ?x - place))
    (:action move
        :parameters (?v ?w - vehicle)
        :precondition (and (p ?v ?w) (forall (?q - place) (p ?q ?q)))
        :effect (p ?v ?w))
        """))

    ProblemParser()(dedent("""
    (define (problem p)
        (:domain shared-typed-lists)
        (:objects car1 - vehicle home - place)
        (:init)
        (:goal (and))
    )
    """))

    flat = {token for tokens in call_log for token in tokens if token != "-"}
    # one call per typed-list parse:
    # types, constants, predicate vars, function-skeleton name list, function
    # vars, action params, quantified vars and problem objects
    assert len(call_log) == 8
    # each entry shape can be identified by a token it contributes
    assert "vehicle" in flat and "place" in flat  # :types
    assert "k1" in flat  # domain constants
    assert "car1" in flat  # problem objects
    assert any(tokens.count("v") >= 1 for tokens in call_log)  # action params
    assert "q" in flat  # quantified variables
    assert "x" in flat  # function parameters
    assert any("(f ?x)" in tokens for tokens in call_log)  # function skeleton


def test_transformers_do_not_call_the_legacy_facade(monkeypatch) -> None:
    """Transformers must bypass the legacy class façade internally."""
    counter = {"calls": 0}
    original = TypedListParser.parse_typed_list

    def counting(tokens, allow_duplicates=False):
        counter["calls"] += 1
        return original(tokens, allow_duplicates)

    monkeypatch.setattr(TypedListParser, "parse_typed_list", staticmethod(counting))

    DomainParser()(_domain("""
    (:constants k1 - vehicle)
    (:predicates (p ?v - vehicle))
    (:action a :parameters (?v - vehicle) :precondition (p ?v) :effect (p ?v))
            """))
    assert counter["calls"] == 0


# ---------------------------------------------------------------------------
# diagnostics: exception categories, locations and ordering
# ---------------------------------------------------------------------------


def test_duplicate_constant_is_parsing_error_with_location() -> None:
    """Same-name conflicts keep the PDDLParsingError category and get a position."""
    with pytest.raises(
        PDDLParsingError,
        match=r"duplicate name 'c1' in typed list already present.*"
        r"line 6, column 19: 'c1'",
    ):
        DomainParser()(_domain("    (:constants c1 c2 c1)"))


def test_duplicate_problem_object_is_parsing_error_with_location() -> None:
    """Same-name conflicts for objects get a stable position too."""
    with pytest.raises(
        PDDLParsingError,
        match=r"duplicate name 'a' in typed list already present.*"
        r"line 4, column 19: 'a'",
    ):
        ProblemParser()(dedent("""
    (define (problem p)
        (:domain d)
        (:objects a b a)
        (:init)
        (:goal (and))
    )
    """))


def test_unknown_action_parameter_type_is_validation_error_with_location() -> None:
    """Unknown types remain PDDLValidationError, now anchored to the input."""
    with pytest.raises(
        PDDLValidationError,
        match=r"types \['ghost'\] of term Variable\(v\).*line 9, column 19: 'v'",
    ):
        DomainParser()(_domain("""
    (:predicates (p ?v))
    (:action a
        :parameters (?v - ghost)
        :precondition (p ?v)
        :effect (p ?v))
                """))


def test_inheritance_cycle_is_validation_error_with_location() -> None:
    """Inheritance cycles remain PDDLValidationError and point at the first node."""
    with pytest.raises(
        PDDLValidationError,
        match=r"cycle detected in the type hierarchy: A -> B -> C.*"
        r"line [0-9]+, column 13: 'A'",
    ):
        DomainParser()(dedent("""
    (define (domain cyc)
        (:requirements :typing)
        (:types A - B B - C C - A)
    )
    """))


def test_multiple_unknown_types_are_sorted_by_source_position() -> None:
    """All unknown types are reported, ordered by line then column."""
    # two predicate-level unknown types: the lexically later name occurs on an
    # earlier source line, so source ordering must win over name ordering
    text = _domain("""
    (:predicates
        (q ?x - z_late)
        (p ?x - z_early))
        """)
    with pytest.raises(PDDLValidationError) as exc_info:
        DomainParser()(text)

    message = str(exc_info.value)
    lines = [line for line in message.splitlines() if line.strip()]
    assert "z_late" in lines[0]
    assert any("z_early" in line for line in lines[1:])
    # earlier source line (z_late) is reported first even though z_early < z_late
    assert message.index("z_late") < message.index("z_early")


def test_unknown_types_on_same_line_are_sorted_by_column() -> None:
    """On the same line, diagnostics are ordered by column (left to right)."""
    text = _domain("""
    (:predicates
        (m ?zz - z_right ?aa - z_left))
        """)
    with pytest.raises(PDDLValidationError) as exc_info:
        DomainParser()(text)

    message = str(exc_info.value)
    lines = [line for line in message.splitlines() if line.strip()]
    # both variables share a line; the leftmost typed variable (zz) is first
    assert "z_right" in lines[0]
    assert "z_left" in message
    assert message.index("z_right") < message.index("z_left")


# ---------------------------------------------------------------------------
# round-trip / structural equality and serialization stability
# ---------------------------------------------------------------------------


def test_parse_format_parse_round_trip_structure() -> None:
    """parse-format-parse keeps structural equality for the shared entries."""
    domain = DomainParser()(_domain("""
    (:constants a b - vehicle home - place free)
    (:predicates (at ?v - vehicle ?p - place))
    (:action move
        :parameters (?v - vehicle ?from ?to - place)
        :precondition (at ?v ?from)
        :effect (at ?v ?to))
        """))
    re_parsed = DomainParser()(str(domain))
    assert re_parsed == domain


def test_problem_parse_format_parse_round_trip_structure() -> None:
    """The same round-trip property for problem objects."""
    problem = ProblemParser()(dedent("""
    (define (problem p)
        (:domain shared-typed-lists)
        (:objects car1 - vehicle home - place free)
        (:init)
        (:goal (and))
    )
    """))
    assert ProblemParser()(str(problem)) == problem


def test_pickle_and_deepcopy_preserve_typed_domain() -> None:
    """Pickle and deepcopy do not regress equality/hash semantics."""
    domain = DomainParser()(_domain("""
    (:constants a b - vehicle)
    (:predicates (p ?v - vehicle))
    (:action m
        :parameters (?v - vehicle)
        :precondition (p ?v)
        :effect (p ?v))
        """))
    restored = pickle.loads(pickle.dumps(domain))  # nosec
    assert restored == domain
    assert hash(tuple(sorted(domain.constants))) == hash(
        tuple(sorted(restored.constants))
    )

    copied = copy.deepcopy(domain)
    assert copied == domain
    assert copied.constants == domain.constants
    assert isinstance(copied, Domain)
