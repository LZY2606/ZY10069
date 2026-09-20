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

"""
Behavior guards for the unified typed-list mechanism.

The typed lists of domain constants, action parameters, problem objects,
quantifier variables and function arguments are all grouped and resolved by
the same internal mechanism (`pddl.parser.typed_list_parser`). These tests
pin down the shared boundary rules for every entry point, the diagnostics
(line/column, input snippet, stable ordering by source position), and the
fact that the mechanism is really shared (call counting).
"""

import copy
import pickle
from textwrap import dedent

import pytest

from pddl.core import Domain, Problem
from pddl.exceptions import PDDLParsingError, PDDLValidationError
from pddl.formatter import domain_to_string, problem_to_string
from pddl.logic.terms import Constant, Variable
from pddl.parser.domain import DomainParser
from pddl.parser.problem import ProblemParser
from pddl.parser.typed_list_parser import TypedListParser


def _domain_with_constants(constants_def: str) -> str:
    """Build a domain string with the given constants definition."""
    return dedent(
        f"""
        (define (domain test)
            (:requirements :typing)
            (:types t1 t2)
            (:constants {constants_def})
        )"""
    )


def _domain_with_action_parameters(parameters_def: str) -> str:
    """Build a domain string with the given action parameters."""
    return dedent(
        f"""
        (define (domain test)
            (:requirements :typing :strips)
            (:types t1 t2)
            (:predicates (p))
            (:action act
                :parameters ({parameters_def})
                :precondition (and)
                :effect (and))
        )"""
    )


def _problem_with_objects(objects_def: str) -> str:
    """Build a problem string with the given objects definition."""
    return dedent(
        f"""
        (define (problem test-problem)
            (:domain test-domain)
            (:requirements :typing)
            (:objects {objects_def})
            (:init )
            (:goal (and ))
        )"""
    )


def _domain_with_quantifier_variables(variables_def: str) -> str:
    """Build a domain string with a forall-precondition over the given variables."""
    return dedent(
        f"""
        (define (domain test)
            (:requirements :typing :strips :universal-preconditions)
            (:types t1 t2)
            (:predicates (p))
            (:action act
                :parameters ()
                :precondition (forall ({variables_def}) (p))
                :effect (and))
        )"""
    )


def _domain_with_function_arguments(arguments_def: str) -> str:
    """Build a domain string with a function taking the given arguments."""
    return dedent(
        f"""
        (define (domain test)
            (:requirements :typing :numeric-fluents)
            (:types t1 t2)
            (:functions (f {arguments_def}))
        )"""
    )


class TestSharedBoundaryRules:
    """The same grouping rules hold for every typed-list entry point."""

    def test_domain_constants_untyped_default(self) -> None:
        """Untyped domain constants get the default (no type tag)."""
        domain = DomainParser()(_domain_with_constants("c1 c2 c3"))
        assert domain.constants == {Constant("c1"), Constant("c2"), Constant("c3")}
        assert all(c.type_tag is None for c in domain.constants)

    def test_domain_constants_multiple_names_one_type(self) -> None:
        """Several names followed by one type all inherit that type."""
        domain = DomainParser()(_domain_with_constants("c1 c2 c3 - t1 c4"))
        assert {c.name: c.type_tag for c in domain.constants} == {
            "c1": "t1",
            "c2": "t1",
            "c3": "t1",
            "c4": None,
        }

    def test_action_parameters_untyped_default(self) -> None:
        """Untyped action parameters get the default (empty type tags)."""
        domain = DomainParser()(_domain_with_action_parameters("?x ?y"))
        (action,) = domain.actions
        assert list(action.parameters) == [Variable("x"), Variable("y")]
        assert all(len(p.type_tags) == 0 for p in action.parameters)

    def test_action_parameters_multiple_names_one_type(self) -> None:
        """Several parameters followed by one type all inherit that type."""
        domain = DomainParser()(_domain_with_action_parameters("?x ?y - t1 ?z"))
        (action,) = domain.actions
        assert [p.name for p in action.parameters] == ["x", "y", "z"]
        assert [sorted(p.type_tags) for p in action.parameters] == [
            ["t1"],
            ["t1"],
            [],
        ]

    def test_problem_objects_untyped_default(self) -> None:
        """Untyped problem objects get the default (no type tag)."""
        problem = ProblemParser()(_problem_with_objects("o1 o2"))
        assert problem.objects == {Constant("o1"), Constant("o2")}
        assert all(o.type_tag is None for o in problem.objects)

    def test_problem_objects_multiple_names_one_type(self) -> None:
        """Several objects followed by one type all inherit that type."""
        problem = ProblemParser()(_problem_with_objects("o1 o2 - t1 o3"))
        assert {o.name: o.type_tag for o in problem.objects} == {
            "o1": "t1",
            "o2": "t1",
            "o3": None,
        }

    def test_quantifier_variables_untyped_default(self) -> None:
        """Untyped quantifier variables get the default (empty type tags)."""
        domain = DomainParser()(_domain_with_quantifier_variables("?x ?y"))
        (action,) = domain.actions
        quantified = action.precondition
        assert {v.name for v in quantified.variables} == {"x", "y"}
        assert all(len(v.type_tags) == 0 for v in quantified.variables)

    def test_quantifier_variables_multiple_names_one_type(self) -> None:
        """Several quantified variables followed by one type all inherit it."""
        domain = DomainParser()(_domain_with_quantifier_variables("?x ?y - t1 ?z"))
        (action,) = domain.actions
        quantified = action.precondition
        assert {v.name: sorted(v.type_tags) for v in quantified.variables} == {
            "x": ["t1"],
            "y": ["t1"],
            "z": [],
        }

    def test_function_arguments_untyped_default(self) -> None:
        """Untyped function arguments get the default (empty type tags)."""
        domain = DomainParser()(_domain_with_function_arguments("?x ?y"))
        (function,) = domain.functions
        assert [t.name for t in function.terms] == ["x", "y"]
        assert all(len(t.type_tags) == 0 for t in function.terms)

    def test_function_arguments_multiple_names_one_type(self) -> None:
        """Several function arguments followed by one type all inherit it."""
        domain = DomainParser()(_domain_with_function_arguments("?x ?y - t1 ?z"))
        (function,) = domain.functions
        assert [t.name for t in function.terms] == ["x", "y", "z"]
        assert [sorted(t.type_tags) for t in function.terms] == [
            ["t1"],
            ["t1"],
            [],
        ]

    def test_input_order_is_preserved(self) -> None:
        """The input order of typed-list items is preserved by the mechanism."""
        tokens = ["z", "a", "-", "t1", "m", "b"]
        parser = TypedListParser.parse_typed_list(tokens)
        assert list(parser.get_typed_list_of_names().keys()) == ["z", "a", "m", "b"]
        assert [item for item, _ in parser.get_typed_list_of_variables()] == [
            "z",
            "a",
            "m",
            "b",
        ]

    def test_keyword_case_is_preserved(self) -> None:
        """Keywords are case-sensitive: 'And' is a name, 'and' is a keyword."""
        domain = DomainParser()(_domain_with_constants("And Or"))
        assert domain.constants == {Constant("And"), Constant("Or")}

        with pytest.raises(PDDLValidationError, match=".*it is a keyword"):
            DomainParser()(_domain_with_constants("and"))

    def test_keyword_case_is_preserved_for_objects(self) -> None:
        """The keyword case rule also holds for problem objects."""
        problem = ProblemParser()(_problem_with_objects("Either"))
        assert problem.objects == {Constant("Either")}

        with pytest.raises(PDDLValidationError, match=".*it is a keyword"):
            ProblemParser()(_problem_with_objects("either"))


class TestDiagnostics:
    """Diagnostics point to the user input and are sorted by source position."""

    def test_name_conflict_has_position_and_snippet(self) -> None:
        """A name conflict reports line, column and the input snippet."""
        domain_str = dedent(
            """
            (define (domain test)
                (:requirements :typing)
                (:types t1)
                (:constants c1 c2 - t1 c1)
            )"""
        )
        with pytest.raises(PDDLParsingError) as excinfo:
            DomainParser()(domain_str)
        message = str(excinfo.value)
        assert "duplicate name 'c1' in typed list already inherits" in message
        assert "(line 5, column 28)" in message
        assert "(:constants c1 c2 - t1 c1)" in message

    def test_multiple_name_conflicts_sorted_by_position(self) -> None:
        """Multiple name conflicts are reported together, sorted by position."""
        domain_str = dedent(
            """
            (define (domain test)
                (:requirements :typing)
                (:types t1)
                (:constants c1 c2 - t1 c2 c1)
            )"""
        )
        with pytest.raises(PDDLParsingError) as excinfo:
            DomainParser()(domain_str)
        message = str(excinfo.value)
        assert "found 2 errors" in message
        assert "duplicate name 'c2'" in message
        assert "duplicate name 'c1'" in message
        # sorted by source position: the second 'c2' occurs before the second 'c1'
        assert "(line 5, column 28)" in message
        assert "(line 5, column 31)" in message
        assert message.index("duplicate name 'c2'") < message.index(
            "duplicate name 'c1'"
        )

    def test_multiple_object_conflicts_sorted_by_position(self) -> None:
        """The same multi-error rule holds for problem objects."""
        problem_str = dedent(
            """
            (define (problem test-problem)
                (:domain test-domain)
                (:requirements :typing)
                (:objects b a b a)
                (:init )
                (:goal (and ))
            )"""
        )
        with pytest.raises(PDDLParsingError) as excinfo:
            ProblemParser()(problem_str)
        message = str(excinfo.value)
        assert "found 2 errors" in message
        assert message.index("duplicate name 'b'") < message.index(
            "duplicate name 'a'"
        )

    def test_unknown_types_sorted_by_position(self) -> None:
        """Unknown types keep the validation error category and gain positions."""
        domain_str = dedent(
            """
            (define (domain test)
                (:requirements :typing)
                (:types t1)
                (:predicates (p ?x - t2) (q ?y - t3))
            )"""
        )
        with pytest.raises(PDDLValidationError) as excinfo:
            DomainParser()(domain_str)
        message = str(excinfo.value)
        # the original validation message is preserved
        assert "are not in available types {'t1'}" in message
        # and all the offending occurrences are listed, sorted by position
        assert "type 't2' of variable 'x'" in message
        assert "type 't3' of variable 'y'" in message
        assert "(line 5, column 26)" in message
        assert "(line 5, column 38)" in message
        assert message.index("type 't2'") < message.index("type 't3'")

    def test_unknown_type_in_action_parameters(self) -> None:
        """Unknown types in action parameters are reported with positions."""
        domain_str = dedent(
            """
            (define (domain test)
                (:requirements :typing :strips)
                (:types t1)
                (:predicates (p))
                (:action act
                    :parameters (?x - t2)
                    :precondition (and)
                    :effect (and))
            )"""
        )
        with pytest.raises(PDDLValidationError) as excinfo:
            DomainParser()(domain_str)
        message = str(excinfo.value)
        assert "are not in available types {'t1'}" in message
        assert "type 't2' of variable 'x'" in message
        assert "(line 7, column 27)" in message

    def test_inheritance_cycle_has_declaration_positions(self) -> None:
        """Inheritance cycles point to the involved type declarations."""
        domain_str = dedent(
            """
            (define (domain test)
                (:requirements :typing)
                (:types a - b b - a)
            )"""
        )
        with pytest.raises(PDDLValidationError) as excinfo:
            DomainParser()(domain_str)
        message = str(excinfo.value)
        assert "cycle detected in the type hierarchy: a -> b" in message
        assert "type 'a' is part of the inheritance cycle" in message
        assert "type 'b' is part of the inheritance cycle" in message
        assert "(line 4, column 13)" in message
        assert "(line 4, column 19)" in message
        assert message.index("type 'a'") < message.index("type 'b'")

    def test_validation_not_anticipated(self) -> None:
        """More specific parse errors still take precedence over type resolution."""
        # the unknown type 't2' would be a validation error, but the duplicate
        # name is a parse error and must be reported first
        domain_str = dedent(
            """
            (define (domain test)
                (:requirements :typing)
                (:types t1)
                (:constants c1 - t2 c1)
            )"""
        )
        with pytest.raises(PDDLParsingError, match=".*duplicate name 'c1'"):
            DomainParser()(domain_str)


class TestSharedMechanism:
    """Prove the entry points share the same typed-list mechanism."""

    DOMAIN_STR = dedent(
        """
        (define (domain shapes)
            (:requirements :typing :strips :numeric-fluents :universal-preconditions)
            (:types t1 t2)
            (:constants c1 c2 - t1)
            (:predicates (p ?x - t1) (q))
            (:functions (f ?x - t2) (g))
            (:action act
                :parameters (?a - t1 ?b - t2)
                :precondition (forall (?v - t1) (p ?v))
                :effect (and (p ?a)))
        )"""
    )

    PROBLEM_STR = dedent(
        """
        (define (problem p1)
            (:domain shapes)
            (:requirements :typing)
            (:objects o1 o2 - t1 o3)
            (:init )
            (:goal (and ))
        )"""
    )

    @pytest.fixture()
    def typed_list_spy(self, monkeypatch):
        """Count the calls to the shared typed-list mechanism."""
        calls = []
        original = TypedListParser.parse_typed_list.__func__

        def _normalize(token):
            if isinstance(token, list):
                return "[" + ", ".join(map(str, token)) + "]"
            return str(token)

        def spy(cls, tokens, allow_duplicates=False, source=None):
            calls.append([_normalize(token) for token in tokens])
            return original(
                cls, tokens, allow_duplicates=allow_duplicates, source=source
            )

        monkeypatch.setattr(
            TypedListParser, "parse_typed_list", classmethod(spy)
        )
        return calls

    def test_domain_entry_points_share_the_mechanism(self, typed_list_spy) -> None:
        """Constants, parameters, quantifiers and functions use one mechanism."""
        DomainParser()(self.DOMAIN_STR)

        # one call per typed-list entry point: types, constants, two predicate
        # skeletons, two function skeletons, the function-name list, the action
        # parameters and the quantified variables
        assert len(typed_list_spy) == 9
        assert ["c1", "c2", "-", "t1"] in typed_list_spy  # domain constants
        assert ["a", "-", "[t1]", "b", "-", "[t2]"] in typed_list_spy
        assert ["v", "-", "[t1]"] in typed_list_spy  # quantified variables
        assert ["x", "-", "[t1]"] in typed_list_spy  # predicate arguments
        assert ["x", "-", "[t2]"] in typed_list_spy  # function arguments

    def test_problem_objects_share_the_mechanism(self, typed_list_spy) -> None:
        """Problem objects go through the same mechanism."""
        ProblemParser()(self.PROBLEM_STR)
        assert typed_list_spy == [["o1", "o2", "-", "t1", "o3"]]


class TestRoundTrips:
    """Parsing, formatting, pickling and copying keep structural equality."""

    DOMAIN_STR = TestSharedMechanism.DOMAIN_STR
    PROBLEM_STR = TestSharedMechanism.PROBLEM_STR

    def test_parse_format_parse_domain(self) -> None:
        """parse-format-parse keeps the domain structurally equal."""
        domain = DomainParser()(self.DOMAIN_STR)
        reparsed = DomainParser()(domain_to_string(domain))
        assert reparsed == domain
        assert domain_to_string(reparsed) == domain_to_string(domain)

    def test_parse_format_parse_problem(self) -> None:
        """parse-format-parse keeps the problem structurally equal."""
        problem = ProblemParser()(self.PROBLEM_STR)
        reparsed = ProblemParser()(problem_to_string(problem))
        assert reparsed == problem
        assert problem_to_string(reparsed) == problem_to_string(problem)

    def test_pickle_roundtrip(self) -> None:
        """Parsed domains and problems survive pickling unchanged."""
        domain = DomainParser()(self.DOMAIN_STR)
        problem = ProblemParser()(self.PROBLEM_STR)
        assert pickle.loads(pickle.dumps(domain)) == domain
        assert pickle.loads(pickle.dumps(problem)) == problem

    def test_deepcopy_roundtrip(self) -> None:
        """Parsed domains and problems survive deepcopy unchanged."""
        domain = DomainParser()(self.DOMAIN_STR)
        problem = ProblemParser()(self.PROBLEM_STR)
        assert copy.deepcopy(domain) == domain
        assert copy.deepcopy(problem) == problem

    def test_parsed_objects_equality_and_hash(self) -> None:
        """Parsed terms keep the equality/hash semantics of the public model."""
        domain = DomainParser()(_domain_with_constants("c1 - t1 c2"))
        assert Constant("c1", "t1") in domain.constants
        assert Constant("c2") in domain.constants
        assert hash(Constant("c1", "t1")) in {hash(c) for c in domain.constants}

        domain = DomainParser()(_domain_with_action_parameters("?x - t1"))
        (action,) = domain.actions
        (parameter,) = action.parameters
        assert parameter == Variable("x", ["t1"])
        assert hash(parameter) == hash(Variable("x", ["t1"]))
