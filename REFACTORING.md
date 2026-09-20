# Typed-list mechanism refactoring

This note documents how typed-list grouping and type resolution were
consolidated into a single internal mechanism, which invariants are
preserved, and how diagnostics are prioritized.

## Old branches

Before the refactoring, the same concerns were spread over several places:

- **Grouping** lived in `pddl/parser/typed_list_parser.py`
  (`TypedListParser.parse_typed_list`), called from three rules of
  `DomainTransformer` (`typed_list_name`, `typed_list_variable`,
  `f_typed_list_atomic_function_skeleton`) and, by delegation, from
  `ProblemTransformer`. Structural conflicts (duplicate names, duplicate or
  inconsistent type tags) were raised from inside `add_item` at the *first*
  occurrence, so only one problem was reported and it carried no source
  position.
- **Type resolution** (unknown types, inheritance cycles) existed only at
  domain-construction time, in `pddl/_validation.py`
  (`Types._check_types_dictionary`, `TypeChecker`), which has no access to
  parser positions.
- The transformers never saw the input text, so no diagnostic could point
  back to the user input.

## Unified mechanism

Everything now lives in `pddl/parser/typed_list_parser.py`:

- `TypedListParser` (grouping) records the source position of every item and
  type-tag token (Lark tokens carry line/column), *collects* structural
  diagnostics instead of raising at the first one, and raises a single
  `ValueError` at the end of grouping with all diagnostics sorted by
  (line, column, token index). The three domain rules and the problem
  `objects` rule all go through `parse_typed_list`; a test-side call counter
  (`tests/test_parser/test_typed_list_mechanism.py`) proves the entry points
  share this one mechanism.
- `SourcePosition` / `TypedListDiagnostic` render `message (line L, column
  C): <input line>`; the input snippet is recovered from the source text
  that `BaseParser.__call__` threads into the transformers (`source_text`,
  forwarded by `ProblemTransformer` to the shared `DomainTransformer`).
- `TypeResolver` (type resolution) mirrors the construction-time checks of
  `pddl/_validation.py` (unknown type tags, inheritance cycles via
  `find_cycle`) on the parser-side representation. It never raises: it
  returns diagnostics sorted by source position. `DomainTransformer`
  registers every typed list whose tags are also checked at construction
  time (constants, predicate skeletons, action parameters, quantified
  variables — exactly the sets `Domain._check_consistency` validates) and,
  when `Domain(**kwargs)` raises a `PDDLValidationError` that *is* a
  type-resolution error, re-raises it with the sorted, position-annotated
  diagnostics appended. Validation is therefore **not** anticipated: parse
  errors and more specific validation errors keep their precedence.

## Preserved invariants

- Untyped items get the default assignment (no type tag / empty tag set).
- Several consecutive names followed by one type all inherit that type.
- Keyword checks stay case-sensitive (`And` is a valid name, `and` is a
  keyword).
- Input order is preserved (insertion-ordered name maps, sequence of
  variable/tag pairs).
- Exception categories are unchanged: `PDDLParsingError` for parse-level
  conflicts, `PDDLValidationError` for name/keyword validation and for type
  resolution, `PDDLMissingRequirementError` for missing requirements.
- Public constructors, `__eq__`/`__hash__` of terms, formatter output and
  the grammar's acceptance of valid inputs are untouched; parse-format-parse
  round trips, `pickle` and `deepcopy` behave as before (guarded by tests).

## Diagnostic priority

1. **Per-token name/keyword validation** (`PDDLValidationError`) — raised
   immediately while grouping; the most specific errors.
2. **Structural conflicts** in a typed list (duplicates, inconsistent tags)
   — collected during grouping and reported together as one
   `PDDLParsingError`, entries sorted by source position.
3. **Type resolution** (unknown types, inheritance cycles) — owned by
   domain-object validation; the parser only enriches the raised
   `PDDLValidationError` with line/column and input snippets, and only when
   the error itself is a type-resolution error. Multiple unknown types are
   listed sorted by source position.

Because (3) runs strictly after a successful parse and never replaces an
error of a different kind, no earlier or more specific error can be masked.
