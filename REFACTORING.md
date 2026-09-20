# Typed-list parsing refactoring

## Why

Typed lists appear in five syntactically identical places:

- domain constants (`:constants`)
- action parameters (`:parameters`)
- problem objects (`:objects`)
- quantified variables (`forall`/`exists`, including effect `forall`)
- predicate and numeric-function skeleton parameters
  (`:predicates`, `:functions`)

Before the refactoring, the grouping rules for these lists lived in
`pddl/parser/typed_list_parser.py` and were invoked from three transformer
callbacks (`typed_list_name`, `typed_list_variable`,
`f_typed_list_atomic_function_skeleton` in
`pddl/parser/domain.py`, with the problem transformer delegating back to the
domain one). Semantic type checks (unknown types, inheritance cycles) lived in
a *different* place: the public model constructors in `pddl/core.py` via
`pddl/_validation.py`. Diagnostics therefore could not refer back to the
user input: they reported object reprs without line/column or input fragment,
and only the first encountered failure surfaced.

## Old branches

- `TypedListParser.parse_typed_list` split tokens on the `-` separator and
  drove the index through a classmethod (`_add_typed_lists`) that mixed
  grouping, name/keyword normalization and duplicate policy.
- The three transformer callbacks each wrapped `parse_typed_list(...)` in
  their own `try/except ValueError -> PDDLParsingError` block, duplicating the
  entry/wrapping logic.
- Unknown-type errors were raised later, while constructing the public
  `Domain`/`Problem` objects (`TypeChecker`/`Types` in
  `pddl/_validation.py`), iterating sets — with no source position and no
  aggregation or deterministic ordering.

## Unified mechanism

Two internal modules now own the shared behavior:

- `pddl/parser/typed_lists.py`
  - `group_typed_list`: pure syntactic grouping of tokens into
    `(item, raw type tags, item location, type location)` tuples.
  - `normalize_typed_item` / `iter_typed_items`: name/keyword normalization
    shared by every entry (via `parse_name`/`parse_type`).
  - `TypedListIndex`: insertion-ordered index with the historical duplicate
    policy (names/constants/objects reject duplicates; variables may repeat
    when the tag sets are identical).
  - `TypedListDiagnostic` / `SourceLocation`: one failure anchored to the
    token (line, column, snippet).
- `pddl/parser/type_resolver.py`
  - `TypeResolver` records the `:types` hierarchy (with declaration
    locations) and every typed term occurrence.
  - `finalize()` runs the *deferred* semantic checks.

`pddl/parser/typed_list_parser.py` remains as a thin, public, backwards
compatible façade (`TypedListParser.parse_typed_list`, `add_item`,
`get_typed_list_of_names`, `get_typed_list_of_variables`) that re-raises
diagnostics as plain `ValueError`, exactly as before. The transformers no
longer call it; they call the shared internal machinery.

## Preserved invariants

- Untyped entries keep the default empty group (`None` for names/constants/
  objects, an empty tag set for variables).
- One or more consecutive names followed by `- type` share that type; a
  trailing block without a type is untyped; `(either t1 t2 ...)` still yields
  a multi-tag group.
- Keyword case sensitivity is unchanged: `object` remains the only keyword
  usable as a type tag; keyword-as-name/type still raises
  `PDDLValidationError`.
- Input order is preserved (ordered mapping for names/objects; ordered
  sequence for variables), so formatter output and public object equality are
  unchanged.
- Exception categories are unchanged at the public boundary:
  - duplicate/inconsistent typed-list entries raise `PDDLParsingError`
    (the legacy façade still raises plain `ValueError`);
  - unknown types and inheritance cycles raise `PDDLValidationError`.
- Public constructors, `__eq__`/`__hash__` of `Constant`/`Variable`/domains/
  problems, formatter output, and the grammar's acceptance of legal inputs
  are untouched.
- A single transformer instance can parse many files; per-document state
  (the type resolver, requirements, name-location stash) is reset in the
  `domain_def` callback, which is the first inner callback the LALR
  transformer invokes (the top-level `domain` callback runs last, after
  every inner section has been processed).
- The model-level checks in `pddl/_validation.py` are retained for objects
  built programmatically (not through a parser).

## Diagnostic priority

Grouping/normalization failures are raised immediately at the offending
entry, so a syntactic/duplicate error is never hidden by a later semantic
check. Examples, in priority order:

1. Lark syntax errors (illegal input).
2. Keyword and malformed-name errors during normalization.
3. Missing-requirement and typed-list duplicate/conflict errors
   (`PDDLParsingError`).
4. Deferred semantic checks in `TypeResolver.finalize()`: inheritance cycle
   first, then **all** unknown-type diagnostics.

Semantic diagnostics carry `at line L, column C: 'fragment'` and are sorted by
source position `(line, column)`. With a single error the historical message
text is preserved verbatim (the position is a suffix); multiple errors are
joined by a blank line.
