from __future__ import annotations

import ast
from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class PlacementCandidate:
    id: str
    line_no: int
    end_line_no: int
    placement: str
    kind: str
    code: str
    indent: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class InlineSpecPlacementExtractor(ast.NodeVisitor):
    """Extract candidate lines using AST statement categories.

    Statement categories:
    - is_simple_statement: Assign, AugAssign, Raise, Pass, Break, Continue,
      and non-print Expr.
    - is_print: Expr(Call(Name("print"))).
    - structured nodes: If, Try, For, While, Return.

    Adaptation:
    - Instead of calling an LLM while traversing, we return candidate locations
      for symbolic `# assert ...` insertion.
    """

    def __init__(self, source: str):
        self.source = source
        self.lines = source.splitlines()
        self.candidates: list[PlacementCandidate] = []
        self._seen: set[tuple[int, str, str]] = set()

    def is_simple_statement(self, command: ast.AST) -> bool:
        return isinstance(command, (ast.Assign, ast.AugAssign, ast.Raise, ast.Pass, ast.Break, ast.Continue)) or (
            isinstance(command, ast.Expr) and not self.is_print(command)
        )

    def is_print(self, command: ast.AST) -> bool:
        return (
            isinstance(command, ast.Expr)
            and isinstance(command.value, ast.Call)
            and isinstance(command.value.func, ast.Name)
            and command.value.func.id == "print"
        )

    def add_candidate(self, node: ast.AST, kind: str, placement: str = "after") -> None:
        if not hasattr(node, "lineno"):
            return
        line_no = int(getattr(node, "lineno"))
        end_line_no = int(getattr(node, "end_lineno", line_no))
        target_line = end_line_no if placement == "after" else line_no
        key = (target_line, kind, placement)
        if key in self._seen:
            return
        self._seen.add(key)

        snippet = ast.get_source_segment(self.source, node) or self.lines[line_no - 1].strip()
        raw_line = self.lines[target_line - 1] if 0 <= target_line - 1 < len(self.lines) else ""
        indent = len(raw_line) - len(raw_line.lstrip())
        candidate_id = f"L{target_line}_{len(self.candidates) + 1}"
        self.candidates.append(
            PlacementCandidate(
                id=candidate_id,
                line_no=target_line,
                end_line_no=end_line_no,
                placement=placement,
                kind=kind,
                code=snippet.strip(),
                indent=indent,
            )
        )

    def visit_Assign(self, node: ast.Assign) -> None:
        self.add_candidate(node, "critical_assignment", "after")
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self.add_candidate(node, "critical_assignment", "after")
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self.add_candidate(node, "critical_assignment", "after")
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> None:
        self.generic_visit(node)
        self.add_candidate(node, "branch_join", "after")

    def visit_Try(self, node: ast.Try) -> None:
        self.generic_visit(node)
        self.add_candidate(node, "try_join", "after")

    def visit_For(self, node: ast.For) -> None:
        self.generic_visit(node)
        self.add_candidate(node, "loop_exit", "after")

    def visit_While(self, node: ast.While) -> None:
        self.generic_visit(node)
        self.add_candidate(node, "loop_exit", "after")

    def visit_Return(self, node: ast.Return) -> None:
        self.add_candidate(node, "return_state", "before")
        self.generic_visit(node)

    def visit_Expr(self, node: ast.Expr) -> None:
        if self.is_print(node):
            self.add_candidate(node, "print_state", "after")
        self.generic_visit(node)


def extract_placement_candidates(source: str, max_candidates: int | None = 30) -> list[dict[str, Any]]:
    tree = ast.parse(source)
    extractor = InlineSpecPlacementExtractor(source)
    extractor.visit(tree)
    candidates = extractor.candidates
    if max_candidates is not None:
        candidates = candidates[:max_candidates]
    return [candidate.to_dict() for candidate in candidates]


def format_candidates_for_prompt(candidates: list[dict[str, Any]]) -> str:
    rows = []
    for candidate in candidates:
        rows.append(
            f"- {candidate['id']}: {candidate['placement']} line {candidate['line_no']} "
            f"({candidate['kind']})\n"
            f"  Code: {candidate['code']}"
        )
    return "\n".join(rows) if rows else "- No candidate lines found; add only essential assertions."
