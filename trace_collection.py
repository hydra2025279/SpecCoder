from __future__ import annotations

import ast
import io
import multiprocessing
import queue as queue_mod
import re
import signal
import sys
import threading
import time
from typing import Any, Optional

from SpecCoder.spec_utils import AssertionInfo, parse_assert_comment


IMPORT_STRING = (
    "import sys\nfrom typing import *\nimport collections\nimport math\n"
    "import itertools\nimport functools\nimport heapq\nimport bisect\n"
    "import string\nimport re\nimport json\nfrom collections import *\n"
    "from itertools import *\nfrom functools import *\nfrom heapq import *\n"
    "from bisect import *\nfrom math import *\nsys.setrecursionlimit(50000)\n"
)

SPEC_CHECK_HEADER_TRACE = """\
_SPEC_TRACE = {}
_MUTANTS_REACHED = []
_CO_REACH = {}

class _EofIO:
    # Predicates must be PURE expressions over program state. If a generated spec
    # accidentally calls input()/sys.stdin.read (e.g. `... len(input().split()) == n`),
    # evaluating it inline would CONSUME the program's real stdin and starve every later
    # statement (EOFError -> crash -> all later checkpoints report 0 reaches). We swap
    # stdin to this EOF stream during predicate eval so such a predicate fails for itself
    # only, leaving the program's stdin/stdout untouched.
    def readline(self, *a): return ''
    def read(self, *a): return ''
    def write(self, *a): return 0
    def flush(self): pass
    def __iter__(self): return iter(())
    def __next__(self): raise StopIteration
_EOF_IO = _EofIO()

def _trace_key(dedup_key, assertion):
    return f"{dedup_key}:{assertion}"

def _check_mutant_reach(dedup_key, original_content):
    _MUTANTS_REACHED.append(f"{dedup_key}:{original_content}")

def _check_trace(dedup_key, assertion, expr_fn):
    line_id = _trace_key(dedup_key, assertion)
    if _MUTANTS_REACHED:
        _CO_REACH.setdefault(line_id, set()).update(_MUTANTS_REACHED)
    _saved_stdin, _saved_stdout = sys.stdin, sys.stdout
    sys.stdin, sys.stdout = _EOF_IO, _EOF_IO
    try:
        result = bool(expr_fn())
        _SPEC_TRACE.setdefault(line_id, []).append(result)
    except TimeoutError:
        _SPEC_TRACE.setdefault(line_id, []).append(False)
        raise
    except Exception:
        _SPEC_TRACE.setdefault(line_id, []).append(False)
    finally:
        sys.stdin, sys.stdout = _saved_stdin, _saved_stdout
"""


def mutant_reach_id(dedup_key: Any, original_content: str) -> str:
    return f"{dedup_key}:{original_content}"


def instrument_mutant_lines_in_completion(
    annotated_completion: str,
    mutant_records: list[dict[str, Any]],
) -> str:
    from SpecCoder.spec_utils import find_mutant_record_line_index

    lines = annotated_completion.split("\n")
    depth_before = [0]
    for s in lines:
        depth_before.append(depth_before[-1] + _bracket_delta(s))

    def _logical_start(idx: int) -> int:
        start = idx
        while start > 0 and (depth_before[start] > 0 or lines[start - 1].rstrip().endswith("\\")):
            start -= 1
        return start

    insertions: dict[int, list[tuple[Any, str]]] = {}
    for record in mutant_records or []:
        if "completion" in record:
            continue
        line_idx = find_mutant_record_line_index(annotated_completion, record)
        if line_idx is None:
            continue
        dedup_key = record.get("dedup_key", 0)
        content = str(record.get("original_content", ""))
        if not content:
            continue
        # A mutated line may be a continuation physical line of a multi-line logical
        # statement (carried over by an open bracket / trailing backslash). A marker
        # cannot be inserted there -- it would land inside the open bracket. Anchor it
        # to the statement's first physical line instead (reach is equivalent).
        insertions.setdefault(_logical_start(line_idx), []).append((dedup_key, content))

    if not insertions:
        return annotated_completion

    def _body_indent(idx: int, header_indent: str) -> str:
        """Indent of the block whose header is line ``idx``: the indent of the next
        non-blank line that is more indented than the header. Falls back to header+4."""
        for j in range(idx + 1, len(lines)):
            stripped = lines[j].strip()
            if not stripped:
                continue
            ind = lines[j][: len(lines[j]) - len(lines[j].lstrip())]
            return ind if len(ind) > len(header_indent) else header_indent + "    "
        return header_indent + "    "

    unit = _indent_unit(annotated_completion)
    out: list[str] = []
    for i, line in enumerate(lines):
        markers = insertions.get(i)
        if not markers:
            out.append(line)
            continue
        indent = line[: len(line) - len(line.lstrip())]
        first_word = _leading_keyword(line)
        # A reach marker is a STATEMENT. Emitting it before a continuation clause
        # (else/elif/except/finally/case) would split the clause from its block ->
        # SyntaxError that nukes the whole program (every checkpoint then reports 0
        # reaches). For those lines, mark reach as the FIRST statement of the clause's
        # body instead; for any other line, mark right before it as usual.
        if first_word in ("else", "elif", "except", "finally", "case"):
            split = _split_compound_oneliner(line)
            if split is not None:
                # One-liner clause (`else: body`): there is no suite to mark into, so
                # expand it onto its own line and put the marker(s) in the new suite.
                header, body = split
                body_indent = indent + unit
                out.append(header)
                out.append(f"{body_indent}{body}")
                for dedup_key, content in markers:
                    out.append(f"{body_indent}_check_mutant_reach({dedup_key!r}, {content!r})")
            else:
                out.append(line)
                body = _body_indent(i, indent)
                for dedup_key, content in markers:
                    out.append(f"{body}_check_mutant_reach({dedup_key!r}, {content!r})")
        else:
            for dedup_key, content in markers:
                out.append(f"{indent}_check_mutant_reach({dedup_key!r}, {content!r})")
            out.append(line)
    return "\n".join(out)


def _is_valid_eval_expr(expr: str) -> bool:
    try:
        ast.parse(expr, mode="eval")
    except SyntaxError:
        return False
    return True


_CONT_KEYWORDS = ("else", "elif", "except", "finally", "case")
_COMPOUND_KEYWORDS = (
    "if", "elif", "else", "for", "while", "with", "try", "except",
    "finally", "def", "class", "async", "match", "case",
)


def _leading_indent(s: str) -> str:
    return s[: len(s) - len(s.lstrip())]


def _leading_keyword(s: str) -> str:
    m = re.match(r"\s*([A-Za-z_]+)", s)
    return m.group(1) if m else ""


def _block_colon_index(line: str) -> int | None:
    """Index of the ``:`` that ends a compound-statement header on ``line``, i.e.
    the first ``:`` at bracket-depth 0 outside any string literal. Returns None if
    there is no such colon (e.g. a simple statement, or a multi-line header)."""
    depth = 0
    quote: str | None = None
    i = 0
    while i < len(line):
        ch = line[i]
        if quote is not None:
            if ch == "\\":
                i += 2
                continue
            if ch == quote:
                quote = None
        elif ch in ("'", '"'):
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == ":" and depth == 0:
            return i
        i += 1
    return None


def _split_compound_oneliner(line: str) -> tuple[str, str] | None:
    """If ``line`` is a one-line compound statement (``header: body``), return
    ``(header_with_colon, body)``; otherwise None. Used so a checkpoint placed
    after the body of a one-liner (before an ``else``/``except`` clause) can be
    moved onto its own line instead of orphaning the continuation clause."""
    if _leading_keyword(line) not in _COMPOUND_KEYWORDS:
        return None
    ci = _block_colon_index(line)
    if ci is None:
        return None
    body = line[ci + 1 :].strip()
    if not body or body.startswith("#"):
        return None
    return line[: ci + 1], body


def _is_block_opener_no_body(line: str) -> bool:
    """True for a compound-statement header whose body is on following lines
    (e.g. ``if x:``, ``try:``) — i.e. the next code line starts a fresh block."""
    if _leading_keyword(line) not in _COMPOUND_KEYWORDS:
        return False
    ci = _block_colon_index(line)
    if ci is None:
        return False
    tail = line[ci + 1 :].strip()
    return not tail or tail.startswith("#")


def _indent_unit(text: str) -> str:
    """A single indentation level matching the file's style (tabs vs spaces)."""
    for line in text.split("\n"):
        ind = _leading_indent(line)
        if ind:
            return "\t" if ind[0] == "\t" else "    "
    return "    "


def _bracket_delta(line: str) -> int:
    """Net ``([{`` opens minus ``)]}`` closes on ``line``, ignoring brackets that
    appear inside string literals or after a ``#`` comment."""
    depth = 0
    quote: str | None = None
    i = 0
    while i < len(line):
        ch = line[i]
        if quote is not None:
            if ch == "\\":
                i += 2
                continue
            if ch == quote:
                quote = None
        elif ch in ("'", '"'):
            quote = ch
        elif ch == "#":
            break
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        i += 1
    return depth


def _last_code_index(out: list[str]) -> int | None:
    for j in range(len(out) - 1, -1, -1):
        s = out[j]
        if s.strip() and not s.lstrip().startswith("#"):
            return j
    return None


def _logical_start_index(out: list[str], pj: int) -> int:
    """First physical-line index of the logical statement whose last physical line
    is ``out[pj]``. Continuation lines (open brackets carried over, or a trailing
    backslash) inherit an arbitrary indent, so the *logical start* is what gives the
    statement's real indentation."""
    depth_before = [0]
    for s in out:
        depth_before.append(depth_before[-1] + _bracket_delta(s))
    start = pj
    while start > 0 and (depth_before[start] > 0 or out[start - 1].rstrip().endswith("\\")):
        start -= 1
    return start


def _prev_indent(out: list[str], pj: int) -> str:
    return _leading_indent(out[_logical_start_index(out, pj)])


def _enclosing_indent(out: list[str], width: int) -> str:
    """Indent string of the DEEPEST preceding code statement whose indentation is
    ``<= width`` — i.e. a legal, still-open enclosing block level for a comment that
    dedents to roughly ``width`` columns. Used when a ``# assert`` dedents OUT of the
    preceding (deeper) block: snapping to this level puts the checkpoint at the scope
    its own indentation indicates (a reachable sibling) instead of burying it inside
    the block it has exited. Module level (``""``) always qualifies as a fallback."""
    best = ""
    for s in out:
        if not s.strip() or s.lstrip().startswith("#"):
            continue
        ind = _leading_indent(s)
        if len(ind) <= width and len(ind) >= len(best):
            best = ind
    return best


def _next_code_index(lines: list[str], i: int) -> int | None:
    for j in range(i + 1, len(lines)):
        s = lines[j]
        if s.strip() and not s.lstrip().startswith("#"):
            return j
    return None


def instrument_assertions_in_completion(
    annotated_completion: str,
    assertions: list[AssertionInfo],
    check_fn: str = "_check_trace",
) -> str:
    """Replace selected `# assert ...` comments with executable trace calls.

    Each checkpoint is identified by its `(assert_key, dedup_key)` and the matching
    `# assert` comment line is replaced IN PLACE by a ``_check_trace(...)`` call at the
    comment's OWN indentation. The annotating LLM already places the comment at the
    column of the scope it intends — a postcondition dedents back to module level, an
    in-loop invariant sits at the loop-body column — so we trust that column rather than
    recomputing one from the surrounding code (which buried dedented postconditions inside
    the preceding block / after a `break`, where they were dead code and never executed).
    The predicate and its trace key are unchanged.
    """
    wanted = {assertion.trace_key: assertion for assertion in assertions}
    lines = annotated_completion.split("\n")
    seen: dict[str, int] = {}
    out: list[str] = []
    for i, line in enumerate(lines):
        expr = parse_assert_comment(line)
        if expr is None:
            out.append(line)
            continue
        assert_key = f"assert {expr}"
        dedup_key = seen.get(assert_key, 0)
        seen[assert_key] = dedup_key + 1
        trace_key = f"{dedup_key}:{assert_key}"
        if trace_key not in wanted:
            out.append(line)
            continue
        expr_fn = f"lambda: ({expr})" if _is_valid_eval_expr(expr) else "lambda: False"
        stmt_tail = f"{check_fn}({dedup_key!r}, {assert_key!r}, {expr_fn})"
        # Replace the `# assert` comment IN PLACE, keeping the COMMENT'S OWN INDENT. The
        # annotating LLM places each checkpoint at the column of the scope it means (a
        # postcondition dedents back to module level; an in-loop invariant sits at the
        # loop-body column), so the comment's indentation IS the intended one. We do not
        # recompute indent from surrounding code — that mis-fired for dedented checkpoints
        # (a module-level postcondition after a nested `break` got buried at the break's
        # deep indent as dead code -> 0 reaches -> silently dropped).
        indent = line[: len(line) - len(line.lstrip())]
        out.append(f"{indent}{stmt_tail}")
    return "\n".join(out)


def evaluate_one_test_trace(
    code: str,
    spec_dict: dict[str, Any],
    func_name: str,
    params: list[str],
    args: list[Any],
    timeout: int = 5,
    trace_keys: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    full = IMPORT_STRING + SPEC_CHECK_HEADER_TRACE + "\n" + code
    namespace: dict[str, Any] = {}
    for param, value in zip(params, args):
        namespace[param] = value

    try:
        exec(compile(full, "<spec>", "exec"), namespace)
    except Exception:
        return {
            key: {"pass_count": 0, "reach_count": 0, "completed": False, "error": "setup failed"}
            for key in spec_dict
        }

    call_code = f"return_value = Solution().{func_name}({', '.join(params)})"
    use_alarm = hasattr(signal, "SIGALRM") and threading.current_thread() is threading.main_thread()
    completed = False
    error: Optional[str] = None

    if use_alarm:
        def _alarm_handler(signum, frame):
            raise TimeoutError("Trace timeout")

        old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(timeout)

    try:
        exec(compile(call_code, "<call>", "exec"), namespace)
        completed = True
    except TimeoutError as exc:
        error = str(exc)
    except SystemExit:
        # exit()/sys.exit() inside the solution is a normal termination; keep the
        # checkpoints already recorded in _SPEC_TRACE instead of dropping the test.
        completed = True
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if use_alarm:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

    results: dict[str, dict[str, Any]] = {}
    trace_dict = namespace.get("_SPEC_TRACE", {}) or {}
    co_reach_dict = namespace.get("_CO_REACH", {}) or {}
    expr_by_key = {key: "" for key in (trace_keys or list(spec_dict))}

    for key in expr_by_key:
        evaluations = trace_dict.get(str(key), [])
        results[key] = {
            "pass_count": sum(1 for result in evaluations if result is True),
            "reach_count": len(evaluations),
            "completed": completed,
            "error": error,
            "co_reach": set(co_reach_dict.get(str(key), set()) or set()),
        }

    for key in expr_by_key:
        results.setdefault(
            key,
            {"pass_count": 0, "reach_count": 0, "completed": completed, "error": error, "co_reach": set()},
        )
    return results


def evaluate_one_test_trace_stdin(
    code: str,
    spec_dict: dict[str, Any],
    stdin_text: str,
    timeout: int = 5,
    trace_keys: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Codeforces (stdin/stdout) analogue of ``evaluate_one_test_trace``.

    ``code`` is a top-level script that reads via ``input()``/``sys.stdin`` and
    writes via ``print()``. It runs during ``exec``; the inlined
    ``_check_trace(...)`` calls fire as the script executes, evaluating each
    ``# assert`` against the real variables in scope at that point. There is no
    ``Solution()``/``func_name``/``params``/``return_value`` — inputs arrive on
    stdin, the answer is whatever the script computes before it prints.
    """
    full = IMPORT_STRING + SPEC_CHECK_HEADER_TRACE + "\n" + code
    # Run as the main module: many codeforces solutions guard their entry point with
    # ``if __name__ == '__main__':`` (read stdin + call the solver there). With an empty
    # namespace ``__name__`` is unset, so that block is skipped, nothing reads input, the
    # solver is never called and EVERY checkpoint gets 0 reaches (on passing AND failing
    # tests alike). Setting ``__name__`` makes the trace run match the real subprocess run.
    namespace: dict[str, Any] = {"__name__": "__main__"}

    try:
        compiled = compile(full, "<spec-stdin>", "exec")
    except Exception:
        return {
            key: {"pass_count": 0, "reach_count": 0, "completed": False, "error": "setup failed"}
            for key in spec_dict
        }

    old_stdin, old_stdout = sys.stdin, sys.stdout
    sys.stdin = io.StringIO(stdin_text)
    sys.stdout = io.StringIO()

    use_alarm = hasattr(signal, "SIGALRM") and threading.current_thread() is threading.main_thread()
    completed = False
    error: Optional[str] = None

    if use_alarm:
        def _alarm_handler(signum, frame):
            raise TimeoutError("Trace timeout")

        old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(timeout)

    try:
        exec(compiled, namespace)
        completed = True
    except TimeoutError as exc:
        error = str(exc)
    except SystemExit:
        # The program called exit()/sys.exit() (common in buggy branches). That is a
        # normal termination, NOT a trace failure: every checkpoint reached before the
        # exit was already recorded in _SPEC_TRACE during exec. Treat it as completed so
        # those reach/pass counts are returned (otherwise SystemExit, a BaseException,
        # escapes both this function and the worker, dropping the whole test → reach=0).
        completed = True
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if use_alarm:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
        sys.stdin, sys.stdout = old_stdin, old_stdout

    results: dict[str, dict[str, Any]] = {}
    trace_dict = namespace.get("_SPEC_TRACE", {}) or {}
    co_reach_dict = namespace.get("_CO_REACH", {}) or {}
    expr_by_key = {key: "" for key in (trace_keys or list(spec_dict))}

    for key in expr_by_key:
        evaluations = trace_dict.get(str(key), [])
        results[key] = {
            "pass_count": sum(1 for result in evaluations if result is True),
            "reach_count": len(evaluations),
            "completed": completed,
            "error": error,
            "co_reach": set(co_reach_dict.get(str(key), set()) or set()),
        }

    for key in expr_by_key:
        results.setdefault(
            key,
            {"pass_count": 0, "reach_count": 0, "completed": completed, "error": error, "co_reach": set()},
        )
    return results


def _trace_worker(code, spec_dict, func_name, params, args, timeout, q, trace_keys=None):
    try:
        q.put(
            evaluate_one_test_trace(
                code,
                spec_dict,
                func_name,
                params,
                args,
                timeout=timeout,
                trace_keys=trace_keys,
            )
        )
    except Exception:
        q.put(None)


def kill_proc(proc, grace_seconds: int = 2) -> None:
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=grace_seconds)
    if proc.is_alive():
        proc.kill()
        proc.join(timeout=grace_seconds)


def run_one_test_trace_subprocess(code, spec_dict, func_name, params, args, timeout, trace_keys=None):
    ctx = multiprocessing.get_context("fork")
    q = ctx.Queue()
    proc = ctx.Process(
        target=_trace_worker,
        args=(code, spec_dict, func_name, params, args, timeout, q, trace_keys),
    )
    proc.start()
    deadline = time.monotonic() + timeout + 5
    result = None
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            result = q.get(timeout=min(0.2, remaining))
            break
        except queue_mod.Empty:
            if not proc.is_alive():
                try:
                    result = q.get_nowait()
                except queue_mod.Empty:
                    pass
                break
    kill_proc(proc)
    if result is None:
        return None, True, "Trace timeout (subprocess killed)"
    soft_timeout = any(v.get("error") == "Trace timeout" for v in result.values() if isinstance(v, dict))
    return result, False, "Trace timeout" if soft_timeout else None


def _trace_worker_stdin(code, spec_dict, stdin_text, timeout, q, trace_keys=None):
    try:
        q.put(
            evaluate_one_test_trace_stdin(
                code,
                spec_dict,
                stdin_text,
                timeout=timeout,
                trace_keys=trace_keys,
            )
        )
    except Exception:
        q.put(None)


def run_one_test_trace_subprocess_stdin(code, spec_dict, stdin_text, timeout, trace_keys=None):
    ctx = multiprocessing.get_context("fork")
    q = ctx.Queue()
    proc = ctx.Process(
        target=_trace_worker_stdin,
        args=(code, spec_dict, stdin_text, timeout, q, trace_keys),
    )
    proc.start()
    deadline = time.monotonic() + timeout + 5
    result = None
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            result = q.get(timeout=min(0.2, remaining))
            break
        except queue_mod.Empty:
            if not proc.is_alive():
                try:
                    result = q.get_nowait()
                except queue_mod.Empty:
                    pass
                break
    kill_proc(proc)
    if result is None:
        return None, True, "Trace timeout (subprocess killed)"
    soft_timeout = any(v.get("error") == "Trace timeout" for v in result.values() if isinstance(v, dict))
    return result, False, "Trace timeout" if soft_timeout else None
