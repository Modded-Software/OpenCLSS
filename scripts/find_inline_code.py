#!/usr/bin/env python3
"""Find interpreter payloads and embedded program text carried inside other files.

A file that hands a program to an interpreter as a string is an unscanned code
channel: the payload never appears as a file of its own, so no reviewer and no
linter ever sees it as code. This walks a tree and reports two forms:

  * interpreter invocation - an argv sequence that names an interpreter and its
    inline flag (``-c``, ``-e``, ``-r``), e.g. ``["python", "-I", "-c", program]``;
  * embedded program text - a multi-line string constant whose lines parse as
    Python statements, or whose text spells out an interpreter invocation.

Usage:
    uv run python scripts/find_inline_code.py [root ...]

With no root, the tree holding this script is walked.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

INLINE_FLAGS = {"-c", "-e", "-r"}
CODE_STATEMENTS = (
    ast.Import,
    ast.ImportFrom,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.If,
    ast.With,
    ast.AsyncWith,
    ast.Try,
    ast.Return,
    ast.Raise,
    ast.Assert,
    ast.Delete,
    ast.Global,
    ast.Nonlocal,
)
PROGRAM_NAMES = {
    "python",
    "python2",
    "python3",
    "py",
    "perl",
    "node",
    "ruby",
    "php",
    "bash",
    "sh",
    "zsh",
    "dash",
    "lua",
    "tclsh",
    "wish",
}
TEXT_SUFFIXES = {
    ".py",
    ".pyi",
    ".sh",
    ".bash",
    ".zsh",
    ".js",
    ".mjs",
    ".cjs",
    ".ts",
    ".tsx",
    ".rb",
    ".pl",
    ".php",
    ".lua",
    ".rs",
    ".c",
    ".h",
    ".cc",
    ".cpp",
    ".hpp",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".txt",
    ".md",
}
SKIP_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    "node_modules",
    "build",
    "dist",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    "work",
    "target",
}
_FLAG_ALT = "|".join(sorted(re.escape(flag) for flag in INLINE_FLAGS))
_NAME_ALT = "|".join(sorted([re.escape(name) for name in PROGRAM_NAMES], key=len, reverse=True))
INVOCATION_TEXT = re.compile(rf"\b(?:{_NAME_ALT})[0-9.]*\s+(?:{_FLAG_ALT})\b")


def _basename(value):
    return Path(value).name.lower().removesuffix(".exe")


def _argv_strings(node):
    if not isinstance(node, (ast.List, ast.Tuple)):
        return None
    values = [
        item.value
        for item in node.elts
        if isinstance(item, ast.Constant) and isinstance(item.value, str)
    ]
    return values or None


def _invoked_here(values):
    names = {_basename(value) for value in values}
    return bool(names & PROGRAM_NAMES and set(values) & INLINE_FLAGS)


def _looks_like_program(text):
    """A multi-line string that is a program: it parses as a module and holds a
    statement only source code carries. Prose, config and data do not qualify."""
    if "\n" not in text:
        return False
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return False
    for node in tree.body:
        if isinstance(node, CODE_STATEMENTS):
            return True
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            return True
    return False


def _findings_in_python(path):
    try:
        tree = ast.parse(path.read_text(encoding="utf-8-sig", errors="replace"))
    except (SyntaxError, ValueError) as error:
        sys.stderr.write(f"find-inline-code: cannot parse {path}: {error}\n")
        return
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if _looks_like_program(node.value) or INVOCATION_TEXT.search(node.value):
                yield node.lineno, "embedded-program-text", _snippet(node.value)
            continue
        strings = _argv_strings(node)
        if strings and _invoked_here(strings):
            yield node.lineno, "interpreter-invocation", " ".join(strings[:5])


def _findings_in_text(path):
    try:
        lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    except (OSError, ValueError) as error:
        sys.stderr.write(f"find-inline-code: cannot read {path}: {error}\n")
        return
    for number, line in enumerate(lines, start=1):
        if INVOCATION_TEXT.search(line):
            yield number, "interpreter-invocation", line.strip()


def _snippet(text):
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return ""


def _walk(root):
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        yield path


def main(argv):
    roots = [Path(arg).resolve() for arg in argv] or [Path(*Path(__file__).resolve().parts[:-2])]
    self_path = Path(__file__).resolve()
    total = 0
    for root in roots:
        for path in _walk(root):
            if path.resolve() == self_path:
                continue
            source = (
                _findings_in_python(path)
                if path.suffix.lower() in (".py", ".pyi")
                else _findings_in_text(path)
            )
            for number, category, snippet in source:
                total += 1
                print(f"{path.relative_to(root)}:{number}: {category}: {snippet}")
    print(f"\n{total} inline-code finding(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
