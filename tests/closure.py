"""
Which of the repository's own modules a Python file needs, followed through
every import it makes — at the top, inside functions, behind conditions.

Lists of files kept by hand (the engine a lab freezes, the modules the browser
is sent) are only right while they equal this, so the tests compare them to it
rather than trusting them.
"""

from __future__ import annotations

import ast
import os
from typing import Dict, Iterable


def imported_names(path: str) -> set:
    """The top-level module names a file imports, wherever in it the import sits."""
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=path)
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def closure(path: str, search: Iterable[str]) -> Dict[str, str]:
    """
    Every local module `path` reaches, as {name: file}. A name resolves to the
    first directory of `search` that holds it, as Python's path would; names
    found in none of them are someone else's (the standard library, numpy).
    """
    search = list(search)
    found: Dict[str, str] = {}
    todo = [path]
    while todo:
        for name in imported_names(todo.pop()):
            if name in found:
                continue
            for folder in search:
                candidate = os.path.join(folder, f"{name}.py")
                if os.path.isfile(candidate):
                    found[name] = candidate
                    todo.append(candidate)
                    break
    return found
