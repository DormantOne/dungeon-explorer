#!/usr/bin/env python3
"""
KG Code Mapper v4.1 (Smart Locator Edition)
===========================================

v4.1 replaces Code CRISPR with a "Smart Locator" workflow:
1. Generate Context (Map/Code) -> Copy to LLM.
2. LLM provides a fixed code block.
3. Paste fix into "Smart Locator".
4. Tool fuzzy-locates the area in the file and highlights it.
5. User pastes/overwrites the fix directly.

Retains the robust Code Map generation and Query features of v3.7.
"""

from __future__ import annotations

import ast
import bisect
import difflib
import hashlib
import json
import os
import re
import sys
import textwrap
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# ============================================================================
# TKINTER IMPORT (OPTIONAL)
# ============================================================================

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, scrolledtext, messagebox
    HAS_TK = True
except ImportError:
    HAS_TK = False


# ============================================================================
# FILE TYPE CONFIGURATION
# ============================================================================

FILE_TYPES: Dict[str, Dict[str, Any]] = {
    "python": {
        "extensions": [".py", ".pyw"],
        "label": "Python",
        "icon": "🐍",
        "parseable": True,
    },
    "javascript": {
        "extensions": [".js", ".jsx", ".mjs"],
        "label": "JavaScript",
        "icon": "📜",
        "parseable": True,
    },
    "typescript": {
        "extensions": [".ts", ".tsx"],
        "label": "TypeScript",
        "icon": "📘",
        "parseable": True,
    },
    "html": {
        "extensions": [".html", ".htm", ".xhtml"],
        "label": "HTML",
        "icon": "🌐",
        "parseable": False,
    },
    "css": {
        "extensions": [".css", ".scss", ".sass", ".less"],
        "label": "CSS",
        "icon": "🎨",
        "parseable": False,
    },
    "json": {
        "extensions": [".json"],
        "label": "JSON",
        "icon": "📋",
        "parseable": False,
    },
    "yaml": {
        "extensions": [".yaml", ".yml"],
        "label": "YAML",
        "icon": "⚙️",
        "parseable": False,
    },
    "markdown": {
        "extensions": [".md", ".markdown", ".rst"],
        "label": "Markdown",
        "icon": "📝",
        "parseable": False,
    },
    "sql": {
        "extensions": [".sql"],
        "label": "SQL",
        "icon": "🗃️",
        "parseable": False,
    },
    "shell": {
        "extensions": [".sh", ".bash", ".zsh"],
        "label": "Shell",
        "icon": "💻",
        "parseable": False,
    },
}

EXT_TO_TYPE: Dict[str, str] = {}
for ftype, config in FILE_TYPES.items():
    for ext in config["extensions"]:
        EXT_TO_TYPE[ext] = ftype

SKIP_DIRS = {
    "__pycache__", ".git", ".svn", ".hg", "node_modules", ".venv", "venv",
    "env", ".env", ".idea", ".vscode", "dist", "build", ".tox", ".pytest_cache",
    ".mypy_cache", ".eggs", "*.egg-info", ".cache", "coverage", ".coverage",
}


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class FuncInfo:
    name: str
    line: int
    args: List[str] = field(default_factory=list)
    docstring: Optional[str] = None
    is_async: bool = False


@dataclass
class ClassInfo:
    name: str
    line: int
    methods: List[str] = field(default_factory=list)
    docstring: Optional[str] = None
    extends: Optional[str] = None


@dataclass
class FileInfo:
    name: str
    rel_path: str
    full_path: str
    extension: str
    file_type: str
    source: str = ""
    line_count: int = 0
    docstring: Optional[str] = None

    module_vars: List[str] = field(default_factory=list)
    functions: List[FuncInfo] = field(default_factory=list)
    classes: List[ClassInfo] = field(default_factory=list)

    imports_internal: List[str] = field(default_factory=list)
    imports_external: List[str] = field(default_factory=list)
    imported_by: List[str] = field(default_factory=list)

    parse_error: Optional[str] = None

    @property
    def display_name(self) -> str:
        return self.rel_path


# ============================================================================
# STDLIB MODULES (used to categorize imports)
# ============================================================================

STDLIB = {
    "abc", "argparse", "ast", "asyncio", "base64", "collections", "concurrent",
    "contextlib", "copy", "csv", "dataclasses", "datetime", "decimal", "enum",
    "functools", "gc", "glob", "gzip", "hashlib", "heapq", "hmac", "html",
    "http", "importlib", "inspect", "io", "itertools", "json", "logging",
    "math", "multiprocessing", "operator", "os", "pathlib", "pickle",
    "platform", "pprint", "queue", "random", "re", "secrets", "shutil",
    "signal", "socket", "sqlite3", "ssl", "statistics", "string", "struct",
    "subprocess", "sys", "tempfile", "textwrap", "threading", "time",
    "traceback", "types", "typing", "unittest", "urllib", "uuid", "warnings",
    "weakref", "xml", "zipfile", "zlib", "__future__", "builtins", "codecs",
    "configparser", "ctypes", "email", "ftplib", "getpass", "imaplib",
    "locale", "mimetypes", "optparse", "smtplib", "tarfile",
}


# ============================================================================
# PARSERS
# ============================================================================

def parse_python_file(filepath: str, source: str, internal_names: Set[str]) -> FileInfo:
    path = Path(filepath)
    info = FileInfo(
        name=path.stem,
        rel_path=str(path.name),
        full_path=str(path.resolve()),
        extension=path.suffix,
        file_type="python",
        source=source,
        line_count=len(source.splitlines()),
    )

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        info.parse_error = f"SyntaxError line {e.lineno}: {e.msg}"
        return info
    except Exception as e:
        info.parse_error = str(e)
        return info

    info.docstring = ast.get_docstring(tree)

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                base = alias.name.split(".")[0]
                if base in internal_names:
                    if base not in info.imports_internal:
                        info.imports_internal.append(base)
                elif base not in STDLIB:
                    if base not in info.imports_external:
                        info.imports_external.append(base)

        elif isinstance(node, ast.ImportFrom) and node.module:
            base = node.module.split(".")[0]
            if base in internal_names:
                if base not in info.imports_internal:
                    info.imports_internal.append(base)
            elif base not in STDLIB:
                if base not in info.imports_external:
                    info.imports_external.append(base)

        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and not target.id.startswith("_"):
                    info.module_vars.append(target.id)

        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and not node.target.id.startswith("_"):
                info.module_vars.append(node.target.id)

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = [a.arg for a in node.args.args]
            doc = ast.get_docstring(node)
            if doc and len(doc) > 100:
                doc = doc[:97] + "..."
            info.functions.append(FuncInfo(
                name=node.name,
                line=node.lineno,
                args=args,
                docstring=doc,
                is_async=isinstance(node, ast.AsyncFunctionDef),
            ))

        elif isinstance(node, ast.ClassDef):
            methods = []
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    methods.append(item.name)
            doc = ast.get_docstring(node)
            if doc and len(doc) > 100:
                doc = doc[:97] + "..."

            extends = None
            if node.bases:
                first_base = node.bases[0]
                if isinstance(first_base, ast.Name):
                    extends = first_base.id
                elif isinstance(first_base, ast.Attribute):
                    extends = first_base.attr

            info.classes.append(ClassInfo(
                name=node.name,
                line=node.lineno,
                methods=methods,
                docstring=doc,
                extends=extends,
            ))

    return info


def parse_javascript_file(filepath: str, source: str, internal_names: Set[str]) -> FileInfo:
    path = Path(filepath)
    ext = path.suffix
    file_type = "typescript" if ext in [".ts", ".tsx"] else "javascript"

    info = FileInfo(
        name=path.stem,
        rel_path=str(path.name),
        full_path=str(path.resolve()),
        extension=ext,
        file_type=file_type,
        source=source,
        line_count=len(source.splitlines()),
    )

    lines = source.splitlines()

    def _starts_with_any(s: str, prefixes: Tuple[str, ...]) -> bool:
        for p in prefixes:
            if s.startswith(p):
                return True
        return False

    for i, line in enumerate(lines, 1):
        ln = line.lstrip()
        if "function " in ln and _starts_with_any(ln, ("function ", "export function ", "export async function ", "async function ")):
            idx = ln.find("function ")
            rest = ln[idx + len("function "):].strip()
            name = ""
            for ch in rest:
                if ch.isalnum() or ch == "_":
                    name += ch
                else:
                    break
            if name:
                info.functions.append(FuncInfo(name=name, line=i, args=[], is_async=("async" in ln[:idx])))

        if _starts_with_any(ln, ("class ", "export class ", "export abstract class ", "abstract class ")):
            parts = ln.replace("{", " ").split()
            if "class" in parts:
                ci = parts.index("class")
                if ci + 1 < len(parts):
                    cname = parts[ci + 1]
                    extends = None
                    if "extends" in parts:
                        ei = parts.index("extends")
                        if ei + 1 < len(parts):
                            extends = parts[ei + 1]
                    info.classes.append(ClassInfo(name=cname, line=i, extends=extends))

        if ln.startswith("import ") and " from " in ln:
            seg = ln.split(" from ", 1)[1].strip()
            q1 = seg.find("'")
            q2 = seg.find('"')
            q = q1 if (q1 != -1 and (q2 == -1 or q1 < q2)) else q2
            if q != -1:
                qend = seg.find(seg[q], q + 1)
                if qend != -1:
                    import_path = seg[q + 1:qend]
                    if import_path.startswith((".", "/")):
                        import_name = Path(import_path).stem
                        if import_name in internal_names and import_name not in info.imports_internal:
                            info.imports_internal.append(import_name)
                    else:
                        pkg = import_path.split("/")[0]
                        if pkg not in info.imports_external:
                            info.imports_external.append(pkg)

    return info


def parse_generic_file(filepath: str, source: str, file_type: str) -> FileInfo:
    path = Path(filepath)
    info = FileInfo(
        name=path.stem,
        rel_path=str(path.name),
        full_path=str(path.resolve()),
        extension=path.suffix,
        file_type=file_type,
        source=source,
        line_count=len(source.splitlines()),
    )

    if file_type == "json":
        try:
            data = json.loads(source)
            if isinstance(data, dict):
                info.module_vars = list(data.keys())[:20]
        except Exception:
            info.parse_error = "Invalid JSON"

    elif file_type == "html":
        lower = source.lower()
        script_count = lower.count("<script")
        style_count = lower.count("<style") + lower.count("stylesheet")
        t0 = lower.find("<title>")
        t1 = lower.find("</title>") if t0 != -1 else -1
        if t0 != -1 and t1 != -1 and t1 > t0:
            title = source[t0 + 7:t1].strip()
            info.docstring = f"Title: {title}"
        info.module_vars = [f"scripts:{script_count}", f"styles:{style_count}"]

    elif file_type == "css":
        rule_count = source.count("{")
        info.module_vars = [f"rules:{rule_count}"]

    return info


def parse_file(filepath: str, internal_names: Set[str]) -> Optional[FileInfo]:
    path = Path(filepath)
    ext = path.suffix.lower()
    if ext not in EXT_TO_TYPE:
        return None

    file_type = EXT_TO_TYPE[ext]

    try:
        source = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            source = path.read_text(encoding="latin-1")
        except Exception:
            return None
    except Exception:
        return None

    if file_type == "python":
        return parse_python_file(filepath, source, internal_names)
    if file_type in ("javascript", "typescript"):
        return parse_javascript_file(filepath, source, internal_names)
    return parse_generic_file(filepath, source, file_type)


# ============================================================================
# DIRECTORY SCANNING
# ============================================================================

def should_skip_dir(dirname: str) -> bool:
    if dirname.startswith("."):
        return True
    if dirname in SKIP_DIRS:
        return True
    if dirname.endswith(".egg-info"):
        return True
    return False


def collect_files(
    directory: str,
    extensions: Optional[Set[str]] = None,
    recursive: bool = True,
    max_depth: Optional[int] = None,
) -> List[Tuple[Path, str]]:
    root = Path(directory).resolve()
    files: List[Tuple[Path, str]] = []

    if extensions is None:
        extensions = set(EXT_TO_TYPE.keys())
    else:
        extensions = {ext if ext.startswith(".") else f".{ext}" for ext in extensions}

    def scan_dir(current: Path, depth: int):
        if max_depth is not None and depth > max_depth:
            return
        try:
            entries = list(current.iterdir())
        except PermissionError:
            return

        for entry in sorted(entries):
            if entry.is_file():
                if entry.suffix.lower() in extensions:
                    rel_path = entry.relative_to(root)
                    files.append((entry, str(rel_path)))
            elif entry.is_dir() and recursive:
                if not should_skip_dir(entry.name):
                    scan_dir(entry, depth + 1)

    scan_dir(root, 0)
    return files


def parse_directory(
    directory: str,
    extensions: Optional[Set[str]] = None,
    recursive: bool = True,
    max_depth: Optional[int] = None,
) -> Dict[str, FileInfo]:
    files_to_parse = collect_files(directory, extensions, recursive, max_depth)

    internal_names: Set[str] = set()
    for filepath, rel_path in files_to_parse:
        internal_names.add(filepath.stem)
        if filepath.parent.name != Path(directory).name:
            internal_names.add(filepath.parent.name)

    files: Dict[str, FileInfo] = {}
    for filepath, rel_path in files_to_parse:
        info = parse_file(str(filepath), internal_names)
        if info:
            info.rel_path = rel_path
            files[rel_path] = info

    for rel_path, info in files.items():
        for imp in info.imports_internal:
            for other_path, other_info in files.items():
                if other_info.name == imp and other_path != rel_path:
                    if rel_path not in other_info.imported_by:
                        other_info.imported_by.append(rel_path)

    return files


# ============================================================================
# OUTPUT GENERATORS
# ============================================================================

def generate_map(files: Dict[str, FileInfo]) -> str:
    lines: List[str] = []
    lines.append("=" * 70)
    lines.append("CODE MAP")
    lines.append("=" * 70)
    lines.append("")

    type_counts: Dict[str, int] = {}
    type_lines: Dict[str, int] = {}
    for f in files.values():
        type_counts[f.file_type] = type_counts.get(f.file_type, 0) + 1
        type_lines[f.file_type] = type_lines.get(f.file_type, 0) + f.line_count

    total_lines = sum(f.line_count for f in files.values())
    total_funcs = sum(len(f.functions) for f in files.values())
    total_classes = sum(len(f.classes) for f in files.values())

    lines.append(f"Total Files: {len(files)}  |  Total Lines: {total_lines:,}")
    lines.append(f"Functions: {total_funcs}  |  Classes: {total_classes}")
    lines.append("")

    lines.append("File Types:")
    for ftype in sorted(type_counts.keys()):
        icon = FILE_TYPES.get(ftype, {}).get("icon", "📄")
        lines.append(f"  {icon} {ftype}: {type_counts[ftype]} files, {type_lines[ftype]:,} lines")
    lines.append("")

    all_external = sorted(set(d for f in files.values() for d in f.imports_external))
    if all_external:
        lines.append(f"External Dependencies: {', '.join(all_external[:20])}")
        if len(all_external) > 20:
            lines.append(f"  ... and {len(all_external) - 20} more")
        lines.append("")

    dirs: Dict[str, List[FileInfo]] = {}
    for rel_path, info in files.items():
        dir_path = str(Path(rel_path).parent)
        if dir_path == ".":
            dir_path = "(root)"
        dirs.setdefault(dir_path, []).append(info)

    lines.append("-" * 70)
    lines.append("FILES AND STRUCTURE")
    lines.append("-" * 70)

    for dir_path in sorted(dirs.keys()):
        lines.append("")
        lines.append(f"📁 {dir_path}/")
        for f in sorted(dirs[dir_path], key=lambda x: x.name):
            icon = FILE_TYPES.get(f.file_type, {}).get("icon", "📄")
            error_mark = " ⚠️" if f.parse_error else ""
            lines.append(f"  {icon} {f.name}{f.extension} ({f.line_count} lines){error_mark}")

            if f.module_vars:
                vars_display = ", ".join(f.module_vars[:8])
                if len(f.module_vars) > 8:
                    vars_display += f" +{len(f.module_vars)-8} more"
                lines.append(f"      Variables: {vars_display}")

            if f.functions:
                func_names = [fn.name for fn in f.functions[:6]]
                funcs_display = ", ".join(func_names)
                if len(f.functions) > 6:
                    funcs_display += f" +{len(f.functions)-6} more"
                lines.append(f"      Functions: {funcs_display}")

            if f.classes:
                for cls in f.classes[:4]:
                    extends_str = f" extends {cls.extends}" if cls.extends else ""
                    lines.append(f"      Class {cls.name}{extends_str}: {len(cls.methods)} methods")
                if len(f.classes) > 4:
                    lines.append(f"      ... +{len(f.classes)-4} more classes")

            if f.imports_internal:
                lines.append(f"      Imports: {', '.join(sorted(f.imports_internal))}")

            if f.imported_by:
                lines.append(f"      Used by: {', '.join(sorted(f.imported_by)[:5])}")

    parseable = {k: v for k, v in files.items() if v.imports_internal or v.imported_by}
    if parseable:
        lines.append("")
        lines.append("-" * 70)
        lines.append("DEPENDENCY GRAPH")
        lines.append("-" * 70)

        for rel_path in sorted(parseable.keys()):
            f = parseable[rel_path]
            if f.imports_internal:
                lines.append(f"{f.name} → {', '.join(sorted(f.imports_internal))}")
            else:
                lines.append(f"{f.name} → (no internal deps)")

    return "\n".join(lines)


def generate_concat(files: Dict[str, FileInfo], selected: List[str]) -> str:
    output_lines: List[str] = []
    for rel_path in sorted(selected):
        if rel_path not in files:
            continue
        f = files[rel_path]
        icon = FILE_TYPES.get(f.file_type, {}).get("icon", "📄")
        output_lines.append("#" + "=" * 69)
        output_lines.append(f"# {icon} FILE: {f.rel_path} ({f.line_count} lines)")
        output_lines.append("#" + "=" * 69)
        output_lines.append("")
        output_lines.append(f.source)
        output_lines.append("")
        output_lines.append("")
    return "\n".join(output_lines)


def generate_map_and_code(files: Dict[str, FileInfo], selected: List[str]) -> str:
    filtered = {k: v for k, v in files.items() if k in selected}
    parts: List[str] = []
    parts.append(generate_map(filtered))
    parts.append("")
    parts.append("")
    parts.append("#" * 70)
    parts.append("# SOURCE CODE")
    parts.append("#" * 70)
    parts.append("")
    parts.append(generate_concat(files, selected))
    return "\n".join(parts)


def generate_json_structure(files: Dict[str, FileInfo], selected: List[str]) -> Dict[str, Any]:
    filtered = {k: v for k, v in files.items() if k in selected}
    by_type: Dict[str, int] = {}
    for f in filtered.values():
        by_type[f.file_type] = by_type.get(f.file_type, 0) + 1

    file_list = []
    for rel_path in sorted(filtered.keys()):
        f = filtered[rel_path]
        file_list.append({
            "path": rel_path,
            "name": f.name,
            "type": f.file_type,
            "extension": f.extension,
            "lines": f.line_count,
            "variables": f.module_vars,
            "functions": [
                {"name": fn.name, "args": fn.args, "line": fn.line, "async": fn.is_async}
                for fn in f.functions
            ],
            "classes": [
                {"name": c.name, "methods": c.methods, "line": c.line, "extends": c.extends}
                for c in f.classes
            ],
            "imports_internal": f.imports_internal,
            "imports_external": f.imports_external,
            "imported_by": f.imported_by,
        })

    return {
        "summary": {
            "total_files": len(filtered),
            "total_lines": sum(f.line_count for f in filtered.values()),
            "by_type": by_type,
            "external_deps": sorted(set(d for f in filtered.values() for d in f.imports_external)),
        },
        "files": file_list,
    }


# ============================================================================
# CUSTOM QUERY PROCESSOR
# ============================================================================

QUERY_HELP = """
CUSTOM QUERY EXAMPLES
=====================

Type a query and click "Run Query". The system will search the codebase
and return structured results.

Example Queries:
----------------
- "functions in worker"           → List all functions in files matching "worker"
- "classes"                       → List all classes in selected files
- "variables in config"           → List module-level variables
- "imports"                       → Show what each file imports
- "who uses db"                   → Which files import db
- "find portfolio"                → Search for files/items containing "portfolio"
- "grep execute"                  → Find functions/classes containing "execute"
- "type python"                   → Show only Python files
- "type javascript"               → Show only JavaScript files
- "type typescript"               → Show only TypeScript files

The output is JSON that you can paste directly to an LLM.
"""


def process_query(files: Dict[str, FileInfo], selected: List[str], query: str) -> str:
    query = query.lower().strip()
    filtered = {k: v for k, v in files.items() if k in selected}
    result: Dict[str, Any] = {"query": query, "results": []}

    if query.startswith("type "):
        target_type = query.split(" ", 1)[1].strip()
        matching = [f for f in filtered.values() if target_type in f.file_type]
        result["results"] = [
            {"path": f.rel_path, "type": f.file_type, "lines": f.line_count}
            for f in sorted(matching, key=lambda x: x.rel_path)
        ]

    elif query.startswith("functions"):
        parts = query.split(" in ")
        target = parts[1].strip() if len(parts) > 1 else None
        for rel_path, f in sorted(filtered.items()):
            if target and target not in f.rel_path.lower() and target not in f.name.lower():
                continue
            if f.functions:
                result["results"].append({
                    "file": rel_path,
                    "functions": [
                        {"name": fn.name, "args": fn.args, "line": fn.line, "doc": fn.docstring}
                        for fn in f.functions
                    ]
                })

    elif query.startswith("classes"):
        parts = query.split(" in ")
        target = parts[1].strip() if len(parts) > 1 else None
        for rel_path, f in sorted(filtered.items()):
            if target and target not in f.rel_path.lower() and target not in f.name.lower():
                continue
            if f.classes:
                result["results"].append({
                    "file": rel_path,
                    "classes": [
                        {"name": c.name, "methods": c.methods, "line": c.line, "extends": c.extends}
                        for c in f.classes
                    ]
                })

    elif query.startswith("variables") or query.startswith("vars"):
        parts = query.split(" in ")
        target = parts[1].strip() if len(parts) > 1 else None
        for rel_path, f in sorted(filtered.items()):
            if target and target not in f.rel_path.lower() and target not in f.name.lower():
                continue
            if f.module_vars:
                result["results"].append({"file": rel_path, "variables": f.module_vars})

    elif query.startswith("imports"):
        for rel_path, f in sorted(filtered.items()):
            if f.imports_internal or f.imports_external:
                result["results"].append({
                    "file": rel_path,
                    "internal": f.imports_internal,
                    "external": f.imports_external
                })

    elif query.startswith("who uses") or query.startswith("imported by"):
        target = query.replace("who uses", "").replace("imported by", "").strip()
        for rel_path, f in sorted(filtered.items()):
            if target in f.name.lower() and f.imported_by:
                result["results"].append({"file": rel_path, "imported_by": f.imported_by})

    elif query.startswith("find ") or query.startswith("grep ") or query.startswith("search "):
        pattern = query.split(" ", 1)[1].strip()
        for rel_path, f in sorted(filtered.items()):
            matches = []
            if pattern in f.name.lower():
                matches.append(f"Filename contains '{pattern}'")
            if pattern in f.rel_path.lower():
                matches.append(f"Path contains '{pattern}'")
            for fn in f.functions:
                if pattern in fn.name.lower():
                    matches.append(f"Function: {fn.name} (line {fn.line})")
            for c in f.classes:
                if pattern in c.name.lower():
                    matches.append(f"Class: {c.name} (line {c.line})")
                for method in c.methods:
                    if pattern in method.lower():
                        matches.append(f"Method: {c.name}.{method}")
            for v in f.module_vars:
                if pattern in v.lower():
                    matches.append(f"Variable: {v}")
            if matches:
                result["results"].append({"file": rel_path, "matches": matches})

    elif query in ("structure", "overview", "all"):
        result["results"] = generate_json_structure(filtered, list(filtered.keys()))

    elif query in ("folders", "dirs", "directories"):
        dirs: Dict[str, List[str]] = {}
        for rel_path in filtered.keys():
            dir_path = str(Path(rel_path).parent)
            if dir_path == ".":
                dir_path = "(root)"
            dirs.setdefault(dir_path, []).append(Path(rel_path).name)
        result["results"] = [{"folder": d, "files": sorted(files_list)} for d, files_list in sorted(dirs.items())]

    elif query == "help":
        return QUERY_HELP

    else:
        result["error"] = f"Unknown query: '{query}'"
        result["hint"] = "Try: functions, classes, variables, imports, 'who uses X', 'find X', 'type X', or 'help'"

    return json.dumps(result, indent=2)


# ============================================================================
# FUZZY GREP LOGIC (SMART LOCATOR)
# ============================================================================

def normalize_code(text: str) -> str:
    """Normalize text to make fuzzy matching robust to whitespace."""
    return re.sub(r'\s+', ' ', text).strip()


def find_best_match(file_content: str, search_block: str) -> Tuple[bool, int, int, float]:
    """
    Locates the region in file_content that is most similar to search_block.
    Returns (found, start_index, end_index, score).
    """
    if not search_block.strip() or not file_content.strip():
        return False, 0, 0, 0.0

    # 1. Exact match check
    if search_block in file_content:
        start = file_content.find(search_block)
        return True, start, start + len(search_block), 1.0

    # 2. Fuzzy Line Matching
    # We break both into lines and assume the replacement is roughly a contiguous block.
    f_lines = file_content.splitlines(keepends=True)
    s_lines = search_block.splitlines(keepends=True)

    if not s_lines:
        return False, 0, 0, 0.0

    # We use a sliding window of roughly the same number of lines
    n_search = len(s_lines)

    # Heuristic: Expand window slightly to account for deleted/added lines
    best_ratio = 0.0
    best_start_line = 0
    best_end_line = 0

    # Normalize for comparison
    norm_search = normalize_code(search_block)

    # Optimization: Check every line as a start point
    # Compare simple string similarity first

    for i in range(len(f_lines)):
        # Check a window of exactly N lines first (fastest)
        end = min(i + n_search, len(f_lines))
        window_text = "".join(f_lines[i:end])

        # Quick ratio using difflib
        ratio = difflib.SequenceMatcher(None, normalize_code(window_text), norm_search).ratio()

        if ratio > best_ratio:
            best_ratio = ratio
            best_start_line = i
            best_end_line = end

    # If we found a decent match, try to refine the boundaries slightly
    if best_ratio > 0.4:  # Threshold
        # Calculate character indices
        char_start = 0
        for k in range(best_start_line):
            char_start += len(f_lines[k])

        char_end = char_start
        for k in range(best_start_line, best_end_line):
            char_end += len(f_lines[k])

        return True, char_start, char_end, best_ratio

    return False, 0, 0, best_ratio


def indent_label_text(text: str, tabsize: int = 4) -> str:
    """
    Convert source text into lines prefixed with [[((N))]] where N is leading spaces.
    Tabs are expanded (tabsize) for counting and normalization.
    """
    out_lines: List[str] = []
    for raw_line in text.splitlines():
        expanded = raw_line.expandtabs(tabsize)
        n = 0
        while n < len(expanded) and expanded[n] == " ":
            n += 1
        code = expanded[n:]
        out_lines.append(f"[[(({n}))]]{code}")
    return "\n".join(out_lines) + ("\n" if text.endswith("\n") else "")


# ============================================================================
# GUI APPLICATION
# ============================================================================

class CodeMapperApp:
    def __init__(self, root: tk.Tk, initial_path: Optional[str] = None):
        self.root = root
        self.root.title("KG Code Mapper v4.1 (Smart Locator)")
        self.root.geometry("1500x980")

        self.files: Dict[str, FileInfo] = {}
        self.file_vars: Dict[str, tk.BooleanVar] = {}
        self.type_vars: Dict[str, tk.BooleanVar] = {}
        self.current_dir: Optional[str] = None

        self.recursive_var = tk.BooleanVar(value=True)
        self.depth_var = tk.StringVar(value="")

        # Smart Locator State
        self.locator_current_rel_path: Optional[str] = None

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        self._build_ui()

        if initial_path:
            self._load_directory(initial_path)

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=5)
        main.grid(row=0, column=0, sticky="nsew")
        main.columnconfigure(1, weight=1)
        main.rowconfigure(1, weight=1)

        # ===== TOP TOOLBAR =====
        toolbar = ttk.Frame(main)
        toolbar.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 5))

        ttk.Button(toolbar, text="📂 Load Directory", command=self._on_load).pack(side="left", padx=2)
        ttk.Button(toolbar, text="🔄 Refresh", command=self._on_refresh).pack(side="left", padx=2)

        ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=10)

        ttk.Checkbutton(toolbar, text="📁 Recursive", variable=self.recursive_var).pack(side="left", padx=2)

        ttk.Label(toolbar, text="Max Depth:").pack(side="left", padx=(10, 2))
        ttk.Entry(toolbar, textvariable=self.depth_var, width=4).pack(side="left", padx=2)

        ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=10)

        ttk.Label(toolbar, text="Directory:").pack(side="left", padx=2)
        self.dir_label = ttk.Label(toolbar, text="(none loaded)", foreground="gray")
        self.dir_label.pack(side="left", padx=2)

        # ===== LEFT PANEL =====
        left = ttk.Frame(main)
        left.grid(row=1, column=0, sticky="ns", padx=(0, 5))

        type_frame = ttk.LabelFrame(left, text="File Types", padding=5)
        type_frame.pack(fill="x", pady=(0, 5))

        for ftype, config in FILE_TYPES.items():
            var = tk.BooleanVar(value=True)
            self.type_vars[ftype] = var
            ttk.Checkbutton(
                type_frame,
                text=f"{config['icon']} {config['label']}",
                variable=var,
                command=self._refresh_file_list
            ).pack(anchor="w")

        type_btns = ttk.Frame(type_frame)
        type_btns.pack(fill="x", pady=(5, 0))
        ttk.Button(type_btns, text="All", command=self._select_all_types, width=6).pack(side="left", expand=True)
        ttk.Button(type_btns, text="None", command=self._select_no_types, width=6).pack(side="left", expand=True)
        ttk.Button(type_btns, text="Code", command=self._select_code_types, width=6).pack(side="left", expand=True)

        file_frame = ttk.LabelFrame(left, text="Files (Check to Map/Copy)", padding=5)
        file_frame.pack(fill="both", expand=True)

        self.file_canvas = tk.Canvas(file_frame, width=280, highlightthickness=0)
        scroll = ttk.Scrollbar(file_frame, orient="vertical", command=self.file_canvas.yview)
        self.file_inner = ttk.Frame(self.file_canvas)

        self.file_canvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.file_canvas.pack(side="left", fill="both", expand=True)

        self.canvas_window = self.file_canvas.create_window((0, 0), window=self.file_inner, anchor="nw")
        self.file_inner.bind("<Configure>", lambda e: self.file_canvas.configure(scrollregion=self.file_canvas.bbox("all")))

        btn_frame = ttk.Frame(file_frame)
        btn_frame.pack(fill="x", pady=(5, 0))
        ttk.Button(btn_frame, text="Select All", command=self._select_all_files).pack(side="left", expand=True, fill="x")
        ttk.Button(btn_frame, text="Select None", command=self._select_no_files).pack(side="left", expand=True, fill="x")

        # ===== RIGHT PANEL =====
        right = ttk.Frame(main)
        right.grid(row=1, column=1, sticky="nsew")
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)

        self.nb = ttk.Notebook(right)
        self.nb.grid(row=0, column=0, sticky="nsew")

        # --------------------
        # Tab 1: Output / Context Generation
        # --------------------
        tab_out = ttk.Frame(self.nb)
        tab_out.rowconfigure(0, weight=1)
        tab_out.columnconfigure(0, weight=1)
        self.nb.add(tab_out, text="1. Output (Get Context)")

        output_frame = ttk.LabelFrame(tab_out, text="Context Output (Copy this to LLM)", padding=5)
        output_frame.grid(row=0, column=0, sticky="nsew")
        output_frame.rowconfigure(0, weight=1)
        output_frame.columnconfigure(0, weight=1)

        self.output_text = scrolledtext.ScrolledText(output_frame, wrap="none", font=("Courier", 11))
        self.output_text.grid(row=0, column=0, sticky="nsew")

        h_scroll = ttk.Scrollbar(output_frame, orient="horizontal", command=self.output_text.xview)
        h_scroll.grid(row=1, column=0, sticky="ew")
        self.output_text.configure(xscrollcommand=h_scroll.set)

        actions = ttk.LabelFrame(tab_out, text="Actions", padding=5)
        actions.grid(row=1, column=0, sticky="ew", pady=(5, 0))

        ttk.Button(actions, text="📋 Concat All", command=self._concat_all).pack(side="left", padx=2)
        ttk.Button(actions, text="📋 Concat Selected", command=self._concat_selected).pack(side="left", padx=2)
        ttk.Button(actions, text="🗺 Map Only", command=self._map_only).pack(side="left", padx=2)
        ttk.Button(actions, text="🗺+📋 Map + Code", command=self._map_and_code).pack(side="left", padx=2)
        ttk.Button(actions, text="📊 JSON Structure", command=self._json_structure).pack(side="left", padx=2)

        ttk.Separator(actions, orient="vertical").pack(side="left", fill="y", padx=10)

        ttk.Button(actions, text="📋 Copy Output", command=self._copy_output).pack(side="left", padx=2)
        ttk.Button(actions, text="💾 Save Output", command=self._save_output).pack(side="left", padx=2)

        query_frame = ttk.LabelFrame(tab_out, text="Custom Query (type 'help' for examples)", padding=5)
        query_frame.grid(row=2, column=0, sticky="ew", pady=(5, 0))
        query_frame.columnconfigure(0, weight=1)

        self.query_entry = ttk.Entry(query_frame, font=("Courier", 12))
        self.query_entry.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        self.query_entry.bind("<Return>", lambda e: self._run_query())
        ttk.Button(query_frame, text="Run", command=self._run_query).grid(row=0, column=1, padx=2)
        ttk.Button(query_frame, text="❓ Help", command=self._query_help).grid(row=0, column=2, padx=2)

        # --------------------
        # Tab 2: Smart Locator (The "Sophisticated Find")
        # --------------------
        tab_loc = ttk.Frame(self.nb)
        self.nb.add(tab_loc, text="2. Smart Locator (Paste Fix)")
        tab_loc.rowconfigure(1, weight=1)
        tab_loc.columnconfigure(0, weight=1)

        # Top Bar: File Selector
        loc_top = ttk.Frame(tab_loc, padding=5)
        loc_top.grid(row=0, column=0, sticky="ew")

        ttk.Label(loc_top, text="Editing File:").pack(side="left")
        self.locator_file_combo = ttk.Combobox(loc_top, state="readonly", width=50)
        self.locator_file_combo.pack(side="left", padx=5)
        self.locator_file_combo.bind("<<ComboboxSelected>>", self._locator_on_file_select)

        ttk.Button(loc_top, text="💾 Save Changes", command=self._locator_save).pack(side="left", padx=5)
        ttk.Button(loc_top, text="🔄 Reload File", command=self._locator_reload).pack(side="left", padx=5)

        # Main Split: Editor | Fix Input
        loc_pane = ttk.PanedWindow(tab_loc, orient="horizontal")
        loc_pane.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        # Left: File Editor
        loc_left = ttk.LabelFrame(loc_pane, text="File Content (Auto-Highlights Here)", padding=5)
        loc_pane.add(loc_left, weight=2)
        
        self.locator_editor = scrolledtext.ScrolledText(loc_left, wrap="none", font=("Consolas", 11), undo=True)
        self.locator_editor.pack(fill="both", expand=True)
        self.locator_editor.tag_config("highlight", background="#ffff99", foreground="black")

        # Right: Fix Input
        loc_right = ttk.LabelFrame(loc_pane, text="Paste Corrected Code Here", padding=5)
        loc_pane.add(loc_right, weight=1)

        ttk.Label(loc_right, text="1. Get fix from LLM.").pack(anchor="w")
        ttk.Label(loc_right, text="2. Paste it below.").pack(anchor="w")
        
        self.locator_search_box = scrolledtext.ScrolledText(loc_right, height=15, font=("Consolas", 10))
        self.locator_search_box.pack(fill="both", expand=True, pady=5)

        ttk.Button(loc_right, text="🔍 Locate & Highlight Area", command=self._locator_do_find).pack(fill="x", pady=5)
        
        ttk.Label(loc_right, text="3. If highlighted area is correct,\n   PASTE over it in the left editor.", foreground="gray").pack(pady=5)
        
        self.locator_status_var = tk.StringVar(value="Ready.")
        ttk.Label(loc_right, textvariable=self.locator_status_var, relief="sunken").pack(fill="x", pady=5)

        # ===== STATUS BAR =====
        self.status_var = tk.StringVar(value="Ready. Load a directory to begin.")
        status = ttk.Label(main, textvariable=self.status_var, relief="sunken", anchor="w")
        status.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(5, 0))

    # -------------------- directory/load --------------------

    def _on_load(self):
        path = filedialog.askdirectory(title="Select Project Directory")
        if path:
            self._load_directory(path)

    def _load_directory(self, path: str):
        self.current_dir = path
        self.dir_label.configure(text=path, foreground="black")
        self.status_var.set(f"Parsing {path}...")
        self.root.update()

        try:
            extensions = set()
            for ftype, var in self.type_vars.items():
                if var.get():
                    extensions.update(FILE_TYPES[ftype]["extensions"])

            max_depth = None
            if self.depth_var.get().strip():
                try:
                    max_depth = int(self.depth_var.get().strip())
                except ValueError:
                    pass

            self.files = parse_directory(
                path,
                extensions=extensions if extensions else None,
                recursive=self.recursive_var.get(),
                max_depth=max_depth,
            )

            self._refresh_file_list()

            type_counts: Dict[str, int] = {}
            for f in self.files.values():
                type_counts[f.file_type] = type_counts.get(f.file_type, 0) + 1
            summary = ", ".join(f"{v} {k}" for k, v in sorted(type_counts.items()))
            self.status_var.set(f"Loaded {len(self.files)} files: {summary}")

            # Prepare Locator Combo
            file_paths = sorted(self.files.keys())
            self.locator_file_combo['values'] = file_paths
            if file_paths:
                self.locator_file_combo.current(0)
                self._locator_load_file(file_paths[0])

            self._map_only()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to parse:\n{e}")
            self.status_var.set(f"Error: {e}")

    def _on_refresh(self):
        if self.current_dir:
            self._load_directory(self.current_dir)

    def _refresh_file_list(self):
        for widget in self.file_inner.winfo_children():
            widget.destroy()
        self.file_vars.clear()

        enabled_types = {ftype for ftype, var in self.type_vars.items() if var.get()}

        dirs: Dict[str, List[str]] = {}
        for rel_path, info in self.files.items():
            if info.file_type not in enabled_types:
                continue
            dir_path = str(Path(rel_path).parent)
            if dir_path == ".":
                dir_path = "(root)"
            dirs.setdefault(dir_path, []).append(rel_path)

        for dir_path in sorted(dirs.keys()):
            ttk.Label(self.file_inner, text=f"📁 {dir_path}/", font=("TkDefaultFont", 9, "bold")).pack(anchor="w", pady=(5, 0))

            for rel_path in sorted(dirs[dir_path]):
                info = self.files[rel_path]
                var = tk.BooleanVar(value=True)
                self.file_vars[rel_path] = var

                icon = FILE_TYPES.get(info.file_type, {}).get("icon", "📄")
                label = f"{icon} {info.name}{info.extension} ({info.line_count})"
                if info.parse_error:
                    label += " ⚠️"

                cb = ttk.Checkbutton(self.file_inner, text=label, variable=var)
                cb.pack(anchor="w", padx=(15, 0))

    def _get_selected(self) -> List[str]:
        enabled_types = {ftype for ftype, var in self.type_vars.items() if var.get()}
        return [
            path for path, var in self.file_vars.items()
            if var.get() and self.files.get(path, FileInfo("", "", "", "", "")).file_type in enabled_types
        ]

    def _select_all_types(self):
        for var in self.type_vars.values():
            var.set(True)
        self._refresh_file_list()

    def _select_no_types(self):
        for var in self.type_vars.values():
            var.set(False)
        self._refresh_file_list()

    def _select_code_types(self):
        code_types = {"python", "javascript", "typescript"}
        for ftype, var in self.type_vars.items():
            var.set(ftype in code_types)
        self._refresh_file_list()

    def _select_all_files(self):
        for var in self.file_vars.values():
            var.set(True)

    def _select_no_files(self):
        for var in self.file_vars.values():
            var.set(False)

    # -------------------- output actions --------------------

    def _set_output(self, text: str):
        self.output_text.delete("1.0", "end")
        self.output_text.insert("1.0", text)

    def _concat_all(self):
        if not self.files:
            messagebox.showinfo("Info", "Load a directory first.")
            return
        output = generate_concat(self.files, list(self.files.keys()))
        self._set_output(output)
        self.status_var.set(f"Concatenated {len(self.files)} files")

    def _concat_selected(self):
        selected = self._get_selected()
        if not selected:
            messagebox.showinfo("Info", "Select at least one file.")
            return
        output = generate_concat(self.files, selected)
        self._set_output(output)
        self.status_var.set(f"Concatenated {len(selected)} files")

    def _map_only(self):
        if not self.files:
            return
        selected = self._get_selected() or list(self.files.keys())
        filtered = {k: v for k, v in self.files.items() if k in selected}
        self._set_output(generate_map(filtered))
        self.status_var.set(f"Generated map for {len(filtered)} files")

    def _map_and_code(self):
        selected = self._get_selected()
        if not selected:
            messagebox.showinfo("Info", "Select at least one file.")
            return
        self._set_output(generate_map_and_code(self.files, selected))
        self.status_var.set(f"Map + code for {len(selected)} files")

    def _json_structure(self):
        selected = self._get_selected()
        if not selected:
            messagebox.showinfo("Info", "Select at least one file.")
            return
        data = generate_json_structure(self.files, selected)
        self._set_output(json.dumps(data, indent=2))
        self.status_var.set(f"JSON structure for {len(selected)} files")

    def _run_query(self):
        query = self.query_entry.get().strip()
        if not query:
            return
        selected = self._get_selected() or list(self.files.keys())
        self._set_output(process_query(self.files, selected, query))
        self.status_var.set(f"Query: {query}")

    def _query_help(self):
        self._set_output(QUERY_HELP)
        self.status_var.set("Showing query help")

    def _copy_output(self):
        text = self.output_text.get("1.0", "end-1c")
        if text:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self.status_var.set("Copied to clipboard")

    def _save_output(self):
        text = self.output_text.get("1.0", "end-1c")
        if not text:
            return
        filepath = filedialog.asksaveasfilename(
            title="Save Output",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filepath:
            Path(filepath).write_text(text, encoding="utf-8")
            self.status_var.set(f"Saved to {filepath}")

    # -------------------- Smart Locator Logic --------------------

    def _locator_on_file_select(self, event):
        sel = self.locator_file_combo.get()
        if sel:
            self._locator_load_file(sel)

    def _locator_load_file(self, rel_path: str):
        if rel_path not in self.files:
            return
        self.locator_current_rel_path = rel_path
        f = self.files[rel_path]
        
        # Re-read from disk to ensure freshness
        try:
            p = Path(f.full_path)
            try:
                content = p.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                content = p.read_text(encoding="latin-1")
            
            # Update internal state
            f.source = content
            f.line_count = len(content.splitlines())
        except Exception as e:
            content = f"Error reading file: {e}"

        self.locator_editor.delete("1.0", "end")
        self.locator_editor.insert("1.0", content)
        self.locator_status_var.set(f"Loaded {rel_path}")

    def _locator_reload(self):
        if self.locator_current_rel_path:
            self._locator_load_file(self.locator_current_rel_path)

    def _locator_save(self):
        if not self.locator_current_rel_path:
            return
        
        f = self.files[self.locator_current_rel_path]
        content = self.locator_editor.get("1.0", "end-1c")
        
        try:
            Path(f.full_path).write_text(content, encoding="utf-8")
            f.source = content
            f.line_count = len(content.splitlines())
            self.locator_status_var.set(f"Saved {f.rel_path}")
            messagebox.showinfo("Saved", f"Successfully saved {f.rel_path}")
        except Exception as e:
            messagebox.showerror("Save Error", str(e))

    def _locator_do_find(self):
        if not self.locator_current_rel_path:
            messagebox.showwarning("No File", "Select a file to edit first.")
            return

        search_text = self.locator_search_box.get("1.0", "end-1c")
        if not search_text.strip():
            self.locator_status_var.set("Search box empty.")
            return

        editor_content = self.locator_editor.get("1.0", "end-1c")
        
        # Run fuzzy matcher
        found, start, end, score = find_best_match(editor_content, search_text)
        
        self.locator_editor.tag_remove("highlight", "1.0", "end")
        
        if found:
            # Convert char indices to Tkinter indices
            def get_tk_index(char_idx):
                return f"1.0 + {char_idx} chars"

            idx_start = get_tk_index(start)
            idx_end = get_tk_index(end)
            
            self.locator_editor.tag_add("highlight", idx_start, idx_end)
            self.locator_editor.see(idx_start)
            
            # Select it so user can just paste
            self.locator_editor.mark_set("insert", idx_start)
            self.locator_editor.tag_add("sel", idx_start, idx_end)
            self.locator_editor.focus_set()
            
            msg = f"Match found ({int(score*100)}%). Highlighted."
            if score < 0.5:
                msg += " (Weak match)"
            self.locator_status_var.set(msg)
        else:
            self.locator_status_var.set("No match found.")


# ============================================================================
# CLI MODE
# ============================================================================

def cli_main(
    directory: str,
    output_file: Optional[str] = None,
    mode: str = "map",
    extensions: Optional[Set[str]] = None,
    recursive: bool = True,
    max_depth: Optional[int] = None,
):
    print(f"Parsing {directory}...")
    print(f"  Recursive: {recursive}")
    if max_depth is not None:
        print(f"  Max depth: {max_depth}")
    if extensions:
        print(f"  Extensions: {', '.join(sorted(extensions))}")

    files = parse_directory(
        directory,
        extensions=extensions,
        recursive=recursive,
        max_depth=max_depth,
    )

    print(f"Found {len(files)} files\n")

    type_counts: Dict[str, int] = {}
    for f in files.values():
        type_counts[f.file_type] = type_counts.get(f.file_type, 0) + 1
    for ftype, count in sorted(type_counts.items()):
        print(f"  {ftype}: {count}")
    print()

    selected = list(files.keys())

    if mode == "map":
        output = generate_map(files)
    elif mode == "concat":
        output = generate_concat(files, selected)
    elif mode == "both":
        output = generate_map_and_code(files, selected)
    elif mode == "json":
        data = generate_json_structure(files, selected)
        output = json.dumps(data, indent=2)
    else:
        output = generate_map(files)

    if output_file:
        Path(output_file).write_text(output, encoding="utf-8")
        print(f"Output saved to {output_file}")
    else:
        print(output)


def parse_type_arg(types_str: str) -> Set[str]:
    extensions = set()
    parts = types_str.split()
    for part in parts:
        part = part.strip().lower()
        if not part:
            continue
        if part in FILE_TYPES:
            extensions.update(FILE_TYPES[part]["extensions"])
        else:
            if not part.startswith("."):
                part = "." + part
            extensions.add(part)
    return extensions


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="KG Code Mapper v4.1 - Smart Locator Edition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                            GUI mode
  %(prog)s /path/to/project           GUI with pre-loaded project
  %(prog)s --cli /path                CLI mode, map output
  %(prog)s --cli /path -m concat      CLI mode, concatenated code
  %(prog)s --cli /path -t "py js"     Only Python and JavaScript
        """
    )
    parser.add_argument("directory", nargs="?", help="Directory to analyze")
    parser.add_argument("--cli", action="store_true", help="CLI mode (no GUI)")
    parser.add_argument("-o", "--output", help="Output file (CLI mode)")
    parser.add_argument("-m", "--mode", choices=["map", "concat", "both", "json"], default="map")
    parser.add_argument("-t", "--types", help="File types to include (e.g., 'py js html')")
    parser.add_argument("-r", "--recursive", dest="recursive", action="store_true", default=True)
    parser.add_argument("--no-recursive", dest="recursive", action="store_false")
    parser.add_argument("-d", "--depth", type=int, default=None)

    args = parser.parse_args()

    extensions = None
    if args.types:
        extensions = set()
        extensions.update(parse_type_arg(args.types))

    if args.cli:
        if not args.directory:
            print("Error: directory required in CLI mode")
            sys.exit(1)
        cli_main(
            args.directory,
            args.output,
            args.mode,
            extensions,
            args.recursive,
            args.depth,
        )
        return

    if not HAS_TK:
        print("Error: tkinter not available. Use --cli mode.")
        sys.exit(1)

    root = tk.Tk()
    style = ttk.Style()
    if "clam" in style.theme_names():
        style.theme_use("clam")

    app = CodeMapperApp(root, args.directory)
    root.mainloop()


if __name__ == "__main__":
    main()