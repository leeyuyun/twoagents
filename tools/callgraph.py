#!/usr/bin/env python
from __future__ import annotations

import argparse
import ast
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


@dataclass
class FuncDef:
    qname: str
    module: str
    cls: Optional[str]
    name: str


@dataclass
class ModuleInfo:
    name: str
    path: Path
    imports: Dict[str, str] = field(default_factory=dict)  # alias -> target
    funcs: Dict[str, FuncDef] = field(default_factory=dict)  # qname -> def


class IndexBuilder(ast.NodeVisitor):
    def __init__(self, mod: ModuleInfo) -> None:
        self.mod = mod
        self.class_stack: List[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            asname = alias.asname or alias.name
            self.mod.imports[asname] = alias.name
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module is None:
            return
        for alias in node.names:
            asname = alias.asname or alias.name
            self.mod.imports[asname] = f"{node.module}.{alias.name}"
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.class_stack.append(node.name)
        self.generic_visit(node)
        self.class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._add_func(node.name)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._add_func(node.name)
        self.generic_visit(node)

    def _add_func(self, name: str) -> None:
        cls = self.class_stack[-1] if self.class_stack else None
        if cls:
            qname = f"{self.mod.name}.{cls}.{name}"
        else:
            qname = f"{self.mod.name}.{name}"
        self.mod.funcs[qname] = FuncDef(qname=qname, module=self.mod.name, cls=cls, name=name)


class CallGraphBuilder(ast.NodeVisitor):
    def __init__(
        self,
        mod: ModuleInfo,
        all_funcs_by_name: Dict[str, Set[str]],
        modules: Set[str],
        class_methods: Dict[Tuple[str, str], Set[str]],
    ) -> None:
        self.mod = mod
        self.all_funcs_by_name = all_funcs_by_name
        self.modules = modules
        self.class_methods = class_methods
        self.current_func: Optional[FuncDef] = None
        self.edges: Set[Tuple[str, str]] = set()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.current_func = self._resolve_current_func(node.name)
        self.generic_visit(node)
        self.current_func = None

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.current_func = self._resolve_current_func(node.name)
        self.generic_visit(node)
        self.current_func = None

    def visit_Call(self, node: ast.Call) -> None:
        if not self.current_func:
            return
        target = self._resolve_call_target(node.func)
        if target:
            self.edges.add((self.current_func.qname, target))
        self.generic_visit(node)

    def _resolve_current_func(self, name: str) -> Optional[FuncDef]:
        # Prefer class context if any
        if self.mod.funcs:
            for qname, fdef in self.mod.funcs.items():
                if fdef.name == name:
                    if fdef.cls is not None and self._is_in_class_scope(qname):
                        return fdef
        # Fallback: first match
        for qname, fdef in self.mod.funcs.items():
            if fdef.name == name:
                return fdef
        return None

    def _is_in_class_scope(self, qname: str) -> bool:
        # Heuristic: if class name appears in qualified name, accept.
        return "." in qname

    def _resolve_call_target(self, node: ast.AST) -> Optional[str]:
        # Name(...) or module.func(...) or self.method(...)
        if isinstance(node, ast.Name):
            return self._resolve_name(node.id)
        if isinstance(node, ast.Attribute):
            return self._resolve_attribute(node)
        return None

    def _resolve_name(self, name: str) -> Optional[str]:
        # Same module function
        qname = f"{self.mod.name}.{name}"
        if qname in self.mod.funcs:
            return qname
        # Any unique match across project
        candidates = self.all_funcs_by_name.get(name, set())
        if len(candidates) == 1:
            return next(iter(candidates))
        return None

    def _resolve_attribute(self, node: ast.Attribute) -> Optional[str]:
        attr = node.attr
        base = node.value
        # self.method or cls.method
        if isinstance(base, ast.Name) and base.id in {"self", "cls"}:
            if self.current_func and self.current_func.cls:
                key = (self.mod.name, self.current_func.cls)
                methods = self.class_methods.get(key, set())
                for qname in methods:
                    if qname.endswith(f".{attr}"):
                        return qname
            return None
        # module.func or Class.method
        if isinstance(base, ast.Name):
            base_name = base.id
            # resolve aliases from imports
            base_target = self.mod.imports.get(base_name, base_name)
            # module.func
            if base_target in self.modules:
                qname = f"{base_target}.{attr}"
                if qname in all_funcs:
                    return qname
            # module.Class.method
            if "." in base_target:
                mod_part, class_part = base_target.rsplit(".", 1)
                key = (mod_part, class_part)
                methods = self.class_methods.get(key, set())
                for qname in methods:
                    if qname.endswith(f".{attr}"):
                        return qname
        return None


def parse_module(path: Path) -> ast.AST:
    # Handle BOM if present
    return ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))


def collect_modules(paths: Iterable[Path]) -> Dict[str, ModuleInfo]:
    modules: Dict[str, ModuleInfo] = {}
    for p in paths:
        mod = p.stem
        modules[mod] = ModuleInfo(name=mod, path=p)
    for mod in modules.values():
        tree = parse_module(mod.path)
        IndexBuilder(mod).visit(tree)
    return modules


def build_call_graph(modules: Dict[str, ModuleInfo]) -> Set[Tuple[str, str]]:
    all_funcs_by_name: Dict[str, Set[str]] = {}
    class_methods: Dict[Tuple[str, str], Set[str]] = {}
    global all_funcs
    all_funcs = set()

    for mod in modules.values():
        for qname, fdef in mod.funcs.items():
            all_funcs.add(qname)
            all_funcs_by_name.setdefault(fdef.name, set()).add(qname)
            if fdef.cls:
                key = (mod.name, fdef.cls)
                class_methods.setdefault(key, set()).add(qname)

    edges: Set[Tuple[str, str]] = set()
    for mod in modules.values():
        tree = parse_module(mod.path)
        builder = CallGraphBuilder(
            mod, all_funcs_by_name, set(modules.keys()), class_methods
        )
        builder.visit(tree)
        edges.update(builder.edges)
    return edges


def write_dot(modules: Dict[str, ModuleInfo], edges: Set[Tuple[str, str]], out: Path) -> None:
    lines: List[str] = []
    lines.append("digraph callgraph {")
    lines.append("  rankdir=LR;")
    lines.append("  node [shape=box, fontsize=10];")
    for mod in modules.values():
        lines.append(f'  subgraph "cluster_{mod.name}" {{')
        lines.append(f'    label="{mod.name}";')
        for qname in sorted(mod.funcs.keys()):
            lines.append(f'    "{qname}";')
        lines.append("  }")
    for src, dst in sorted(edges):
        lines.append(f'  "{src}" -> "{dst}";')
    lines.append("}")
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a simple call graph.")
    parser.add_argument("files", nargs="+", help="Python files to analyze")
    parser.add_argument("--out", default="out/callgraph.dot", help="DOT output path")
    args = parser.parse_args()

    paths = [Path(f) for f in args.files]
    modules = collect_modules(paths)
    edges = build_call_graph(modules)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_dot(modules, edges, out_path)


if __name__ == "__main__":
    main()
