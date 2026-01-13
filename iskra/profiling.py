# Copyright (c) 2025 - present, Ana Dodik. All rights reserved.

import functools
import json
import time
from collections import defaultdict, deque
from contextlib import ContextDecorator
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Callable, Literal, cast

import rich
from rich.align import Align
from rich.bar import Bar
from rich.columns import Columns
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.tree import Tree


@dataclass
class TreeNode[T: "TreeNode"]:
    name: str
    children: list[T] = field(default_factory=list)
    parent: T | None = None

    def parents(self) -> list["TreeNode"]:
        parent_names = []
        parent = self.parent
        while parent is not None:
            parent_names.append(parent)
            parent = parent.parent
        return list(reversed(parent_names))

    def add_child(self, child: T):
        self.children.append(child)
        assert child.parent is None
        child.parent = cast(T, self)

    def to_json(self):
        serialized = {}
        for f in fields(self):
            if f.name not in ["children", "parent"]:
                name = f.name
                value = getattr(self, name)
                serialized[f.name] = value
                if name.endswith("ns"):
                    name = name[:-2] + "ms"
                    serialized[f.name] = value * 1e-6
        serialized["children"] = []
        for child in self.children:
            serialized["children"].append(child.to_json())
        return serialized


def flatten_trees[T: TreeNode](trees: list[T]):
    name_map = defaultdict(list[T])
    for root in trees:
        stack: deque[T] = deque([root])
        while len(stack) > 0:
            node = stack.popleft()
            names = [n.name for n in node.parents()] + [node.name]
            name_map[tuple(names)].append(node)

            for child in node.children:
                stack.append(child)
    return name_map


@dataclass
class ProfileSegment(TreeNode):
    time_start_ns: int = -1
    time_end_ns: int = -1

    @property
    def time_diff_ns(self):
        return self.time_end_ns - self.time_start_ns

    @property
    def time_diff_ms(self):
        return self.time_diff_ns * 1e-6

    @property
    def time_diff(self):
        return self.time_diff_ns * 1e-9

    def __str__(self) -> str:
        if self.time_end_ns == -1:
            return f"{self.name}: did not terminate"
        return f"{self.name}: {self.time_diff_ms:.4f} ms"

    def rich_format(self):
        label = Text()
        label.append(self.name, style="bold cyan")
        label.append(": ")
        label.append(f"{self.time_diff_ms:.2f}ms", style="bold magenta underline")
        return label


@dataclass
class SegmentSummary(TreeNode):
    mean_time_ns: float = -1
    min_time_ns: int = -1
    max_time_ns: int = -1
    count: int = -1

    @property
    def mean_time_ms(self):
        return self.mean_time_ns * 1e-6

    @property
    def min_time_ms(self):
        return self.min_time_ns * 1e-6

    @property
    def max_time_ms(self):
        return self.max_time_ns * 1e-6

    def rich_format(self):
        formatted = {
            "Name": Text(self.name, style="bold cyan"),
            "Count": Text(f"{self.count}x", style="italic"),
            "Mean (ms)": Text(f"~{self.mean_time_ms:.2f}", style="bold magenta"),
            "range": Text(
                f"{self.min_time_ms:.2f}-{self.max_time_ms:.2f}",
                style="bold magenta",
            ),
            "Min (ms)": Text(f"{self.min_time_ms:.2f}", style="italic"),
            "Max (ms)": Text(f"{self.max_time_ms:.2f}", style="italic"),
        }
        return formatted


def compute_summary(segments: list[ProfileSegment]):
    assert len(segments) > 0
    count = 0
    min_time_ns = float("inf")
    max_time_ns = -float("inf")
    mean_time_ns = 0
    block_name = segments[0].name
    for segment in segments:
        if segment.time_end_ns == -1:
            continue
        assert segment.name == block_name
        count += 1
        mean_time_ns += segment.time_diff_ns
        min_time_ns = min(min_time_ns, segment.time_diff_ns)
        max_time_ns = max(max_time_ns, segment.time_diff_ns)
    mean_time_ns /= count

    return SegmentSummary(
        name=block_name,
        mean_time_ns=mean_time_ns,
        min_time_ns=int(min_time_ns),
        max_time_ns=int(max_time_ns),
        count=count,
    )


def display_profiling_table(roots: list[SegmentSummary]):
    table = Table(
        title="[bold magenta]Iskra Profiling Results[/bold magenta]",
        border_style="bright_blue",
        header_style="bold white on blue",
        padding=(0, 1),
        expand=False,
    )

    field_names = ["Name", "Count", "Mean (ms)", "Min (ms)", "Max (ms)"]
    for field_name in field_names:
        if field_name == "Name":
            table.add_column(field_name, style="cyan", no_wrap=False)
        else:
            table.add_column(field_name, justify="right", style="bold", width=12)

    def add_rows(node, depth=0, is_last=False, prefix=""):
        if depth == 0:
            branch = ""
            new_prefix = ""
        else:
            branch = "└── " if is_last else "├── "
            new_prefix = prefix + ("    " if is_last else "│   ")
        formatted = node.rich_format()
        row_values = []
        for i, field_name in enumerate(field_names):
            if field_name == "Name":
                name_text = Text()
                name_text.append(prefix + branch, style="bright_black")
                name_text.append_text(formatted[field_name])
                row_values.append(name_text)
            else:
                row_values.append(formatted[field_name])
        table.add_row(*row_values)
        for i, child in enumerate(node.children):
            add_rows(child, depth + 1, i == len(node.children) - 1, new_prefix)

    for root in roots:
        add_rows(root)
    return table


class Profiler:
    def __init__(self):
        self.running_stack: list[ProfileSegment] = []
        self.segment_trees: list[ProfileSegment] = []

    def push(self, name: str):
        segment = ProfileSegment(name=name, time_start_ns=time.perf_counter_ns())
        if len(self.running_stack) > 0:
            self.running_stack[-1].add_child(segment)
        else:
            self.segment_trees.append(segment)
        self.running_stack.append(segment)

    def reset(self):
        self.running_stack = []
        self.segment_trees = []

    def pop(self):
        self.running_stack[-1].time_end_ns = time.perf_counter_ns()
        self.running_stack.pop()

    def summary(self):
        block_map = flatten_trees(self.segment_trees)
        roots = []
        summary_map: dict[tuple[str, ...], SegmentSummary] = {}
        for k, v in block_map.items():
            summary = compute_summary(v)
            summary_map[k] = summary
            if len(k) == 1:
                roots.append(summary)
        for k, v in summary_map.items():
            if len(k) > 1:
                summary_map[k[:-1]].add_child(v)
        return roots

    def all_to_json(self):
        result = []
        for root in self.segment_trees:
            result.append(root.to_json())
        return result

    def summary_to_json(self):
        summary_trees = self.summary()
        result = []
        for root in summary_trees:
            result.append(root.to_json())
        return result

    def dump(
        self, summary_type: Literal["none", "stats"] = "stats", path: Path | None = None
    ):
        if summary_type == "none":
            tree = self.segment_trees
        else:
            assert summary_type == "stats"
            tree = self.summary()
        rich_tree = Tree("root")
        for root in tree:
            stack = [root]
            rich_stack = [rich_tree.add(root.rich_format())]
            while len(stack) > 0:
                node = stack.pop()
                rich_node = rich_stack.pop()
                for child in node.children:
                    rich_stack.append(rich_node.add(child.rich_format()))
                    stack.append(child)

        table = display_profiling_table(tree)
        console = Console(
            width=200, soft_wrap=True, color_system="truecolor", record=True
        )
        console.print(table)
        if path is not None:
            console.save_text(str(path) + ".txt", clear=False)
            console.save_svg(str(path) + ".svg", clear=False)


global_profiler = Profiler()


def profile_fn[T, **P](
    fn: Callable[P, T] | None = None,
    *,
    name: str | None = None,
    profiler: Profiler = global_profiler,
) -> Callable[[Callable[P, T]], Callable[P, T]] | Callable[P, T]:
    def decorator(fn: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            block_name = name
            if block_name is None:
                block_name = fn.__qualname__
            profiler.push(block_name)
            result: T = fn(*args, **kwargs)
            profiler.pop()
            return result

        return wrapper

    if callable(fn):
        return decorator(fn)

    return decorator


class profile_block(ContextDecorator):  # noqa: N801
    def __init__(self, name: str, profiler: Profiler = global_profiler):
        self.name = name
        self.profiler = profiler

    def __enter__(self):
        self.profiler.push(self.name)

    def __exit__(self, exc_type, exc, exc_tb):
        self.profiler.pop()
