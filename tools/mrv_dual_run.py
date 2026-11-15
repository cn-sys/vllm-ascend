# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""Offline planner for the MRV1/MRV2 dual runs of the pinned GLM E2E configs.

``VLLM_USE_V2_MODEL_RUNNER`` is read once per server process, so a parity run
needs two launches whose configs pin it in opposite directions. Fourteen GLM E2E
configs pin it themselves and the yaml wins over ``os.environ``, so ``export``
cannot flip them: the copies below have to be rewritten. ``discover`` lists those
configs with line, value and carrier family; ``plan`` copies each into two trees
under ``--dest`` (one per backend), rewrites only the pinned line and emits the
serve commands; ``rewrite`` flips one file into a copy and proves that exactly
that line changed; ``verify-log`` proves the runner from the server log the Ascend
plugin prints (``vllm_ascend/mrv2_utils.py``), and only an explicit
``VLLM_USE_V2_MODEL_RUNNER=... is set; using Model Runner V1|V2`` line passes: an
``is unset`` default or a blacklist fallback means the yaml/env never reached the
server, so it fails even when the effective runner happens to match; ``manifest``
writes the run skeleton and leaves every unreadable value ``null``.

``plan`` emits every serve command as ``display`` (the text read from the yaml),
``argv`` only when ``shlex`` can rebuild that display exactly (otherwise
``null``), plus ``requires_runner_render`` and the ``${VAR}`` ``placeholders``.

Carriers, read from the runners: ``test_cases[].envs`` (``conftest.py``:
``env = os.environ.copy(); env.update(env_dict)``), ``deployment[].envs`` on the
``env_common`` anchor (``multi_node_config.py``: ``Popen(env={**os.environ,
**self.envs})``) and ``external templates[].envs`` (``external_dp_config.py`` ->
``utils.py`` merge ``{**os.environ, **env}`` over ``env_common`` ->
``env_prefill``/``env_decode``).

Carrier rule: a pin line inside a top level ``test_cases``, ``deployment`` or
``templates`` section belongs to it; a pin line inside an anchor definition
(``env_common: &env_common``, ``_envs: &envs``) belongs to the carrier section
that merges it through the ``<<: *anchor`` chain, with ``templates`` over
``deployment`` over ``test_cases`` when a config merges it into several.

Only the standard library plus ``regex`` (the repository lint whitelist, because
``import re`` is forbidden by ``tools/check_forbidden_imports.py``) is imported and
PyYAML never is: the tool runs on a CPU-only host and must not re-serialize a
config, so copies keep comments, quotes and layout byte for byte except for the
pinned value. Exit codes: 0 ok, 1
verification failed, 2 unusable input or missing premise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import regex as re

EXIT_OK, EXIT_MISMATCH, EXIT_INPUT_ERROR = 0, 1, 2
BACKEND_ENV_VAR = "VLLM_USE_V2_MODEL_RUNNER"
BACKENDS = ("mrv1", "mrv2")
BACKEND_VALUES = {"mrv1": "0", "mrv2": "1"}
LOG_TAGS = {"mrv1": "V1", "mrv2": "V2"}
LOG_TAG_BACKENDS = {tag: backend for backend, tag in LOG_TAGS.items()}
CARRIERS = ("test_cases", "deployment", "external_templates")
SECTION_CARRIERS = {"test_cases": "test_cases", "deployment": "deployment", "templates": "external_templates"}
CARRIER_PRIORITY = ("external_templates", "deployment", "test_cases")
GLM_MODEL_CONFIG_DIR = "tests/e2e/cases/models/configs/GLM"
FEATURE_CONFIG_DIR = "tests/e2e/cases/features"
GLM_FAMILY_MARKER = "glm"
CASE_TREE = "tests/e2e/cases"
RELEASE_TAG_FILE = ".github/vllm-release-tag.commit"
TOOLS_DIR = Path(__file__).resolve().parent
DEFAULT_REPO_ROOT = TOOLS_DIR.parent
MANIFEST_FIELDS = (
    "repo_commit", "vllm_release_tag", "image_tag", "cann_version", "torch_npu_version", "model_id",
    "model_revision", "hardware", "graph_mode", "config_yaml", "backend_expected", "server_log_sha256",
    "commands", "generated_at",
)  # fmt: skip
PIN_PATTERN = re.compile(
    rf"^(?P<indent>[ \t]*){BACKEND_ENV_VAR}[ \t]*:[ \t]*(?P<quote>['\"]?)(?P<value>[^'\"\s#]+)(?P=quote)(?P<rest>.*)$"
)
SECTION_PATTERN = re.compile(r"^(?P<key>[A-Za-z_][A-Za-z0-9_]*)[ \t]*:")
ANCHOR_PATTERN = re.compile(r"&[ \t]*(?P<name>[A-Za-z_][A-Za-z0-9_]*)")
ALIAS_PATTERN = re.compile(r"\*(?P<name>[A-Za-z_][A-Za-z0-9_]*)")
EXPLICIT_PATTERN = re.compile(r"VLLM_USE_V2_MODEL_RUNNER=(?P<value>\S+) is set; using Model Runner (?P<tag>V1|V2)")
UNSET_PATTERN = re.compile(r"VLLM_USE_V2_MODEL_RUNNER is unset; using Model Runner V2 by default")
FALLBACK_PATTERN = re.compile(r"Model Runner V2 does not yet support .*; using the V1 model runner instead")
PLACEHOLDER_PATTERN = re.compile(r"\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)\}")
BRACED_SPACE_PATTERN = re.compile(r"\{[^{}]*\s[^{}]*\}")
TRUTHY_RUNNER_VALUES = frozenset({"1", "true", "on", "yes"})
FALSY_RUNNER_VALUES = frozenset({"0", "false", "off", "no"})
BLOCK_INDICATORS = (">", "|")
EXAMPLES = """examples:
  python tools/mrv_dual_run.py discover
  python tools/mrv_dual_run.py plan --dest /tmp/mrv-dual --print-only
  python tools/mrv_dual_run.py plan --dest /tmp/mrv-dual
  python tools/mrv_dual_run.py rewrite --src <yaml> --dest /tmp/mrv-dual/single --value 0
  python tools/mrv_dual_run.py verify-log --log server.log --expect mrv1
  python tools/mrv_dual_run.py manifest --out manifest.json --backend-expected mrv1 --server-log server.log
"""


class InputError(ValueError):
    """The input, the destination or the premise of a subcommand is unusable."""


class CommandNotFound(InputError):
    """A config has no serve command that could be extracted without inventing one."""


@dataclass(frozen=True)
class Pin:
    """One runner assignment as it is written in one yaml line."""

    line: int
    value: str
    start: int
    end: int


@dataclass(frozen=True)
class Section:
    """One top level yaml key with the line range it owns."""

    key: str
    start: int
    end: int
    anchor: str | None


@dataclass(frozen=True)
class ConfigInfo:
    """A GLM config that pins the runner variable."""

    path: Path
    rel: str
    carrier: str
    pin: Pin


def fail(message: str, code: int = EXIT_INPUT_ERROR) -> int:
    """Print one error on stderr and return the exit code that carries it."""
    print(f"error: {message}", file=sys.stderr)
    return code


def find_pins(text: str) -> list[Pin]:
    """Every runner assignment in ``text``, comment lines excluded."""
    matches = [(number, PIN_PATTERN.match(line)) for number, line in enumerate(text.splitlines(), start=1)]
    return [Pin(number, m.group("value"), m.start("value"), m.end("value")) for number, m in matches if m]


def top_level_sections(lines: list[str]) -> list[Section]:
    """Index the top level yaml keys with their line ranges and anchors."""
    found = [
        (index, match.group("key"), (anchor := ANCHOR_PATTERN.search(line)) and anchor.group("name"))
        for index, line in enumerate(lines)
        if line and line[0] not in " \t#" and (match := SECTION_PATTERN.match(line))
    ]
    bounds = [start for start, _, _ in found[1:]] + [len(lines)]
    return [Section(key, start, stop, anchor) for (start, key, anchor), stop in zip(found, bounds, strict=True)]


def classify_carrier(text: str) -> str:
    """Name the yaml carrier family that delivers this runner pin to the server process."""
    pins, lines = find_pins(text), text.splitlines()
    if not pins:
        raise InputError(f"no {BACKEND_ENV_VAR} line in the config")
    sections = top_level_sections(lines)
    section = next((item for item in sections if item.start <= pins[0].line - 1 < item.end), None)
    if section is None:
        raise InputError(f"{BACKEND_ENV_VAR} is outside every top level section")
    if section.key in SECTION_CARRIERS:
        return SECTION_CARRIERS[section.key]
    if section.anchor is None:
        raise InputError(f"{BACKEND_ENV_VAR} sits in {section.key!r}, which is neither a carrier nor an anchor")

    def merged(item: Section, names: set[str]) -> bool:
        return any(
            a.group("name") in names for line in lines[item.start + 1 : item.end] for a in ALIAS_PATTERN.finditer(line)
        )

    closure = {section.anchor}
    while True:
        extra = {
            item.anchor for item in sections if item.anchor and item.anchor not in closure and merged(item, closure)
        }
        if not extra:
            break
        closure |= extra
    carriers = {
        SECTION_CARRIERS[item.key] for item in sections if item.key in SECTION_CARRIERS and merged(item, closure)
    }
    if not carriers:
        raise InputError(f"anchor {section.anchor!r} is not merged by test_cases, deployment or templates")
    return next(carrier for carrier in CARRIER_PRIORITY if carrier in carriers)


def stray_pin_lines(text: str) -> list[int]:
    """Lines that mention the runner variable in a form :func:`find_pins` cannot read."""
    strays = []
    for number, line in enumerate(text.splitlines(), start=1):
        if BACKEND_ENV_VAR in line and PIN_PATTERN.match(line) is None and not line.lstrip().startswith("#"):
            strays.append(number)
    return strays


def discover_configs(repo_root: Path) -> list[ConfigInfo]:
    """Find the GLM configs that pin the runner and classify each carrier.

    A config that mentions the variable in a shape this scanner cannot read (a
    flow mapping such as ``envs: {VLLM_USE_V2_MODEL_RUNNER: "1"}``, for example)
    raises instead of being skipped: a silently incomplete parity matrix is worse
    than a failing command.
    """
    model_dir, feature_dir = repo_root / GLM_MODEL_CONFIG_DIR, repo_root / FEATURE_CONFIG_DIR
    paths = sorted(model_dir.glob("*.yaml")) if model_dir.is_dir() else []
    if feature_dir.is_dir():
        paths += [path for path in sorted(feature_dir.rglob("*.yaml")) if GLM_FAMILY_MARKER in path.name.lower()]
    found: list[ConfigInfo] = []
    problems: list[str] = []
    for path in sorted(set(paths)):
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(repo_root).as_posix()
        pins = find_pins(text)
        if pins:
            found.append(ConfigInfo(path, rel, classify_carrier(text), pins[0]))
            continue
        strays = stray_pin_lines(text)
        if strays:
            problems.append(f"{rel} line(s) {', '.join(str(number) for number in strays)}")
    if problems:
        raise InputError(
            f"{BACKEND_ENV_VAR} cannot be read in {len(problems)} config(s): {'; '.join(problems)}. "
            "Only a block styled 'VLLM_USE_V2_MODEL_RUNNER: \"0\"' line is understood; "
            "refusing to drop these configs from the matrix silently"
        )
    return found


def parse_scalar(raw: str) -> str | None:
    """Read a one line yaml scalar, dropping quotes and a trailing comment."""
    text = raw.strip()
    if not text or text.startswith("#"):
        return None
    if len(text) >= 2 and text[0] in "'\"" and text[-1] == text[0]:
        return text[1:-1]
    return (text.split(" #", 1)[0].strip() or None) if " #" in text else text


def find_key_line(lines: list[str], start: int, end: int, key: str) -> tuple[int, int, str] | None:
    """First line in ``[start, end)`` that assigns ``key``, with its indent and raw tail."""
    pattern = re.compile(r"^(?P<indent>[ \t]*)" + re.escape(key) + r"[ \t]*:(?P<rest>.*)$")
    for index in range(start, end):
        match = pattern.match(lines[index])
        if match is not None:
            return index, len(match.group("indent")), match.group("rest")
    return None


def indented_parts(lines: list[str], key_index: int, key_indent: int) -> list[str]:
    """Stripped lines of the yaml block that follows one key, stopping at the dedent."""
    parts: list[str] = []
    for line in lines[key_index + 1 :]:
        if not line.strip():
            continue
        if len(line) - len(line.lstrip()) <= key_indent:
            break
        parts.append(line.strip())
    return parts


def block_list_items(lines: list[str], key_index: int, key_indent: int) -> list[str]:
    """Arguments of a yaml block list written under one key."""
    return [
        parse_scalar(part[1:])
        for part in indented_parts(lines, key_index, key_indent)
        if part.startswith("-") and parse_scalar(part[1:]) is not None
    ]


def list_items(lines: list[str], section: Section) -> list[tuple[int, int]]:
    """Line ranges of the block list items directly inside ``section``."""
    starts: list[int] = []
    for index in range(section.start + 1, section.end):
        body = lines[index].strip()
        if not body or not body.startswith("-") or body.startswith("#"):
            continue
        if len(lines[index]) - len(lines[index].lstrip()) == (
            len(lines[starts[0]]) - len(lines[starts[0]].lstrip())
            if starts
            else len(lines[index]) - len(lines[index].lstrip())
        ):
            starts.append(index)
    return list(zip(starts, starts[1:] + [section.end], strict=True))


def anchor_list_args(lines: list[str], sections: list[Section], name: str) -> list[str]:
    """Arguments of an anchor that carries a block list, e.g. ``server_cmd: *server_cmd``."""
    section = next((item for item in sections if item.anchor == name), None)
    if section is None:
        raise CommandNotFound(f"anchor &{name} is not defined in this file")
    args = block_list_items(lines, section.start, len(lines[section.start]) - len(lines[section.start].lstrip()))
    if not args:
        raise CommandNotFound(f"anchor &{name} does not carry a list of arguments")
    return args


def model_of_command(command: str) -> str | None:
    """Served model name of a command that already starts with ``vllm serve``."""
    tokens = command.split()
    return tokens[tokens.index("serve") + 1] if "serve" in tokens[:-1] else None


def command_record(node: str, display: str, source: str, notes: list[str] | None = None) -> dict[str, Any]:
    """Describe one serve command without pretending that it can always be split.

    ``display`` keeps the text read from the yaml. ``argv`` is filled only when
    ``shlex`` rebuilds the normalised display exactly, which fails for arguments
    with embedded spaces such as a JSON override; ``requires_runner_render`` marks
    commands with ``${VAR}`` placeholders or with such JSON, because the runner has
    to render them before anything may execute the list.
    """
    normalized = " ".join(display.split())
    placeholders = list(dict.fromkeys(PLACEHOLDER_PATTERN.findall(display)))
    try:
        argv: list[str] | None = shlex.split(normalized)
    except ValueError:
        argv = []
    if not argv or shlex.join(argv) != normalized:
        argv = None
    return {
        "node": node,
        "display": display,
        "argv": argv,
        # A command whose argv is not trustworthy cannot be executed as-is, so it
        # needs the runner to render it, just like the placeholder and JSON cases.
        "requires_runner_render": bool(placeholders) or bool(BRACED_SPACE_PATTERN.search(display)) or argv is None,
        "placeholders": placeholders,
        "source": source,
        "model": model_of_command(display),
        "notes": notes or [],
    }


def extract_templates(lines: list[str], section: Section) -> list[dict[str, Any]]:
    """Serve commands of an external DP config: one per ``templates[]`` node."""
    model_line = find_key_line(lines, 0, section.start, "model")
    model = parse_scalar(model_line[2]) if model_line is not None else None
    if not model:
        raise CommandNotFound("templates section found, but the top level 'model' key is missing")
    commands = []
    for number, (start, stop) in enumerate(list_items(lines, section)):
        node_line = find_key_line(lines, start, stop, "node_index")
        args_line = find_key_line(lines, start, stop, "server_cmd_template")
        args = block_list_items(lines, args_line[0], args_line[1]) if args_line is not None else []
        node = f"node_index={parse_scalar(node_line[2])}" if node_line is not None else f"templates[{number}]"
        if not args:
            raise CommandNotFound(f"templates[{number}] ({node}) has no server_cmd_template arguments")
        display = " ".join(["vllm", "serve", model, *args])
        commands.append(command_record(node, display, f"templates[{number}].server_cmd_template + top level model"))
    if not commands:
        raise CommandNotFound("templates section has no entries")
    return commands


def extract_deployment(lines: list[str], section: Section) -> list[dict[str, Any]]:
    """Serve commands of a multi node config: one per ``deployment[]`` entry."""
    commands = []
    for number, (start, stop) in enumerate(list_items(lines, section)):
        found = find_key_line(lines, start, stop, "server_cmd")
        if found is None:
            raise CommandNotFound(f"deployment[{number}] has no server_cmd")
        key_index, key_indent, rest = found
        indicator = rest.strip()[:1]
        if indicator == "|":
            raise CommandNotFound(
                f"deployment[{number}].server_cmd is a literal block scalar this tool does not unfold"
            )
        if indicator in BLOCK_INDICATORS:
            command, source = " ".join(indented_parts(lines, key_index, key_indent)), "folded block"
        else:
            command, source = parse_scalar(rest) or "", "inline"
        if not command:
            raise CommandNotFound(f"deployment[{number}].server_cmd is empty")
        commands.append(command_record(f"deployment[{number}]", command, f"deployment[{number}].server_cmd ({source})"))
    if not commands:
        raise CommandNotFound("deployment section has no entries")
    return commands


def extract_test_cases(lines: list[str], sections: list[Section], section: Section) -> list[dict[str, Any]]:
    """Serve commands of a single node config: one per ``test_cases[]`` entry."""
    commands = []
    for number, (start, stop) in enumerate(list_items(lines, section)):
        name_line = find_key_line(lines, start, stop, "name")
        node = (parse_scalar(name_line[2]) if name_line is not None else None) or f"test_cases[{number}]"
        model_line = find_key_line(lines, start, stop, "model")
        model = parse_scalar(model_line[2]) if model_line is not None else None
        found = find_key_line(lines, start, stop, "server_cmd")
        if found is None:
            raise CommandNotFound(f"test_cases[{number}] ({node}) has no server_cmd")
        key_index, key_indent, rest = found
        raw = rest.strip()
        notes: list[str] = []
        if raw.startswith("*"):
            alias = ALIAS_PATTERN.search(raw)
            if alias is None:
                raise CommandNotFound(f"test_cases[{number}].server_cmd has an unreadable alias {raw!r}")
            command, source = (
                shlex.join(anchor_list_args(lines, sections, alias.group("name"))),
                f"*{alias.group('name')}",
            )
        elif raw:
            command, source = parse_scalar(raw) or "", "inline"
        else:
            # Keep the arguments shell-safe: a JSON argument with spaces must
            # stay one token when the executor passes the display to sh -c.
            command, source = shlex.join(block_list_items(lines, key_index, key_indent)), "block list"
        if not command:
            raise CommandNotFound(f"test_cases[{number}].server_cmd is empty")
        if model_of_command(command) is None:
            if not model:
                raise CommandNotFound(f"test_cases[{number}].server_cmd lists only flags and the case has no model")
            command = shlex.join(["vllm", "serve", model, *shlex.split(command)])
            notes.append(
                "'vllm serve <model>' prefix added: the case runner composes it before the server_cmd arguments"
            )
        commands.append(command_record(node, command, f"test_cases[{number}].server_cmd ({source})", notes))
    if not commands:
        raise CommandNotFound("test_cases section has no entries")
    return commands


def extract_commands(text: str) -> list[dict[str, Any]]:
    """Extract the serve command(s) one config would launch, by config shape."""
    lines = text.splitlines()
    sections = top_level_sections(lines)
    section = next((item for item in sections if item.key in SECTION_CARRIERS), None)
    if section is None:
        raise CommandNotFound("no test_cases, deployment or templates section")
    if section.key == "templates":
        return extract_templates(lines, section)
    return (
        extract_deployment(lines, section)
        if section.key == "deployment"
        else extract_test_cases(lines, sections, section)
    )


def rewrite_pin_value(text: str, value: str) -> tuple[str, list[int]]:
    """Replace the value of every runner pin, leaving quotes, comments and layout untouched."""
    lines = text.splitlines(keepends=True)
    rewritten: list[str] = []
    changed: list[int] = []
    for number, line in enumerate(lines, start=1):
        match = PIN_PATTERN.match(line)
        if match is not None and match.group("value") != value:
            line = line[: match.start("value")] + value + line[match.end("value") :]
            changed.append(number)
        rewritten.append(line)
    if not find_pins(text):
        raise InputError(f"no {BACKEND_ENV_VAR} line to rewrite")
    return "".join(rewritten), changed


def verify_rewritten(src_text: str, dest_text: str, value: str, must_change: bool) -> list[int]:
    """Prove a copy carries ``value`` and that only the pinned line moved."""
    dest_pins = find_pins(dest_text)
    if not dest_pins:
        raise InputError(f"the copy lost the {BACKEND_ENV_VAR} line")
    wrong = [pin.line for pin in dest_pins if pin.value != value]
    if wrong:
        raise InputError(f"the copy still carries a wrong value on line(s) {wrong}, expected {value!r}")
    left, right = src_text.splitlines(keepends=True), dest_text.splitlines(keepends=True)
    if len(left) != len(right):
        raise InputError(f"line count changed ({len(left)} -> {len(right)})")
    pinned = {pin.line for pin in find_pins(src_text)}
    changed = {number for number, (old, new) in enumerate(zip(left, right, strict=True), start=1) if old != new}
    if not changed <= pinned:
        raise InputError(f"line(s) {sorted(changed - pinned)} changed but only {sorted(pinned)} may change")
    if must_change and changed != pinned:
        raise InputError(f"expected line(s) {sorted(pinned)} to change, got {sorted(changed)}")
    if not must_change and changed:
        raise InputError(f"expected an untouched copy, but line(s) {sorted(changed)} differ")
    return sorted(changed)


def carrier_counts(configs: list[ConfigInfo]) -> dict[str, int]:
    """How many configs use each carrier family."""
    return {carrier: sum(1 for info in configs if info.carrier == carrier) for carrier in CARRIERS}


def plan_target(
    info: ConfigInfo, text: str, commands: list[dict[str, Any]], target: str, dest_root: Path, print_only: bool
) -> dict[str, Any]:
    """Copy one config for one target backend and describe the resulting state."""
    value = BACKEND_VALUES[target]
    action = "rewrite" if info.pin.value != value else "no-change-needed"
    dest_path = None
    if not print_only:
        new_text = rewrite_pin_value(text, value)[0] if action == "rewrite" else text
        dest_file = dest_root / target / info.rel
        dest_file.parent.mkdir(parents=True, exist_ok=True)
        dest_file.write_text(new_text, encoding="utf-8")
        verify_rewritten(text, dest_file.read_text(encoding="utf-8"), value, action == "rewrite")
        dest_path = str(dest_file)
    return {
        "value": value,
        "action": action,
        "env_prefix": f"{BACKEND_ENV_VAR}={value}",
        "dest": dest_path,
        "commands": commands,
    }


def run_discover(args: argparse.Namespace) -> int:
    """Entry point of the ``discover`` subcommand."""
    repo_root = Path(args.repo_root).resolve()
    try:
        configs = discover_configs(repo_root)
    except (InputError, OSError) as exc:
        return fail(str(exc))
    print(
        json.dumps(
            {
                "repo_root": str(repo_root),
                "total": len(configs),
                "counts": carrier_counts(configs),
                "configs": [
                    {"path": info.rel, "line": info.pin.line, "value": info.pin.value, "carrier": info.carrier}
                    for info in configs
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return EXIT_OK


def run_plan(args: argparse.Namespace) -> int:
    """Entry point of the ``plan`` subcommand."""
    repo_root, dest_root = Path(args.repo_root).resolve(), Path(args.dest)
    try:
        configs = discover_configs(repo_root)
    except (InputError, OSError) as exc:
        return fail(str(exc))
    if args.only:
        wanted = Path(args.only).as_posix().removeprefix("./")
        configs = [info for info in configs if info.rel == wanted or info.rel.endswith("/" + wanted)]
    if not configs:
        return fail("no pinned GLM config matched the request")
    case_tree, target_root = (repo_root / CASE_TREE).resolve(), dest_root.resolve()
    if not args.print_only and (target_root == case_tree or case_tree in target_root.parents):
        return fail(f"--dest {target_root} is inside the repository case tree {case_tree}; copies must stay outside it")
    if not args.print_only and target_root.exists() and not target_root.is_dir():
        return fail(f"--dest {target_root} exists and is not a directory")
    prepared, failures = [], []
    for info in configs:
        try:
            text = info.path.read_text(encoding="utf-8")
            prepared.append((info, text, extract_commands(text)))
        except (InputError, OSError) as exc:
            failures.append(f"{info.rel}: {exc}")
    if failures:
        for failure in failures:
            print(f"error: {failure}", file=sys.stderr)
        return fail(f"{len(failures)} config(s) have no usable serve command; nothing was written")
    payload: dict[str, Any] = {
        "repo_root": str(repo_root),
        "dest": None if args.print_only else str(dest_root),
        "print_only": bool(args.print_only),
        "counts": carrier_counts(configs),
        "configs": [],
    }
    for info, text, commands in prepared:
        entry: dict[str, Any] = {
            "path": info.rel,
            "carrier": info.carrier,
            "pin_line": info.pin.line,
            "current_value": info.pin.value,
            "targets": {},
        }
        try:
            entry["targets"] = {
                target: plan_target(info, text, commands, target, dest_root, args.print_only) for target in BACKENDS
            }
        except (InputError, OSError) as exc:
            return fail(f"{info.rel}: {exc}")
        payload["configs"].append(entry)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return EXIT_OK


def run_rewrite(args: argparse.Namespace) -> int:
    """Entry point of the ``rewrite`` subcommand."""
    source = Path(args.src)
    destination = Path(args.dest) / source.name
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        return fail(f"cannot read {source}: {exc}")
    if destination.resolve() == source.resolve():
        return fail(f"refusing to overwrite the source file {source}; pass a different --dest directory")
    try:
        rewritten, changed = rewrite_pin_value(text, args.value)
        if not changed:
            raise InputError(f"{source} already pins {BACKEND_ENV_VAR}={args.value!r}, so nothing would change")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rewritten, encoding="utf-8")
        verified = verify_rewritten(text, destination.read_text(encoding="utf-8"), args.value, True)
    except (InputError, OSError) as exc:
        return fail(str(exc))
    print(f"rewrote line(s) {verified} of {source.name}: {BACKEND_ENV_VAR}={args.value}")
    print(f"copy: {destination}")
    return EXIT_OK


def scan_runner_log(lines: list[str]) -> list[dict[str, Any]]:
    """Classify every runner line of a server log.

    ``explicit`` is the only mode that proves the env override reached the
    server. ``unset`` means the variable was never set and V2 is the default;
    ``fallback`` means a blacklist fell back to V1. Both mean the yaml pin did not
    take effect, so they are failures even when the effective runner matches.
    """
    found: list[dict[str, Any]] = []
    for number, line in enumerate(lines, start=1):
        for match in EXPLICIT_PATTERN.finditer(line):
            found.append(
                {
                    "line": number,
                    "mode": "explicit",
                    "tag": match.group("tag"),
                    "value": match.group("value"),
                    "text": line.strip(),
                }
            )
        if UNSET_PATTERN.search(line) is not None:
            found.append({"line": number, "mode": "unset", "tag": "V2", "value": None, "text": line.strip()})
        if FALLBACK_PATTERN.search(line) is not None:
            found.append({"line": number, "mode": "fallback", "tag": "V1", "value": None, "text": line.strip()})
    return found


def runner_value_agrees(value: str, tag: str) -> bool:
    """Whether the env value printed on an explicit line agrees with the tag on that line."""
    normalized = value.strip().strip("'\"").lower()
    if normalized in TRUTHY_RUNNER_VALUES:
        return tag == "V2"
    if normalized in FALSY_RUNNER_VALUES:
        return tag == "V1"
    return True


def run_verify_log(args: argparse.Namespace) -> int:
    """Entry point of the ``verify-log`` subcommand."""
    log_path = Path(args.log)
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        return fail(f"cannot read {log_path}: {exc}")
    found = scan_runner_log(lines)
    for item in found:
        print(f"{log_path}:{item['line']}: [{item['mode']}] {item['text']}")
    if not found:
        return fail(
            f"no 'is set; using Model Runner V1|V2' line in {log_path}; "
            "a log without the explicit runner line is not a pass"
        )
    blocked = sorted({item["mode"] for item in found if item["mode"] in {"unset", "fallback"}})
    if blocked:
        return fail(
            f"{log_path} carries a {', '.join(blocked)} line: the yaml/env override was NOT applied to the server, "
            "so the effective runner cannot prove anything here",
            EXIT_MISMATCH,
        )
    explicit = [item for item in found if item["mode"] == "explicit"]
    expected = LOG_TAGS[args.expect]
    tags = {item["tag"] for item in explicit}
    if len(tags) > 1:
        return fail(
            f"the log carries contradictory explicit runner lines {sorted(tags)}; cannot prove {args.expect}",
            EXIT_MISMATCH,
        )
    if tags != {expected}:
        return fail(f"the log explicitly reports {sorted(tags)}, expected {expected}", EXIT_MISMATCH)
    inconsistent = [item["line"] for item in explicit if not runner_value_agrees(item["value"], expected)]
    if inconsistent:
        return fail(
            f"explicit line(s) {inconsistent} print a value that contradicts Model Runner {expected}",
            EXIT_MISMATCH,
        )
    print(f"runner log matches --expect {args.expect} ({len(explicit)} explicit line(s))")
    return EXIT_OK


def read_release_tag(repo_root: Path) -> str | None:
    """The vLLM release tag the repository is aligned with, or None when unreadable."""
    try:
        tag = (repo_root / RELEASE_TAG_FILE).read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return tag or None


def git_head(repo_root: Path) -> str | None:
    """HEAD commit of ``repo_root``, or None when git cannot answer."""
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=30, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return (completed.stdout.strip() or None) if completed.returncode == 0 else None


def utc_now() -> str:
    """Current UTC time as an ISO 8601 string with a trailing Z."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    """Assemble the manifest skeleton; anything unreadable stays None."""
    manifest: dict[str, Any] = {field: getattr(args, field, None) for field in MANIFEST_FIELDS}
    repo_root = Path(args.repo_root).resolve()
    manifest["repo_commit"] = manifest["repo_commit"] or git_head(repo_root)
    manifest["vllm_release_tag"] = read_release_tag(repo_root)
    manifest["generated_at"] = manifest["generated_at"] or utc_now()
    manifest["commands"] = None
    if args.server_log:
        log_path = Path(args.server_log)
        if not log_path.is_file():
            raise InputError(f"--server-log {log_path} does not exist")
        manifest["server_log_sha256"] = hashlib.sha256(log_path.read_bytes()).hexdigest()
    if args.commands:
        try:
            commands = json.loads(Path(args.commands).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise InputError(f"cannot read --commands {args.commands}: {exc}") from exc
        if not isinstance(commands, list) or not all(isinstance(item, str) for item in commands):
            raise InputError("--commands must point at a JSON array of command strings")
        manifest["commands"] = commands
    return manifest


def run_manifest(args: argparse.Namespace) -> int:
    """Entry point of the ``manifest`` subcommand."""
    out_path = Path(args.out)
    if out_path.suffix.lower() != ".json":
        return fail(f"--out must end with .json, got {out_path.name!r}")
    try:
        manifest = build_manifest(args)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    except (InputError, OSError) as exc:
        return fail(str(exc))
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"manifest: {out_path}")
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    """Command line definition shared by ``main`` and the unit tests."""
    parser = argparse.ArgumentParser(
        prog="mrv_dual_run",
        description="Plan MRV1/MRV2 dual runs for the pinned GLM configs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EXAMPLES,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    discover = subparsers.add_parser("discover", help="list the GLM configs that pin the runner variable")
    discover.add_argument("--repo-root", default=str(DEFAULT_REPO_ROOT), help="repository root to scan")
    discover.set_defaults(handler=run_discover)
    plan = subparsers.add_parser("plan", help="copy each config twice, one pinned per backend, and print the commands")
    plan.add_argument("--dest", required=True, help="directory that receives <dest>/<backend>/<config> copies")
    plan.add_argument("--repo-root", default=str(DEFAULT_REPO_ROOT), help="repository root to scan")
    plan.add_argument("--only", default=None, help="restrict the plan to one config path")
    plan.add_argument("--print-only", action="store_true", help="print the plan instead of writing copies")
    plan.set_defaults(handler=run_plan)
    rewrite = subparsers.add_parser("rewrite", help="flip the pin of one config into a verified copy")
    rewrite.add_argument("--src", required=True, help="config to read")
    rewrite.add_argument("--dest", required=True, help="directory that receives the rewritten copy")
    rewrite.add_argument(
        "--value", choices=tuple(BACKEND_VALUES.values()), required=True, help="value the pin must carry"
    )
    rewrite.set_defaults(handler=run_rewrite)
    verify_log = subparsers.add_parser("verify-log", help="prove the runner from a server log")
    verify_log.add_argument("--log", required=True, help="server log to scan")
    verify_log.add_argument("--expect", choices=BACKENDS, required=True, help="runner the log must prove")
    verify_log.set_defaults(handler=run_verify_log)
    manifest = subparsers.add_parser("manifest", help="write the run manifest skeleton")
    manifest.add_argument("--out", required=True, help="manifest path, must end with .json")
    manifest.add_argument(
        "--repo-root", default=str(DEFAULT_REPO_ROOT), help="repository root for the tag and the commit"
    )
    for field in MANIFEST_FIELDS:
        manifest.add_argument("--" + field.replace("_", "-"), default=None, help=f"{field}; null when unknown")
    manifest.add_argument("--server-log", default=None, help="server log whose SHA256 is recorded")
    manifest.set_defaults(handler=run_manifest)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse ``argv`` and dispatch to the selected subcommand."""
    args = build_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
