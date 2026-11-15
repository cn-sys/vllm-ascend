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
"""Execute the MRV1/MRV2 parity run: serve, gate, collect, compare, manifest.

Every step guards the next one:

1. ``mrv_dual_run.py plan`` writes the two rewritten config trees; the serve
   command of the matching backend is rendered (never split: JSON arguments and
   `${VAR}` placeholders survive verbatim) and launched through ``sh -c``.
2. Readiness polls ``/health`` until the timeout; a dead, silent or port-owning
   server is aborted and killed before anything else runs.
3. The log gate calls ``mrv_dual_run.py verify-log``. Only an explicit
   ``... is set; using Model Runner V1|V2`` line passes; unset/fallback/
   contradictory logs abort and **no collect file is written**, so a failed run
   cannot leave half data that later looks complete.
4. ``mrv1_mrv2_compare.py collect`` replays the prompt set with the client env
   matching the server, then the server is stopped (SIGTERM, then kill) and the
   log hash is recorded.
5. After both backends, ``mrv1_mrv2_compare.py compare`` writes compare.md and
   compare.json. A comparison mismatch is a result (exit 1), not an
   orchestration failure.

``rehearse --stub`` runs exactly the same orchestration with an injectable serve
command and health URL: the bundled ``_stub-server`` answers ``/health`` and
``POST /v1/completions`` deterministically and prints a real runner log line, so
the rehearsal exercises the same ``Popen``, gate, collect and compare calls a real
NPU run would.

Only the standard library plus ``regex`` (the repository lint whitelist) is
imported. No command is ever handed to a shell keyword: the serve command is
rendered as one string and passed to ``Popen(["sh", "-c", rendered])``, and no
display string is split into tokens. Exit codes: 0 parity holds, 1 compare
reports a mismatch, 2 input, placeholder, readiness, gate or collect failure.
"""

# ruff: noqa: E402

from __future__ import annotations

import os
import sys

# Running 'python tools/mrv_parity_run.py' puts tools/ first on sys.path, where
# the tools/bisect subpackage shadows the standard library bisect module that
# urllib.request -> tempfile -> random need. Drop that entry so the standard
# library always wins, exactly like tools/mrv1_mrv2_compare.py does.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if sys.path and os.path.abspath(sys.path[0] or ".") == _SCRIPT_DIR:
    sys.path.pop(0)

import argparse
import atexit
import hashlib
import json
import shlex
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import regex as re

EXIT_OK, EXIT_MISMATCH, EXIT_INPUT_ERROR = 0, 1, 2
RUNNER_ENV = "VLLM_USE_V2_MODEL_RUNNER"
BACKENDS = ("mrv1", "mrv2")
BACKEND_VALUES = {"mrv1": "0", "mrv2": "1"}
LOG_TAGS = {"mrv1": "V1", "mrv2": "V2"}
MANIFEST_FIELDS = (
    "repo_commit", "vllm_release_tag", "image_tag", "cann_version", "torch_npu_version", "model_id",
    "model_revision", "hardware", "graph_mode", "config_yaml", "backend_expected", "server_log_sha256",
    "commands", "generated_at",
)  # fmt: skip
PLACEHOLDER_PATTERN = re.compile(r"\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)\}")
SHELL_VAR_PATTERN = re.compile(r"\$(?P<name>[A-Za-z_][A-Za-z0-9_]*)")
PD_MARKERS = (
    "kv-transfer-config",
    "kv_transfer_config",
    "prefill-decode",
    "prefill_decode",
    "env_prefill",
    "env_decode",
)
SERVER_LOG_SHA256_NOTE = "null by design: one run has two server logs, so read backends.<backend>.log_sha256 instead"
SECTION_PATTERN = re.compile(r"^(?P<key>[A-Za-z_][A-Za-z0-9_]*)[ \t]*:")
TOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_DIR.parent
DUAL_TOOL = TOOLS_DIR / "mrv_dual_run.py"
COMPARE_TOOL = TOOLS_DIR / "mrv1_mrv2_compare.py"
DEFAULT_REQUESTS = TOOLS_DIR / "mrv1_mrv2_prompts.jsonl"
RELEASE_TAG_FILE = ".github/vllm-release-tag.commit"
DEFAULT_PORT_BASE = 8000
DEFAULT_READINESS_TIMEOUT = 900.0
DEFAULT_COLLECT_TIMEOUT = 3600.0
DEFAULT_STOP_GRACE = 10.0
DEFAULT_HEALTH_URL = "http://127.0.0.1:${PORT}/health"
DEFAULT_BASE_URL = "http://127.0.0.1:${PORT}"
DEFAULT_SERVE_COMMAND = "{python} {tool} _stub-server --port ${PORT} --backend ${BACKEND} --log-mode ${STUB_LOG_MODE}"
STUB_LOG_MODES = ("explicit", "unset", "fallback", "both")
STUB_PROMPT_TOKENS = ("stub", "token", "one", "two")
STUB_PROMPT_LOGPROBS = (-0.1, -0.2, -0.3, -0.4)
HEALTH_POLL_SECONDS = 0.2
RESERVED_VARIABLES = (RUNNER_ENV, "SERVER_PORT", "PORT", "BACKEND", "CONFIG")
LIVE_PROCESSES: set[Any] = set()
EXAMPLES = """examples:
  python tools/mrv_parity_run.py rehearse --stub --data-dir /tmp/t4rehearse
  python tools/mrv_parity_run.py run --configs GLM-5.2-W8A8C8-MTP-A5-dual-nodes --dest /tmp/t4dest \
      --data-dir /tmp/t4data --dry-run
  python tools/mrv_parity_run.py run --configs <config> --dest /tmp/t4dest --data-dir /tmp/t4data
"""


class InputError(ValueError):
    """The request, the plan or the rendered command cannot be used."""


class Aborted(RuntimeError):
    """A run step failed and the run must stop without collecting any data."""


def fail(message: str, code: int = EXIT_INPUT_ERROR) -> int:
    """Print one error on stderr and return the exit code that carries it."""
    print(f"error: {message}", file=sys.stderr)
    return code


def cleanup_live_processes() -> None:
    """Kill every server this process still owns; registered with atexit."""
    for process in list(LIVE_PROCESSES):
        try:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            pass
        LIVE_PROCESSES.discard(process)


atexit.register(cleanup_live_processes)


@dataclass(frozen=True)
class CommandResult:
    """The parts of a finished child command this tool reasons about."""

    returncode: int
    stdout: str = ""
    stderr: str = ""


@dataclass
class Launch:
    """A running server process and the log file it writes to."""

    process: Any
    log_path: Path


@dataclass(frozen=True)
class Options:
    """Everything the orchestration needs that the CLI can change."""

    dest: Path
    requests: Path
    health_url: str
    base_url: str
    port_base: int
    readiness_timeout: float
    collect_timeout: float
    stop_grace: float
    variables: dict[str, str]
    dry_run: bool
    allow_single_command: bool


@dataclass
class ConfigPlan:
    """One config to run: its rewritten copies, its model and its serve commands."""

    slug: str
    index: int
    data_dir: Path
    config_yaml: str | None
    model: str
    commands: dict[str, str]
    copies: dict[str, Path]
    env: dict[str, str] = field(default_factory=dict)
    command_count: int = 1
    command_counts: dict[str, int] = field(default_factory=dict)
    carrier: str | None = None
    num_nodes: int | None = None
    is_pd: bool = False
    single_command_override: bool = False
    blocked_reason: str | None = None


@dataclass
class BackendRun:
    """What one backend produced; a step that never ran leaves its values None."""

    backend: str
    port: int = 0
    log_path: Path | None = None
    collect_path: Path | None = None
    rendered: str | None = None
    readiness_s: float | None = None
    log_sha256: str | None = None
    exit_status: int | None = None
    collect_exit: int | None = None
    collect_records: int | None = None
    gate_stdout: str | None = None
    env_overrides: dict[str, str] = field(default_factory=dict)
    collect_argv: list[str] = field(default_factory=list)
    ignored_commands: int = 0
    missing_placeholders: list[dict[str, Any]] = field(default_factory=list)
    unresolved_variables: list[dict[str, Any]] = field(default_factory=list)
    resolved_from_environ: list[str] = field(default_factory=list)


class Executor:
    """Runs child commands and servers; the unit tests substitute this class."""

    def run(
        self, argv: list[str], cwd: Path, env: dict[str, str] | None = None, timeout: float | None = None
    ) -> CommandResult:
        """Run one CLI command without a shell and capture its output."""
        completed = subprocess.run(
            argv, cwd=str(cwd), env=env, capture_output=True, text=True, timeout=timeout, check=False
        )
        return CommandResult(completed.returncode, completed.stdout, completed.stderr)

    def launch(self, rendered: str, cwd: Path, env: dict[str, str], log_path: Path, port: int) -> Launch:
        """Start one server through ``sh -c`` with its log redirected to a file."""
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as handle:
            process = subprocess.Popen(
                ["sh", "-c", rendered],
                cwd=str(cwd),
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=os.name != "nt",
            )
        LIVE_PROCESSES.add(process)
        return Launch(process, log_path)

    def stop(self, launch: Launch, grace: float) -> int | None:
        """Stop a server: SIGTERM (or taskkill /T on Windows), then kill, then reap."""
        process = launch.process
        if process.poll() is None:
            if os.name == "nt":
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)], capture_output=True, check=False)
            else:
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                except OSError:
                    process.terminate()
            try:
                process.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                if os.name == "nt":
                    process.kill()
                else:
                    try:
                        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    except OSError:
                        process.kill()
                process.wait(timeout=grace)
        LIVE_PROCESSES.discard(process)
        return process.returncode


def indent_of(line: str) -> int:
    """Leading whitespace count of one line."""
    return len(line) - len(line.lstrip())


def scan_scalar(raw: str) -> str | None:
    """Read a one line yaml scalar, dropping quotes and a trailing comment."""
    text = raw.strip()
    if not text or text.startswith("#"):
        return None
    if len(text) >= 2 and text[0] in "'\"" and text[-1] == text[0]:
        return text[1:-1]
    return text.split(" #", 1)[0].strip() or None


def top_section(lines: list[str], key: str) -> tuple[int, int] | None:
    """Line range of one top level yaml key."""
    for index, line in enumerate(lines):
        match = SECTION_PATTERN.match(line)
        if match is None or match.group("key") != key:
            continue
        for later in range(index + 1, len(lines)):
            if SECTION_PATTERN.match(lines[later]) is not None:
                return index, later
        return index, len(lines)
    return None


def anchor_index(lines: list[str], name: str) -> int | None:
    """Line index where ``&name`` is defined."""
    pattern = re.compile(r"&[ \t]*" + re.escape(name) + r"\b")
    return next((index for index, line in enumerate(lines) if indent_of(line) == 0 and pattern.search(line)), None)


def mapping_entries(lines: list[str], key_index: int, key_indent: int, depth: int = 0) -> dict[str, str]:
    """Env entries of one yaml mapping, resolving ``<<: *anchor`` merges."""
    if depth > 5:
        raise InputError("yaml anchor chain is too deep or recursive")
    entries: dict[str, str] = {}
    for index in range(key_index + 1, len(lines)):
        line = lines[index]
        body = line.strip()
        if not body or body.startswith("#"):
            continue
        if indent_of(line) <= key_indent:
            break
        if body.startswith("<<:"):
            name = body.split("*")[-1].strip()
            anchor = anchor_index(lines, name)
            if anchor is None:
                raise InputError(f"yaml anchor &{name} is not defined")
            entries.update(mapping_entries(lines, anchor, indent_of(lines[anchor]), depth + 1))
            continue
        key, separator, raw = body.partition(":")
        value = scan_scalar(raw) if separator else None
        if value is not None:
            entries[key.strip()] = value
    return entries


def config_env(text: str) -> dict[str, str]:
    """The env mapping the case runner would export for this config's first case/node."""
    lines = text.splitlines()
    for section_key in ("templates", "deployment", "test_cases"):
        section = top_section(lines, section_key)
        if section is None:
            continue
        start, end = section
        item = next((index for index in range(start + 1, end) if lines[index].lstrip().startswith("-")), None)
        if item is None:
            continue
        pattern = re.compile(r"^(?P<lead>[ \t]*(?:-[ \t]*)?)envs[ \t]*:(?P<rest>.*)$")
        found = None
        for index in range(item, end):
            match = pattern.match(lines[index])
            if match is not None:
                found = (index, len(match.group("lead")), match.group("rest"))
                break
        if found is None:
            continue
        key_index, key_indent, rest = found
        raw = rest.strip()
        if raw.startswith("*"):
            name = raw.split("*")[-1].strip()
            anchor = anchor_index(lines, name)
            if anchor is None:
                raise InputError(f"yaml anchor &{name} is not defined")
            return mapping_entries(lines, anchor, indent_of(lines[anchor]))
        return mapping_entries(lines, key_index, key_indent)
    return {}


def top_scalar(text: str, key: str) -> str | None:
    """Value of one top level yaml scalar."""
    match = re.search(rf"(?m)^{re.escape(key)}:[ \t]*(?P<rest>[^\n#]*)", text)
    return scan_scalar(match.group("rest")) if match is not None else None


def config_num_nodes(text: str) -> int | None:
    """The num_nodes header of a config, None when the config is single node."""
    raw = top_scalar(text, "num_nodes")
    try:
        return int(raw) if raw is not None else None
    except ValueError:
        return None


def is_pd_config(rel: str, text: str) -> bool:
    """Whether a config describes a prefill/decode disaggregated deployment."""
    if "-PD" in Path(rel).stem.upper() or "-PD." in rel.upper():
        return True
    lowered = text.lower()
    return any(marker in lowered for marker in PD_MARKERS)


def single_command_problem(
    rel: str, count: int, carrier: str | None, num_nodes: int | None, pd: bool, allow: bool
) -> str | None:
    """Refuse to run only the first of several node commands, and say why."""
    if count <= 1:
        return None
    if not allow:
        return (
            f"{rel}: {count} serve commands (one per node); running only the first would drop {count - 1} of them. "
            "Pass --allow-single-command only for an otherwise single-node config"
        )
    if carrier == "external_templates":
        return (
            f"{rel}: external_templates configs need every node command; this executor cannot run them, "
            "not even with --allow-single-command"
        )
    if num_nodes is not None and num_nodes > 1:
        return f"{rel}: num_nodes={num_nodes}; multi-node launch is out of scope, not even with --allow-single-command"
    if pd:
        return (
            f"{rel}: PD (prefill/decode disaggregated) configs are out of scope, not even with --allow-single-command"
        )
    return None


def format_variable_problems(plan: ConfigPlan, backend: str, problems: list[dict[str, Any]]) -> str:
    """One message that names every missing placeholder and unresolved shell variable."""
    placeholders = [item for item in problems if item["kind"] == "placeholder"]
    variables = [item for item in problems if item["kind"] == "variable"]
    parts: list[str] = []
    if placeholders:
        parts.append(f"unresolved placeholder(s) {json.dumps(placeholders)}; pass --var NAME=VALUE")
    if variables:
        parts.append(f"unresolved variable(s) {json.dumps(variables)}; pass --var NAME=VALUE or export them")
    return f"{plan.config_yaml or plan.slug} ({backend}): " + "; ".join(parts)


def base_variables(port: int, backend: str) -> dict[str, str]:
    """Variables every serve command, health URL and base URL may use."""
    value = BACKEND_VALUES[backend]
    return {"PORT": str(port), "SERVER_PORT": str(port), "BACKEND": backend, RUNNER_ENV: value}


def port_for(port_base: int, config_index: int, backend: str) -> int:
    """Port for one config/backend pair; the two backends of a config never collide."""
    return port_base + config_index * len(BACKENDS) + BACKENDS.index(backend)


def port_is_open(port: int) -> bool:
    """Whether something already listens on the loopback port."""
    with socket.socket() as sock:
        sock.settimeout(0.2)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def sha256_file(path: Path) -> str:
    """SHA256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tail_of(path: Path, limit: int = 20) -> str:
    """Last lines of a log, for an abort message."""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-limit:])


def wait_for_health(url: str, process: Any, timeout: float) -> float:
    """Poll ``url`` until it answers; abort on timeout or on a server that already exited."""
    started = time.monotonic()
    deadline = started + timeout
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise Aborted(f"server exited with {process.returncode} before it became ready")
        try:
            with urllib.request.urlopen(url, timeout=2):
                return round(time.monotonic() - started, 3)
        except (urllib.error.URLError, OSError):
            time.sleep(HEALTH_POLL_SECONDS)
    raise Aborted(f"server was not ready within {timeout}s ({url})")


def git_head() -> str | None:
    """HEAD commit of the repository, or None when git cannot answer."""
    try:
        completed = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=30, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return (completed.stdout.strip() or None) if completed.returncode == 0 else None


def read_release_tag() -> str | None:
    """The aligned vLLM release tag, or None when unreadable."""
    try:
        tag = (REPO_ROOT / RELEASE_TAG_FILE).read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return tag or None


def utc_now() -> str:
    """Current UTC time as an ISO 8601 string ending in Z."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def record_command(
    records: list[dict[str, Any]],
    stage: str,
    backend: str | None,
    argv: list[str],
    cwd: Path,
    env_overrides: dict[str, str],
    result: CommandResult,
) -> None:
    """Append one child command to the manifest command log."""
    records.append(
        {
            "stage": stage,
            "backend": backend,
            "argv": list(argv),
            "cwd": str(cwd),
            "env": dict(env_overrides),
            "returncode": result.returncode,
        }
    )


class VariableResolver:
    """Expand ``${VAR}`` and bare ``$VAR`` with --var > config env > os.environ precedence.

    Env values reach the child process as a mapping, so no shell expands them:
    this class is the only place that turns ``$PYTHON_LIB_DIR`` into a path.
    A variable that references itself (``$LD_LIBRARY_PATH`` inside
    ``LD_LIBRARY_PATH``) takes the previous ``os.environ`` value instead of
    recursing, so prepending to a path cannot loop.
    """

    def __init__(self, variables: dict[str, str], config_env: dict[str, str], environ: Any) -> None:
        self.variables = dict(variables)
        self.raw = dict(config_env)
        self.environ = environ
        self.rendered: dict[str, str] = {}
        self.problems: dict[str, list[tuple[str, str]]] = {}
        self.resolved_from_environ: set[str] = set()
        self.busy: set[str] = set()

    def value_of(self, name: str, owner: str | None) -> str | None:
        """Resolve one variable name; ``owner`` enables the self-reference rule."""
        if name in self.variables:
            return self.variables[name]
        if name == owner:
            value = self.environ.get(name)
            if value is not None:
                self.resolved_from_environ.add(name)
            return value
        if name in self.rendered:
            return self.rendered[name]
        if name in self.raw:
            return self.render_env(name)
        value = self.environ.get(name)
        if value is not None:
            self.resolved_from_environ.add(name)
        return value

    def render_env(self, key: str) -> str | None:
        """Render one config env value, expanding the variables it references."""
        if key in self.rendered:
            return self.rendered[key]
        if key in self.busy or key not in self.raw:
            return None
        self.busy.add(key)
        text, missing = self.expand(self.raw[key], owner=key)
        self.busy.discard(key)
        self.problems[key] = missing
        if missing:
            return None
        self.rendered[key] = text
        return text

    def expand(
        self, text: str, owner: str | None = None, substitute_bare: bool = True
    ) -> tuple[str, list[tuple[str, str]]]:
        """Expand placeholders and optionally bare variables; report whatever is left."""
        missing: list[tuple[str, str]] = []

        def placeholder(match: Any) -> str:
            value = self.value_of(match.group("name"), owner)
            if value is None:
                missing.append((match.group("name"), "placeholder"))
                return match.group(0)
            return value

        expanded = PLACEHOLDER_PATTERN.sub(placeholder, text)

        def bare(match: Any) -> str:
            value = self.value_of(match.group("name"), owner)
            if value is None:
                missing.append((match.group("name"), "variable"))
                return match.group(0)
            return value if substitute_bare else match.group(0)

        expanded = SHELL_VAR_PATTERN.sub(bare, expanded)
        return expanded, missing

    def render_all_env(self) -> tuple[dict[str, str], dict[str, list[tuple[str, str]]]]:
        """Render every config env value; return the values plus the per-key problems."""
        for key in self.raw:
            self.render_env(key)
        return dict(self.rendered), {key: value for key, value in self.problems.items() if value}


def collect_command(plan: ConfigPlan, options: Options, run: BackendRun, base_url: str) -> list[str]:
    """The ``mrv1_mrv2_compare.py collect`` argv for one backend."""
    return [
        sys.executable,
        str(COMPARE_TOOL),
        "collect",
        "--backend",
        run.backend,
        "--base-url",
        base_url,
        "--model",
        plan.model,
        "--requests",
        str(options.requests),
        "--out",
        str(run.collect_path),
    ]


def execute_backend(
    plan: ConfigPlan, run: BackendRun, options: Options, executor: Executor, records: list[dict[str, Any]]
) -> None:
    """Launch, gate, collect and stop one backend; mutate ``run`` with what happened."""
    run.port = port_for(options.port_base, plan.index, run.backend)
    if plan.single_command_override:
        run.ignored_commands = max(0, plan.command_counts.get(run.backend, plan.command_count) - 1)
    variables = dict(options.variables)
    variables.update(base_variables(run.port, run.backend))
    variables["CONFIG"] = str(plan.copies[run.backend])
    resolver = VariableResolver(variables, plan.env, os.environ)
    problems: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    def record(entries: list[tuple[str, str]], location: str) -> None:
        for name, kind in entries:
            key = (name, location, kind)
            if key in seen:
                continue
            seen.add(key)
            problems.append(
                {
                    "config": plan.config_yaml or plan.slug,
                    "backend": run.backend,
                    "variable": name,
                    "location": location,
                    "kind": kind,
                }
            )

    rendered, missing = resolver.expand(plan.commands[run.backend], substitute_bare=False)
    run.rendered = rendered
    record(missing, "command")
    health_url, missing = resolver.expand(options.health_url, substitute_bare=False)
    record(missing, "health")
    base_url, missing = resolver.expand(options.base_url, substitute_bare=False)
    record(missing, "base")
    rendered_env, env_problems = resolver.render_all_env()
    for name, entries in env_problems.items():
        record(entries, f"env:{name}")
    overrides = {name: rendered_env.get(name, plan.env[name]) for name in plan.env}
    overrides[RUNNER_ENV] = BACKEND_VALUES[run.backend]
    overrides["SERVER_PORT"] = str(run.port)
    run.env_overrides = overrides
    run.resolved_from_environ = sorted(resolver.resolved_from_environ)
    run.log_path = plan.data_dir / run.backend / "server.log"
    run.collect_path = plan.data_dir / run.backend / "collect.jsonl"
    run.collect_argv = collect_command(plan, options, run, base_url)
    run.missing_placeholders = [item for item in problems if item["kind"] == "placeholder"]
    run.unresolved_variables = [item for item in problems if item["kind"] == "variable"]
    if problems and not options.dry_run:
        raise InputError(format_variable_problems(plan, run.backend, problems))
    if options.dry_run:
        return
    if port_is_open(run.port):
        raise Aborted(f"port {run.port} is already in use before the {run.backend} server starts")
    env = {**os.environ, **overrides, **options.variables}
    launch: Launch | None = None
    log_path = run.log_path
    try:
        launch = executor.launch(rendered, REPO_ROOT, env, log_path, run.port)
        run.readiness_s = wait_for_health(health_url, launch.process, options.readiness_timeout)
        gate_argv = [sys.executable, str(DUAL_TOOL), "verify-log", "--log", str(log_path), "--expect", run.backend]
        gate = executor.run(gate_argv, REPO_ROOT, env, 120)
        record_command(records, "verify-log", run.backend, gate_argv, REPO_ROOT, overrides, gate)
        run.gate_stdout = gate.stdout.strip() or None
        if gate.returncode != 0:
            detail = (gate.stderr or gate.stdout).strip() or tail_of(log_path)
            raise Aborted(f"{run.backend} log gate failed with {gate.returncode}: {detail}")
        collected = executor.run(run.collect_argv, REPO_ROOT, env, options.collect_timeout)
        record_command(records, "collect", run.backend, run.collect_argv, REPO_ROOT, overrides, collected)
        run.collect_exit = collected.returncode
        if collected.returncode != 0:
            raise Aborted(f"{run.backend} collect failed with {collected.returncode}: {collected.stderr.strip()}")
        run.collect_records = count_records(run.collect_path)
    finally:
        if launch is not None:
            run.exit_status = executor.stop(launch, options.stop_grace)
        if log_path.is_file():
            run.log_sha256 = sha256_file(log_path)


def count_records(path: Path | None) -> int | None:
    """Number of JSONL records in a collect output, None when unreadable."""
    if path is None or not path.is_file():
        return None
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def build_manifest(
    plan: ConfigPlan,
    runs: dict[str, BackendRun],
    records: list[dict[str, Any]],
    compare: dict[str, Any],
    abort_reason: str | None,
    options: Options,
) -> dict[str, Any]:
    """The manifest: the 14 plan fields, the per backend facts and the command log."""
    manifest: dict[str, Any] = {field: None for field in MANIFEST_FIELDS}
    manifest["repo_commit"] = git_head()
    manifest["vllm_release_tag"] = read_release_tag()
    manifest["config_yaml"] = plan.config_yaml
    manifest["backend_expected"] = list(BACKENDS)
    manifest["generated_at"] = utc_now()
    manifest["commands"] = records
    manifest["backends"] = {
        backend: {
            "log_sha256": run.log_sha256,
            "exit_status": run.exit_status,
            "readiness_s": run.readiness_s,
            "collect_exit": run.collect_exit,
            "collect_records": run.collect_records,
            "server_log": str(run.log_path) if run.log_path else None,
            "collect": str(run.collect_path) if run.collect_path else None,
            "gate_stdout": run.gate_stdout,
            "ignored_commands": run.ignored_commands,
        }
        for backend, run in runs.items()
    }
    manifest["command_count"] = plan.command_count
    manifest["resolved_from_environ"] = sorted({name for run in runs.values() for name in run.resolved_from_environ})
    manifest["single_command_override"] = plan.single_command_override
    manifest["server_log_sha256_note"] = SERVER_LOG_SHA256_NOTE
    manifest["config_slug"] = plan.slug
    manifest["data_dir"] = str(plan.data_dir)
    manifest["dest"] = str(options.dest)
    manifest["abort_reason"] = abort_reason
    manifest["compare_exit"] = compare.get("exit")
    manifest["compare_md"] = compare.get("md")
    manifest["compare_json"] = compare.get("json")
    return manifest


def run_compare(
    plan: ConfigPlan, options: Options, executor: Executor, records: list[dict[str, Any]]
) -> dict[str, Any]:
    """Run the N1 compare tool for both report formats."""
    collect_a = plan.data_dir / "mrv1" / "collect.jsonl"
    collect_b = plan.data_dir / "mrv2" / "collect.jsonl"
    if not collect_a.is_file() or not collect_b.is_file():
        raise Aborted("compare requested but a collect file is missing")
    base = [sys.executable, str(COMPARE_TOOL), "compare", "--a", str(collect_a), "--b", str(collect_b)]
    info: dict[str, Any] = {"exit": None, "md": None, "json": None}
    for suffix, path in (("md", plan.data_dir / "compare.md"), ("json", plan.data_dir / "compare.json")):
        argv = [*base, "--out", str(path)]
        result = executor.run(argv, REPO_ROOT, None, None)
        record_command(records, "compare", None, argv, REPO_ROOT, {}, result)
        info[suffix] = str(path)
        if suffix == "md":
            info["exit"] = result.returncode
        if result.returncode not in (EXIT_OK, EXIT_MISMATCH):
            raise Aborted(f"compare failed with {result.returncode}: {result.stderr.strip()}")
    return info


def archive_dry_run(
    plan: ConfigPlan, runs: dict[str, BackendRun], options: Options, records: list[dict[str, Any]]
) -> None:
    """Print the sequence ``run`` would execute; never launch, never write."""
    unresolved = [item for run in runs.values() for item in run.missing_placeholders + run.unresolved_variables]
    payload = {
        "dry_run": True,
        "config": plan.slug,
        "config_yaml": plan.config_yaml,
        "data_dir": str(plan.data_dir),
        "dest": str(options.dest),
        "command_count": plan.command_count,
        "single_command_override": plan.single_command_override,
        "ignored_commands": {run.backend: run.ignored_commands for run in runs.values()},
        "unresolved_variables": unresolved,
        "resolved_from_environ": sorted({name for run in runs.values() for name in run.resolved_from_environ}),
        "plan_argv": [record["argv"] for record in records if record["stage"] == "plan"],
        "backends": [
            {
                "backend": run.backend,
                "port": run.port,
                "serve_command": run.rendered,
                "env_overrides": run.env_overrides,
                "gate_argv": [
                    sys.executable,
                    str(DUAL_TOOL),
                    "verify-log",
                    "--log",
                    str(run.log_path),
                    "--expect",
                    run.backend,
                ],
                "collect_argv": run.collect_argv,
                "server_log": str(run.log_path),
                "collect": str(run.collect_path),
                "ignored_commands": run.ignored_commands,
                "unresolved_variables": run.missing_placeholders + run.unresolved_variables,
                "resolved_from_environ": run.resolved_from_environ,
            }
            for run in runs.values()
        ],
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def execute_config(
    plan: ConfigPlan, options: Options, executor: Executor, records: list[dict[str, Any]] | None = None
) -> int:
    """Run one config end to end; returns the process exit code."""
    runs: dict[str, BackendRun] = {}
    records = [] if records is None else records
    compare: dict[str, Any] = {"exit": None, "md": None, "json": None}
    abort_reason: str | None = None
    try:
        for backend in BACKENDS:
            run = BackendRun(backend=backend)
            runs[backend] = run
            execute_backend(plan, run, options, executor, records)
        if options.dry_run:
            archive_dry_run(plan, runs, options, records)
            unresolved = [item for run in runs.values() for item in run.missing_placeholders + run.unresolved_variables]
            if unresolved:
                return fail(
                    f"{plan.slug}: dry-run found {len(unresolved)} unresolved value(s): {json.dumps(unresolved)}"
                )
            return EXIT_OK
        compare = run_compare(plan, options, executor, records)
    except (Aborted, InputError) as exc:
        abort_reason = str(exc)
    if options.dry_run:
        return fail(abort_reason or "dry-run failed")
    manifest = build_manifest(plan, runs, records, compare, abort_reason, options)
    manifest_path = plan.data_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if abort_reason is not None:
        return fail(f"{plan.slug}: {abort_reason}")
    print(f"manifest: {manifest_path}")
    if compare.get("exit") == EXIT_MISMATCH:
        print(f"compare reports a mismatch: {compare['md']}", file=sys.stderr)
        return EXIT_MISMATCH
    print(f"parity holds for {plan.slug}: {compare['md']}")
    return EXIT_OK


def parse_variables(pairs: list[str]) -> dict[str, str]:
    """Turn ``NAME=VALUE`` arguments into the variable map `${NAME}` renders from."""
    variables: dict[str, str] = {}
    for pair in pairs:
        name, separator, value = pair.partition("=")
        if not separator or not name:
            raise InputError(f"--var expects NAME=VALUE, got {pair!r}")
        if name in RESERVED_VARIABLES:
            raise InputError(f"--var {name} is reserved ({', '.join(RESERVED_VARIABLES)}); the executor owns that name")
        variables[name] = value
    return variables


def plan_argv(options: Options, slugs: list[str] | None = None) -> list[str]:
    """The ``mrv_dual_run.py plan`` invocation the orchestration uses to materialise copies."""
    argv = [sys.executable, str(DUAL_TOOL), "plan", "--dest", str(options.dest), "--repo-root", str(REPO_ROOT)]
    if options.dry_run:
        argv.append("--print-only")
    if slugs and len(slugs) == 1:
        slug = slugs[0]
        argv += ["--only", slug if slug.endswith(".yaml") else slug + ".yaml"]
    return argv


def matches_slug(path: str, slug: str) -> bool:
    """Whether a plan entry belongs to the ``--configs`` slug."""
    return Path(path).stem == slug or path == slug or path.endswith("/" + slug + ".yaml")


def load_plans(
    slugs: list[str], data_dir: Path, options: Options, executor: Executor, records: list[dict[str, Any]]
) -> tuple[list[ConfigPlan], list[str]]:
    """Materialise the copies once and turn the plan JSON into the execution inputs.

    A config with several node commands is not silently reduced to its first
    command: that is recorded as a problem and ``run_real`` refuses to start
    anything unless ``--allow-single-command`` explicitly allows a single-node
    config to run only its first command.
    """
    argv = plan_argv(options, slugs)
    result = executor.run(argv, REPO_ROOT, None, 300)
    record_command(records, "plan", None, argv, REPO_ROOT, {}, result)
    if result.returncode != 0:
        raise InputError(f"mrv_dual_run.py plan failed with {result.returncode}: {result.stderr.strip()}")
    payload = json.loads(result.stdout)
    total = len(slugs)
    plans: list[ConfigPlan] = []
    problems: list[str] = []
    for index, slug in enumerate(slugs):
        entry = next((item for item in payload["configs"] if matches_slug(item["path"], slug)), None)
        if entry is None:
            raise InputError(f"{slug!r} is not one of the pinned GLM configs the planner found")
        commands: dict[str, str] = {}
        copies: dict[str, Path] = {}
        models: set[str] = set()
        command_counts: dict[str, int] = {}
        for backend in BACKENDS:
            candidates = entry["targets"][backend]["commands"]
            command_counts[backend] = len(candidates)
            commands[backend] = candidates[0]["display"]
            models.add(candidates[0]["model"])
            dest = entry["targets"][backend]["dest"]
            copies[backend] = Path(dest) if dest else REPO_ROOT / entry["path"]
        if None in models or len(models) != 1:
            raise InputError(f"{entry['path']}: cannot determine the served model name for collect")
        try:
            config_text = (REPO_ROOT / entry["path"]).read_text(encoding="utf-8")
        except OSError:
            config_text = copies["mrv1"].read_text(encoding="utf-8")
        command_count = max(command_counts.values())
        num_nodes = config_num_nodes(config_text)
        pd_config = is_pd_config(entry["path"], config_text)
        problem = single_command_problem(
            entry["path"], command_count, entry.get("carrier"), num_nodes, pd_config, options.allow_single_command
        )
        override = bool(options.allow_single_command and command_count > 1 and problem is None)
        if override:
            print(
                f"warning: {entry['path']}: only the first of {command_count} serve commands will run "
                f"(--allow-single-command); {command_count - 1} node command(s) ignored",
                file=sys.stderr,
            )
        if problem is not None:
            problems.append(problem)
        plan_data_dir = data_dir if total == 1 else data_dir / slug
        plans.append(
            ConfigPlan(
                slug=slug,
                index=index,
                data_dir=plan_data_dir,
                config_yaml=entry["path"],
                model=models.pop(),
                commands=commands,
                copies=copies,
                env=config_env(copies["mrv1"].read_text(encoding="utf-8")),
                command_count=command_count,
                command_counts=command_counts,
                carrier=entry.get("carrier"),
                num_nodes=num_nodes,
                is_pd=pd_config,
                single_command_override=override,
                blocked_reason=problem,
            )
        )
    return plans, problems


def stub_command() -> str:
    """The bundled stub server as a serve command template."""
    return DEFAULT_SERVE_COMMAND.replace("{python}", shlex.quote(sys.executable)).replace(
        "{tool}", shlex.quote(str(Path(__file__).resolve()))
    )


def rehearse_options(args: argparse.Namespace) -> Options:
    """Options for a rehearsal: identical to a real run except for the serve command."""
    variables = parse_variables(args.var)
    variables["STUB_LOG_MODE"] = args.log_mode
    return Options(
        dest=Path(args.data_dir) / "dest",
        requests=Path(args.requests),
        health_url=args.health_url,
        base_url=args.base_url,
        port_base=args.port_base,
        readiness_timeout=args.readiness_timeout,
        collect_timeout=args.collect_timeout,
        stop_grace=DEFAULT_STOP_GRACE,
        variables=variables,
        dry_run=False,
        allow_single_command=False,
    )


def run_rehearse(args: argparse.Namespace) -> int:
    """Entry point of ``rehearse``: the real orchestration with a stub server."""
    if not args.stub:
        return fail("rehearse currently only supports --stub; pass --stub to confirm the stub server")
    serve_cmd = args.serve_cmd or stub_command()
    try:
        options = rehearse_options(args)
        plan = ConfigPlan(
            slug="rehearse",
            index=0,
            data_dir=Path(args.data_dir),
            config_yaml=None,
            model="stub-model",
            commands={backend: serve_cmd for backend in BACKENDS},
            copies={backend: Path(".") for backend in BACKENDS},
            env={},
        )
    except (InputError, OSError) as exc:
        return fail(str(exc))
    return execute_config(plan, options, Executor())


def run_real(args: argparse.Namespace) -> int:
    """Entry point of ``run``: the real configs, without a shell."""
    slugs = [slug.strip() for slug in args.configs.split(",") if slug.strip()]
    if not slugs:
        return fail("--configs needs at least one config slug")
    try:
        options = Options(
            dest=Path(args.dest),
            requests=Path(args.requests),
            health_url=args.health_url,
            base_url=args.base_url,
            port_base=args.port_base,
            readiness_timeout=args.readiness_timeout,
            collect_timeout=args.collect_timeout,
            stop_grace=DEFAULT_STOP_GRACE,
            variables=parse_variables(args.var),
            dry_run=bool(args.dry_run),
            allow_single_command=bool(args.allow_single_command),
        )
        records: list[dict[str, Any]] = []
        plans, problems = load_plans(slugs, Path(args.data_dir), options, Executor(), records)
    except (InputError, OSError, json.JSONDecodeError) as exc:
        return fail(str(exc))
    if options.dry_run:
        exit_code = EXIT_INPUT_ERROR if problems else EXIT_OK
        for plan in plans:
            if execute_config(plan, options, Executor(), records) == EXIT_INPUT_ERROR:
                exit_code = EXIT_INPUT_ERROR
        for problem in problems:
            print(f"error: {problem}", file=sys.stderr)
        return exit_code
    if problems:
        for problem in problems:
            print(f"error: {problem}", file=sys.stderr)
        return fail(f"{len(problems)} config(s) cannot be run by this executor; nothing was launched")
    for plan in plans:
        plan.data_dir.mkdir(parents=True, exist_ok=True)
    exit_code = EXIT_OK
    for plan in plans:
        code = execute_config(plan, options, Executor(), records)
        exit_code = max(exit_code, code) if code != EXIT_OK else exit_code
        if code == EXIT_INPUT_ERROR:
            return code
    return exit_code


class StubHandler(BaseHTTPRequestHandler):
    """Deterministic stand-in for the OpenAI completions endpoint."""

    def log_message(self, *args: Any) -> None:
        """Keep the stub log to the runner line this tool has to verify."""
        return None

    def send_json(self, status: int, payload: dict[str, Any]) -> None:
        """Write one JSON response."""
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        """Health endpoint used by readiness polling."""
        if self.path.split("?")[0] == "/health":
            self.send_json(200, {"status": "ok"})
        else:
            self.send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        """Completions endpoint used by ``collect``."""
        length = int(self.headers.get("Content-Length", "0"))
        if length:
            self.rfile.read(length)
        if self.path.split("?")[0] != "/v1/completions":
            self.send_json(404, {"error": "not found"})
            return
        self.send_json(
            200,
            {
                "id": "stub",
                "object": "text_completion",
                "model": "stub-model",
                "choices": [
                    {
                        "index": 0,
                        "text": "stub token one two",
                        "logprobs": {
                            "tokens": list(STUB_PROMPT_TOKENS),
                            "token_logprobs": list(STUB_PROMPT_LOGPROBS),
                        },
                        "finish_reason": "length",
                    }
                ],
            },
        )


def runner_log_lines(backend: str, log_mode: str) -> list[str]:
    """The runner line(s) the stub prints, shaped like ``vllm_ascend/mrv2_utils.py``."""
    explicit = f"{RUNNER_ENV}={BACKEND_VALUES[backend]} is set; using Model Runner {LOG_TAGS[backend]}."
    unset = f"{RUNNER_ENV} is unset; using Model Runner V2 by default."
    fallback = "Model Runner V2 does not yet support drafted tokens; using the V1 model runner instead."
    if log_mode == "both":
        return [explicit, unset]
    if log_mode == "unset":
        return [unset]
    if log_mode == "fallback":
        return [fallback]
    return [explicit]


def run_stub_server(args: argparse.Namespace) -> int:
    """Entry point of the hidden ``_stub-server`` used by the rehearsal."""
    for line in runner_log_lines(args.backend, args.log_mode):
        print(line, flush=True)
    handler = type("BoundStubHandler", (StubHandler,), {})
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    print(f"stub server listening on 127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return EXIT_OK
    finally:
        server.server_close()
    return EXIT_OK


def add_common_options(parser: argparse.ArgumentParser) -> None:
    """Options shared by ``run`` and ``rehearse``."""
    parser.add_argument("--port-base", type=int, default=DEFAULT_PORT_BASE, help="first port; backends take +1")
    parser.add_argument("--requests", default=str(DEFAULT_REQUESTS), help="prompt JSONL replayed by collect")
    parser.add_argument("--health-url", default=DEFAULT_HEALTH_URL, help="readiness URL, placeholders allowed")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="server root for collect, placeholders allowed")
    parser.add_argument("--readiness-timeout", type=float, default=DEFAULT_READINESS_TIMEOUT)
    parser.add_argument("--collect-timeout", type=float, default=DEFAULT_COLLECT_TIMEOUT)
    parser.add_argument("--var", action="append", default=[], metavar="NAME=VALUE", help="placeholder value")


def build_parser() -> argparse.ArgumentParser:
    """Command line definition shared by ``main`` and the unit tests."""
    parser = argparse.ArgumentParser(
        prog="mrv_parity_run",
        description="Run the MRV1/MRV2 parity pipeline: serve, gate, collect, compare, manifest.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EXAMPLES,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    rehearse = subparsers.add_parser("rehearse", help="run the whole pipeline against the bundled stub")
    rehearse.add_argument("--stub", action="store_true", help="confirm the stub server instead of a real one")
    rehearse.add_argument("--data-dir", required=True, help="directory that receives the run products")
    rehearse.add_argument("--serve-cmd", default=None, help="serve command template to launch instead of the stub")
    rehearse.add_argument("--log-mode", choices=STUB_LOG_MODES, default="explicit", help="runner line the stub prints")
    add_common_options(rehearse)
    rehearse.set_defaults(handler=run_rehearse)
    run = subparsers.add_parser("run", help="run the pipeline for one or more pinned GLM configs")
    run.add_argument("--configs", required=True, help="comma separated config slugs")
    run.add_argument("--data-dir", required=True, help="directory that receives the run products")
    run.add_argument("--dest", required=True, help="directory that receives the rewritten config copies")
    run.add_argument("--dry-run", action="store_true", help="print the sequence; launch nothing")
    run.add_argument(
        "--allow-single-command",
        action="store_true",
        help="allow a single-node config with several node commands to run only the first one",
    )
    add_common_options(run)
    run.set_defaults(handler=run_real)
    stub = subparsers.add_parser("_stub-server", help="deterministic stub server used by rehearse")
    stub.add_argument("--port", type=int, required=True)
    stub.add_argument("--backend", choices=BACKENDS, default="mrv1")
    stub.add_argument("--log-mode", choices=STUB_LOG_MODES, default="explicit")
    stub.set_defaults(handler=run_stub_server)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse ``argv`` and dispatch to the selected subcommand."""
    args = build_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
