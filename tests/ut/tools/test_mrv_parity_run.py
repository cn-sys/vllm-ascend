import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from dataclasses import replace
from pathlib import Path

import pytest

from tools import mrv_parity_run as parity

REPO_ROOT = Path(__file__).resolve().parents[3]
GLM_DIR = "tests/e2e/cases/models/configs/GLM"
DEPLOYMENT_CONFIG = f"{GLM_DIR}/GLM-5.2-W8A8C8-MTP-A5-dual-nodes.yaml"
PROMPTS = (
    '{"id":"p1","prompt":"hello","max_tokens":4,"temperature":0}\n'
    '{"id":"p2","prompt":"world","max_tokens":4,"temperature":0}\n'
)
ENV_FIXTURE = """test_name: "fake"
model: "org/model"
env_common: &env_common
  VLLM_USE_V2_MODEL_RUNNER: "1"
  SERVER_PORT: "8080"
  HCCL_BUFFSIZE: "1024"
deployment:
  - envs:
      <<: *env_common
    server_cmd: >
      vllm serve org/model
      --port $SERVER_PORT
"""
CASE_FIXTURE = """test_cases:
  - name: "case"
    model: "org/model"
    envs:
      VLLM_USE_V2_MODEL_RUNNER: "1"
      SERVER_PORT: "8080"
    server_cmd:
      - "--port"
      - "$SERVER_PORT"
"""
ALIAS_FIXTURE = """test_name: "fake"
_envs: &envs
  VLLM_USE_V2_MODEL_RUNNER: "1"
  SERVER_PORT: "8080"
test_cases:
  - name: "case"
    model: "org/model"
    envs: *envs
    server_cmd:
      - "--port"
      - "$SERVER_PORT"
"""


def write_prompts(tmp_path: Path) -> Path:
    path = tmp_path / "prompts.jsonl"
    path.write_text(PROMPTS, encoding="utf-8")
    return path


def free_port_pair() -> int:
    """Two consecutive free loopback ports, because port_for uses base and base+1."""
    for _ in range(50):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            base = sock.getsockname()[1]
        if not parity.port_is_open(base) and not parity.port_is_open(base + 1):
            return base
    raise RuntimeError("no free consecutive ports")


class FakeProcess:
    """Stand-in for a Popen object."""

    def __init__(self) -> None:
        self.returncode: int | None = None

    def poll(self) -> int | None:
        return self.returncode

    def kill(self) -> None:
        self.returncode = -9


class FakeExecutor(parity.Executor):
    """Records every call and answers with canned results; never spawns anything."""

    def __init__(
        self,
        *,
        gate_rc: int = 0,
        collect_rc: int = 0,
        compare_rc: int = 0,
        log_text: str = "explicit\n",
        compare_writes: bool = True,
        compare_rcs: dict[str, int] | None = None,
        stale_suffix: str | None = None,
    ) -> None:
        self.gate_rc = gate_rc
        self.collect_rc = collect_rc
        self.compare_rc = compare_rc
        self.compare_writes = compare_writes
        self.compare_rcs = compare_rcs or {}
        self.stale_suffix = stale_suffix
        self.log_text = log_text
        self.calls: list[str] = []
        self.launches: list[tuple[str, int]] = []
        self.launch_envs: list[dict[str, str]] = []
        self.stops = 0
        self.plan_payload: dict | None = None

    def stage_of(self, argv: list[str]) -> str:
        if "mrv_dual_run" in Path(argv[1]).name:
            return argv[2] if len(argv) > 2 else "?"
        return argv[2] if len(argv) > 2 else "?"

    def run(self, argv, cwd, env=None, timeout=None) -> parity.CommandResult:
        stage = self.stage_of(argv)
        self.calls.append(stage)
        if stage == "plan":
            return parity.CommandResult(0, json.dumps(self.plan_payload or {"configs": []}), "")
        if stage == "verify-log":
            return parity.CommandResult(self.gate_rc, "log line\n", "" if self.gate_rc == 0 else "gate error")
        if stage == "collect":
            out = Path(argv[argv.index("--out") + 1])
            out.parent.mkdir(parents=True, exist_ok=True)
            if self.collect_rc == 0:
                out.write_text('{"id":"p1"}\n{"id":"p2"}\n', encoding="utf-8")
            return parity.CommandResult(self.collect_rc, "", "" if self.collect_rc == 0 else "collect error")
        if stage == "compare":
            out = Path(argv[argv.index("--out") + 1])
            suffix = out.suffix.lstrip(".")
            if self.compare_writes:
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text("report\n", encoding="utf-8")
                if self.stale_suffix == suffix:
                    old = time.time() - 3600
                    os.utime(out, (old, old))
            return parity.CommandResult(self.compare_rcs.get(suffix, self.compare_rc), "", "")
        return parity.CommandResult(0, "", "")

    def launch(self, rendered, cwd, env, log_path, port) -> parity.Launch:
        self.launches.append((rendered, port))
        self.launch_envs.append(dict(env))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(self.log_text, encoding="utf-8")
        return parity.Launch(FakeProcess(), log_path)

    def stop(self, launch, grace) -> int | None:
        self.stops += 1
        launch.process.returncode = 0
        return 0


def make_options(tmp_path: Path, *, dry_run: bool = False, variables: dict[str, str] | None = None) -> parity.Options:
    return parity.Options(
        dest=tmp_path / "dest",
        requests=tmp_path / "prompts.jsonl",
        health_url=parity.DEFAULT_HEALTH_URL,
        base_url=parity.DEFAULT_BASE_URL,
        port_base=18000,
        readiness_timeout=1.0,
        collect_timeout=10.0,
        stop_grace=2.0,
        variables={} if variables is None else variables,
        dry_run=dry_run,
        allow_single_command=False,
        reuse_dir=False,
    )


def make_plan(
    tmp_path: Path, command: str = "vllm serve org/model --port ${PORT}", env: dict | None = None
) -> parity.ConfigPlan:
    return parity.ConfigPlan(
        slug="fake-config",
        index=0,
        data_dir=tmp_path / "data",
        config_yaml="tests/e2e/cases/models/configs/GLM/fake.yaml",
        model="org/model",
        commands={"mrv1": command, "mrv2": command},
        copies={"mrv1": tmp_path / "copy.yaml", "mrv2": tmp_path / "copy.yaml"},
        env={} if env is None else env,
    )


def no_http(monkeypatch) -> None:
    monkeypatch.setattr(parity, "wait_for_health", lambda url, process, timeout: 0.0)
    monkeypatch.setattr(parity, "port_is_open", lambda port: False)


def read_manifest(plan: parity.ConfigPlan) -> dict:
    return json.loads((plan.data_dir / "manifest.json").read_text(encoding="utf-8"))


def test_dry_run_prints_the_sequence_and_never_launches(tmp_path, monkeypatch, capsys):
    no_http(monkeypatch)
    plan = make_plan(tmp_path, command="vllm serve org/model --port ${PORT} --seed 1024")
    options = make_options(tmp_path, dry_run=True)
    executor = FakeExecutor()
    assert parity.execute_config(plan, options, executor) == 0
    payload = json.loads(capsys.readouterr().out)
    assert executor.launches == []
    assert executor.calls == []
    assert not (plan.data_dir / "manifest.json").exists()
    for backend, entry in zip(parity.BACKENDS, payload["backends"], strict=True):
        assert entry["backend"] == backend
        assert entry["port"] == options.port_base + parity.BACKENDS.index(backend)
        assert entry["serve_command"].startswith("vllm serve org/model")
        assert entry["env_overrides"]["SERVER_PORT"] == str(entry["port"])
        assert entry["env_overrides"][parity.RUNNER_ENV] == parity.BACKEND_VALUES[backend]
        assert entry["gate_argv"][2] == "verify-log"
        assert entry["collect_argv"][2] == "collect"
        assert Path(entry["server_log"]).parent.name == backend
        assert Path(entry["server_log"]).name == "server.log"


def test_gate_failure_aborts_before_collect_and_writes_no_collect_file(tmp_path, monkeypatch, capsys):
    no_http(monkeypatch)
    plan = make_plan(tmp_path)
    options = make_options(tmp_path)
    executor = FakeExecutor(gate_rc=1)
    assert parity.execute_config(plan, options, executor) == 2
    manifest = read_manifest(plan)
    assert "gate failed" in manifest["abort_reason"]
    assert executor.calls == ["verify-log"]
    assert manifest["backends"]["mrv1"]["collect_exit"] is None
    assert not (plan.data_dir / "mrv1" / "collect.jsonl").exists()
    assert not (plan.data_dir / "mrv2" / "collect.jsonl").exists()
    assert "compare" not in executor.calls


def test_gate_runs_before_collect_for_each_backend(tmp_path, monkeypatch):
    no_http(monkeypatch)
    plan = make_plan(tmp_path)
    executor = FakeExecutor()
    assert parity.execute_config(plan, make_options(tmp_path), executor) == 0
    assert executor.calls == ["verify-log", "collect", "verify-log", "collect", "compare", "compare"]


def test_collect_failure_aborts_the_run(tmp_path, monkeypatch):
    no_http(monkeypatch)
    plan = make_plan(tmp_path)
    executor = FakeExecutor(collect_rc=1)
    assert parity.execute_config(plan, make_options(tmp_path), executor) == 2
    manifest = read_manifest(plan)
    assert "collect failed" in manifest["abort_reason"]
    assert executor.calls == ["verify-log", "collect"]
    assert "mrv2" not in manifest["backends"]


def test_compare_mismatch_is_a_result_and_exits_one(tmp_path, monkeypatch):
    no_http(monkeypatch)
    plan = make_plan(tmp_path)
    executor = FakeExecutor(compare_rc=1)
    assert parity.execute_config(plan, make_options(tmp_path), executor) == 1
    manifest = read_manifest(plan)
    assert manifest["compare_exit"] == 1
    assert manifest["compare_verdict"] == "fail"
    assert manifest["compare_json"] is not None
    assert manifest["abort_reason"] is None


def test_compare_inconclusive_is_a_result_not_a_pipeline_failure(tmp_path, monkeypatch):
    no_http(monkeypatch)
    plan = make_plan(tmp_path)
    executor = FakeExecutor(compare_rc=2)
    assert parity.execute_config(plan, make_options(tmp_path), executor) == 2
    manifest = read_manifest(plan)
    assert manifest["compare_exit"] == 2
    assert manifest["compare_verdict"] == "inconclusive"
    assert manifest["abort_reason"] is None
    assert manifest["compare_md"] is not None
    assert manifest["compare_json"] is not None
    assert (plan.data_dir / "compare.json").is_file()


def test_compare_crash_without_a_report_still_aborts(tmp_path, monkeypatch):
    no_http(monkeypatch)
    plan = make_plan(tmp_path)
    executor = FakeExecutor(compare_rc=2, compare_writes=False)
    assert parity.execute_config(plan, make_options(tmp_path), executor) == 2
    manifest = read_manifest(plan)
    assert "produced no compare.md this run" in manifest["abort_reason"]
    assert "no output" in manifest["abort_reason"]
    assert manifest["compare_exit"] is None
    assert "no compare.md" in manifest["compare_note"]
    assert manifest["compare_verdict"] is None
    assert manifest["compare_md"] is None
    assert manifest["compare_json"] is None


def test_manifest_carries_the_plan_fields_and_per_backend_facts(tmp_path, monkeypatch):
    no_http(monkeypatch)
    plan = make_plan(tmp_path)
    executor = FakeExecutor()
    assert parity.execute_config(plan, make_options(tmp_path), executor) == 0
    manifest = read_manifest(plan)
    for field in parity.MANIFEST_FIELDS:
        assert field in manifest
    assert manifest["image_tag"] is None and manifest["cann_version"] is None
    assert manifest["backend_expected"] == ["mrv1", "mrv2"]
    assert manifest["config_yaml"].endswith("fake.yaml")
    assert manifest["compare_exit"] == 0
    assert "backends.<backend>.log_sha256" in manifest["server_log_sha256_note"]
    for backend in parity.BACKENDS:
        entry = manifest["backends"][backend]
        assert entry["readiness_s"] == 0.0
        assert entry["exit_status"] == 0
        assert entry["log_sha256"] == parity.sha256_file(plan.data_dir / backend / "server.log")
        assert entry["collect_records"] == 2
    stages = [record["stage"] for record in manifest["commands"]]
    assert stages == ["verify-log", "collect", "verify-log", "collect", "compare", "compare"]
    assert all(record["argv"] for record in manifest["commands"])


def test_manifest_log_sha256_matches_the_file(tmp_path, monkeypatch):
    no_http(monkeypatch)
    plan = make_plan(tmp_path)
    executor = FakeExecutor(log_text="some log body\n")
    assert parity.execute_config(plan, make_options(tmp_path), executor) == 0
    expected = parity.sha256_file(plan.data_dir / "mrv2" / "server.log")
    assert read_manifest(plan)["backends"]["mrv2"]["log_sha256"] == expected


def test_ports_follow_port_base_and_never_collide():
    assert parity.port_for(8000, 0, "mrv1") == 8000
    assert parity.port_for(8000, 0, "mrv2") == 8001
    assert parity.port_for(8000, 1, "mrv1") == 8002
    ports = {parity.port_for(8000, index, backend) for index in range(3) for backend in parity.BACKENDS}
    assert len(ports) == 6


def test_resolver_fills_placeholders_from_the_variable_map():
    resolver = parity.VariableResolver({"PORT": "8000"}, {}, {})
    rendered, missing = resolver.expand("vllm serve org/model --port ${PORT}")
    assert (rendered, missing) == ("vllm serve org/model --port 8000", [])


def test_resolver_reports_every_missing_placeholder():
    resolver = parity.VariableResolver({}, {}, {})
    rendered, missing = resolver.expand("--port ${PORT} --tp ${TP_SIZE}")
    assert rendered == "--port ${PORT} --tp ${TP_SIZE}"
    assert missing == [("PORT", "placeholder"), ("TP_SIZE", "placeholder")]


def test_missing_placeholder_aborts_before_any_launch(tmp_path, monkeypatch):
    no_http(monkeypatch)
    plan = make_plan(tmp_path, command="vllm serve org/model --tp ${TP_SIZE}")
    executor = FakeExecutor()
    assert parity.execute_config(plan, make_options(tmp_path), executor) == 2
    assert executor.launches == []
    assert "unresolved placeholder" in read_manifest(plan)["abort_reason"]


def test_config_env_resolves_a_deployment_anchor():
    assert parity.config_env(ENV_FIXTURE) == {
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "SERVER_PORT": "8080",
        "HCCL_BUFFSIZE": "1024",
    }


def test_config_env_resolves_an_inline_test_cases_block():
    assert parity.config_env(CASE_FIXTURE) == {"VLLM_USE_V2_MODEL_RUNNER": "1", "SERVER_PORT": "8080"}


def test_config_env_resolves_an_aliased_envs_block():
    assert parity.config_env(ALIAS_FIXTURE)["VLLM_USE_V2_MODEL_RUNNER"] == "1"


def test_source_never_uses_a_shell_keyword_or_splits_a_command():
    source = Path(parity.__file__).read_text(encoding="utf-8")
    assert "shell=True" not in source
    assert '["sh", "-c"' in source
    assert "display.split" not in source
    assert "rendered.split" not in source
    assert not [line for line in source.splitlines() if ".split(" in line and "display" in line]


def test_file_descriptor_module_shadow_is_avoided():
    """Running this tool as a script must not let tools/bisect shadow the stdlib."""
    source = Path(parity.__file__).read_text(encoding="utf-8")
    assert "sys.path.pop(0)" in source
    assert source.index("sys.path.pop(0)") < source.index("import urllib.request")


def test_parse_variables_rejects_a_pair_without_a_name():
    assert parity.parse_variables(["A=1"]) == {"A": "1"}
    with pytest.raises(parity.InputError):
        parity.parse_variables(["A"])


def test_matches_slug_accepts_a_stem_or_a_suffix():
    rel = f"{GLM_DIR}/GLM-5.2-W8A8C8-MTP-A5-dual-nodes.yaml"
    assert parity.matches_slug(rel, "GLM-5.2-W8A8C8-MTP-A5-dual-nodes")
    assert parity.matches_slug(rel, rel)
    assert not parity.matches_slug(rel, "GLM-5.2-W8A8C8-layerwise-PD")


def test_stub_runner_lines_cover_the_three_modes():
    assert parity.runner_log_lines("mrv2", "explicit") == ["VLLM_USE_V2_MODEL_RUNNER=1 is set; using Model Runner V2."]
    assert "is unset" in parity.runner_log_lines("mrv1", "unset")[0]
    assert "using the V1 model runner instead" in parity.runner_log_lines("mrv1", "fallback")[0]
    assert len(parity.runner_log_lines("mrv1", "both")) == 2


def test_stub_server_answers_health_and_completions():
    port = free_port_pair()
    tool = Path(parity.__file__).resolve()
    process = subprocess.Popen(
        [sys.executable, str(tool), "_stub-server", "--port", str(port), "--backend", "mrv1"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as response:
                    assert response.status == 200
                break
            except OSError:
                time.sleep(0.1)
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/completions",
            data=b'{"prompt":"hi"}',
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            payload = json.loads(response.read())
        assert payload["choices"][0]["logprobs"]["tokens"] == list(parity.STUB_PROMPT_TOKENS)
    finally:
        process.terminate()
        process.wait(timeout=20)


def test_rehearse_publishes_the_full_archive(tmp_path):
    data = tmp_path / "data"
    port = free_port_pair()
    prompts = write_prompts(tmp_path)
    assert (
        parity.main(
            [
                "rehearse",
                "--stub",
                "--data-dir",
                str(data),
                "--requests",
                str(prompts),
                "--port-base",
                str(port),
            ]
        )
        == 0
    )
    for relative in (
        "manifest.json",
        "mrv1/server.log",
        "mrv1/collect.jsonl",
        "mrv2/server.log",
        "mrv2/collect.jsonl",
        "compare.md",
        "compare.json",
    ):
        assert (data / relative).is_file(), relative
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["compare_exit"] == 0
    assert manifest["abort_reason"] is None
    assert set(manifest["compare_reports"]) == {"md", "json"}
    assert "Traceback" not in (data / "mrv1" / "server.log").read_text(encoding="utf-8")
    for backend in parity.BACKENDS:
        assert manifest["backends"][backend]["collect_records"] == 2
        assert manifest["backends"][backend]["log_sha256"]
    assert "PASS" in (data / "compare.md").read_text(encoding="utf-8")


def test_rehearse_aborts_on_an_unset_log(tmp_path):
    data = tmp_path / "data"
    port = free_port_pair()
    assert (
        parity.main(
            [
                "rehearse",
                "--stub",
                "--log-mode",
                "unset",
                "--data-dir",
                str(data),
                "--requests",
                str(write_prompts(tmp_path)),
                "--port-base",
                str(port),
            ]
        )
        == 2
    )
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    assert "NOT applied" in manifest["abort_reason"]
    assert not (data / "mrv1" / "collect.jsonl").exists()
    assert not (data / "mrv2" / "collect.jsonl").exists()


def test_rehearse_aborts_on_a_fallback_log(tmp_path):
    data = tmp_path / "data"
    port = free_port_pair()
    assert (
        parity.main(
            [
                "rehearse",
                "--stub",
                "--log-mode",
                "fallback",
                "--data-dir",
                str(data),
                "--requests",
                str(write_prompts(tmp_path)),
                "--port-base",
                str(port),
            ]
        )
        == 2
    )
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    assert "fallback" in manifest["abort_reason"]
    assert not (data / "mrv1" / "collect.jsonl").exists()
    assert not parity.LIVE_PROCESSES


def test_readiness_timeout_aborts_and_kills_the_server(tmp_path):
    data = tmp_path / "data"
    port = free_port_pair()
    assert (
        parity.main(
            [
                "rehearse",
                "--stub",
                "--serve-cmd",
                "sleep 60",
                "--readiness-timeout",
                "1.0",
                "--data-dir",
                str(data),
                "--requests",
                str(write_prompts(tmp_path)),
                "--port-base",
                str(port),
            ]
        )
        == 2
    )
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    assert "not ready" in manifest["abort_reason"]
    assert not (data / "mrv1" / "collect.jsonl").exists()
    assert not parity.LIVE_PROCESSES
    assert not parity.port_is_open(port)


def test_run_dry_run_on_a_real_config_writes_nothing(tmp_path, capsys):
    data = tmp_path / "data"
    assert (
        parity.main(
            [
                "run",
                "--configs",
                "glm-5.2-W4A8C8-DSpark-PCP",
                "--dest",
                str(tmp_path / "dest"),
                "--data-dir",
                str(data),
                "--dry-run",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True
    assert payload["command_count"] == 1
    assert payload["single_command_override"] is False
    assert payload["ignored_commands"] == {"mrv1": 0, "mrv2": 0}
    assert payload["unresolved_variables"] == []
    assert payload["config_yaml"].endswith("glm-5.2-W4A8C8-DSpark-PCP.yaml")
    assert len(payload["backends"]) == 2
    assert payload["backends"][0]["env_overrides"]["SERVER_PORT"] == str(payload["backends"][0]["port"])
    assert payload["backends"][0]["serve_command"].startswith("vllm serve Eco-Tech/GLM-5.2-w4a8c8")
    assert not data.exists()


def test_run_dry_run_reports_missing_placeholders(tmp_path, capsys):
    assert (
        parity.main(
            [
                "run",
                "--configs",
                "GLM-5.2-W8A8C8-layerwise-PD",
                "--dest",
                str(tmp_path / "dest"),
                "--data-dir",
                str(tmp_path / "data"),
                "--dry-run",
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert "DP_SIZE" in captured.out
    assert "PYTHON_LIB_DIR" in captured.out
    assert "serve commands" in captured.err
    assert not (tmp_path / "data").exists()


def test_load_plans_reads_the_plan_json_and_the_copy_env(tmp_path):
    copy = tmp_path / "dest" / "mrv1" / "fake.yaml"
    copy.parent.mkdir(parents=True)
    copy.write_text(ENV_FIXTURE, encoding="utf-8")
    executor = FakeExecutor()
    executor.plan_payload = {
        "configs": [
            {
                "path": "tests/e2e/cases/models/configs/GLM/fake.yaml",
                "targets": {
                    "mrv1": {
                        "dest": str(copy),
                        "commands": [{"display": "vllm serve org/model", "model": "org/model"}],
                    },
                    "mrv2": {
                        "dest": str(copy),
                        "commands": [{"display": "vllm serve org/model", "model": "org/model"}],
                    },
                },
            }
        ]
    }
    records: list[dict] = []
    plans, problems = parity.load_plans(["fake"], tmp_path / "data", make_options(tmp_path), executor, records)
    assert problems == []
    assert [record["stage"] for record in records] == ["plan"]
    assert len(plans) == 1
    assert plans[0].model == "org/model"
    assert plans[0].commands["mrv1"] == "vllm serve org/model"
    assert plans[0].env["SERVER_PORT"] == "8080"
    assert plans[0].data_dir == tmp_path / "data"
    plan_argv = records[0]["argv"]
    assert plan_argv[plan_argv.index("--only") + 1] == "fake.yaml"


def plan_payload(copy: Path, count: int = 1, carrier: str = "test_cases") -> dict:
    commands = [{"display": "vllm serve org/model --port $SERVER_PORT", "model": "org/model"} for _ in range(count)]
    return {
        "configs": [
            {
                "path": "tests/e2e/cases/models/configs/GLM/fake.yaml",
                "carrier": carrier,
                "targets": {backend: {"dest": str(copy), "commands": commands} for backend in parity.BACKENDS},
            }
        ]
    }


def test_allow_single_command_records_ignored_commands(tmp_path):
    copy = tmp_path / "copy.yaml"
    copy.write_text(CASE_FIXTURE, encoding="utf-8")
    executor = FakeExecutor()
    executor.plan_payload = plan_payload(copy, count=2)
    options = replace(make_options(tmp_path), allow_single_command=True)
    plans, problems = parity.load_plans(["fake"], tmp_path / "data", options, executor, [])
    assert problems == []
    assert plans[0].single_command_override is True
    assert plans[0].command_count == 2
    assert plans[0].command_counts == {"mrv1": 2, "mrv2": 2}


def test_allow_single_command_override_reaches_the_manifest(tmp_path, monkeypatch):
    no_http(monkeypatch)
    plan = make_plan(tmp_path)
    plan.command_counts = {"mrv1": 2, "mrv2": 2}
    plan.command_count = 2
    plan.single_command_override = True
    assert parity.execute_config(plan, make_options(tmp_path), FakeExecutor()) == 0
    manifest = read_manifest(plan)
    assert manifest["single_command_override"] is True
    assert manifest["command_count"] == 2
    assert manifest["backends"]["mrv1"]["ignored_commands"] == 1
    assert manifest["backends"]["mrv2"]["ignored_commands"] == 1
    assert "backends.<backend>.log_sha256" in manifest["server_log_sha256_note"]


def test_dry_run_payload_exposes_command_counts_and_ignored(tmp_path, monkeypatch, capsys):
    no_http(monkeypatch)
    plan = make_plan(tmp_path)
    plan.command_counts = {"mrv1": 3, "mrv2": 3}
    plan.command_count = 3
    plan.single_command_override = True
    assert parity.execute_config(plan, make_options(tmp_path, dry_run=True), FakeExecutor()) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["command_count"] == 3
    assert payload["single_command_override"] is True
    assert payload["ignored_commands"] == {"mrv1": 2, "mrv2": 2}
    assert payload["backends"][0]["ignored_commands"] == 2
    assert payload["unresolved_variables"] == []


def test_real_multi_command_dry_run_reports_the_truncation(tmp_path, capsys):
    assert (
        parity.main(
            [
                "run",
                "--configs",
                "GLM-5.2-W8A8C8-MTP-A5-dual-nodes",
                "--dest",
                str(tmp_path / "dest"),
                "--data-dir",
                str(tmp_path / "data"),
                "--dry-run",
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert "2 serve commands" in captured.err
    assert "MASTER_IP" in captured.out
    assert not (tmp_path / "data").exists()


def test_allow_single_command_still_refuses_multi_node(tmp_path, capsys):
    assert (
        parity.main(
            [
                "run",
                "--configs",
                "GLM-5.2-W8A8C8-MTP-A5-dual-nodes",
                "--dest",
                str(tmp_path / "dest"),
                "--data-dir",
                str(tmp_path / "data"),
                "--dry-run",
                "--allow-single-command",
            ]
        )
        == 2
    )
    assert "num_nodes=2" in capsys.readouterr().err


def test_allow_single_command_still_refuses_external_templates(tmp_path, capsys):
    assert (
        parity.main(
            [
                "run",
                "--configs",
                "GLM-5.2-W8A8C8-layerwise-PD",
                "--dest",
                str(tmp_path / "dest"),
                "--data-dir",
                str(tmp_path / "data"),
                "--dry-run",
                "--allow-single-command",
            ]
        )
        == 2
    )
    assert "external_templates" in capsys.readouterr().err


def test_pd_and_multinode_detection():
    assert parity.is_pd_config("GLM-5.2-W8A8C8-layerwise-PD.yaml", "")
    assert parity.is_pd_config("x.yaml", "kv-transfer-config: {}")
    assert parity.is_pd_config("x.yaml", "env_prefill: &env_prefill")
    assert not parity.is_pd_config("GLM-5.2-W8A8C8-MTP-A5-dual-nodes.yaml", "env_common: &env_common")
    assert parity.config_num_nodes("num_nodes: 4\nmodel: x\n") == 4
    assert parity.config_num_nodes("model: x\n") is None
    assert parity.single_command_problem("a.yaml", 1, None, None, False, True) is None
    assert "serve commands" in parity.single_command_problem("a.yaml", 2, "test_cases", 1, False, False)
    assert "external_templates" in parity.single_command_problem("a.yaml", 2, "external_templates", 1, False, True)
    assert "num_nodes=4" in parity.single_command_problem("a.yaml", 2, "deployment", 4, False, True)
    assert "PD" in parity.single_command_problem("a-PD.yaml", 2, "deployment", 1, True, True)


def test_unresolved_bare_variable_aborts_before_launch(tmp_path, monkeypatch):
    no_http(monkeypatch)
    plan = make_plan(tmp_path, command="vllm serve org/model --master-addr $MASTER_IP")
    executor = FakeExecutor()
    assert parity.execute_config(plan, make_options(tmp_path), executor) == 2
    manifest = read_manifest(plan)
    assert "unresolved variable" in manifest["abort_reason"]
    assert "MASTER_IP" in manifest["abort_reason"]
    assert executor.launches == []


def test_unresolved_variable_in_an_env_value_aborts(tmp_path, monkeypatch):
    no_http(monkeypatch)
    plan = make_plan(tmp_path, env={"LD_LIBRARY_PATH": "$PYTHON_LIB_DIR:/usr/lib"})
    executor = FakeExecutor()
    assert parity.execute_config(plan, make_options(tmp_path), executor) == 2
    reason = read_manifest(plan)["abort_reason"]
    assert "PYTHON_LIB_DIR" in reason
    assert "env:LD_LIBRARY_PATH" in reason
    assert executor.launches == []


def test_var_option_is_injected_into_the_child_env(tmp_path, monkeypatch):
    no_http(monkeypatch)
    plan = make_plan(tmp_path, command="vllm serve org/model --master-addr $MASTER_IP")
    executor = FakeExecutor()
    options = make_options(tmp_path, variables={"MASTER_IP": "10.0.0.9"})
    assert parity.execute_config(plan, options, executor) == 0
    assert executor.launch_envs[0]["MASTER_IP"] == "10.0.0.9"


def test_dry_run_reports_unresolved_variables_and_exits_two(tmp_path, monkeypatch, capsys):
    no_http(monkeypatch)
    plan = make_plan(
        tmp_path, command="vllm serve org/model --master-addr $MASTER_IP", env={"LD_LIBRARY_PATH": "$PYTHON_LIB_DIR"}
    )
    assert parity.execute_config(plan, make_options(tmp_path, dry_run=True), FakeExecutor()) == 2
    payload = json.loads(capsys.readouterr().out)
    entries = payload["unresolved_variables"]
    assert {item["variable"] for item in entries} == {"MASTER_IP", "PYTHON_LIB_DIR"}
    assert {item["location"] for item in entries} == {"command", "env:LD_LIBRARY_PATH"}
    assert not (plan.data_dir / "manifest.json").exists()


def test_resolving_the_variables_lets_dry_run_pass(tmp_path, monkeypatch, capsys):
    no_http(monkeypatch)
    plan = make_plan(
        tmp_path, command="vllm serve org/model --master-addr $MASTER_IP", env={"LD_LIBRARY_PATH": "$PYTHON_LIB_DIR"}
    )
    options = make_options(tmp_path, dry_run=True, variables={"MASTER_IP": "10.0.0.9", "PYTHON_LIB_DIR": "/usr/lib"})
    assert parity.execute_config(plan, options, FakeExecutor()) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["unresolved_variables"] == []


def test_real_layerwise_dry_run_lists_the_env_variable(tmp_path, capsys):
    assert (
        parity.main(
            [
                "run",
                "--configs",
                "GLM-5.2-W8A8C8-layerwise-PD",
                "--dest",
                str(tmp_path / "dest"),
                "--data-dir",
                str(tmp_path / "data"),
                "--dry-run",
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert "PYTHON_LIB_DIR" in captured.out
    assert "serve commands" in captured.err


def test_env_value_bare_variable_is_expanded_for_the_child(tmp_path, monkeypatch):
    no_http(monkeypatch)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/old")
    plan = make_plan(tmp_path, env={"LD_LIBRARY_PATH": "$PYTHON_LIB_DIR:/usr/local/lib:$LD_LIBRARY_PATH"})
    executor = FakeExecutor()
    options = make_options(tmp_path, variables={"PYTHON_LIB_DIR": "/opt"})
    assert parity.execute_config(plan, options, executor) == 0
    assert executor.launch_envs[0]["LD_LIBRARY_PATH"] == "/opt:/usr/local/lib:/old"


def test_env_value_missing_variable_aborts_with_the_env_location(tmp_path, monkeypatch):
    no_http(monkeypatch)
    monkeypatch.delenv("PYTHON_LIB_DIR", raising=False)
    plan = make_plan(tmp_path, env={"LD_LIBRARY_PATH": "$PYTHON_LIB_DIR:/usr/local/lib"})
    executor = FakeExecutor()
    assert parity.execute_config(plan, make_options(tmp_path), executor) == 2
    reason = read_manifest(plan)["abort_reason"]
    assert "PYTHON_LIB_DIR" in reason
    assert "env:LD_LIBRARY_PATH" in reason
    assert executor.launches == []


def test_plain_env_values_are_passed_through_unchanged(tmp_path, monkeypatch):
    no_http(monkeypatch)
    env = {"HCCL_BUFFSIZE": "1024", "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True"}
    plan = make_plan(tmp_path, env=env)
    executor = FakeExecutor()
    assert parity.execute_config(plan, make_options(tmp_path), executor) == 0
    captured = executor.launch_envs[0]
    for key, value in env.items():
        assert captured[key] == value


def test_env_value_can_reference_another_config_env_key(tmp_path, monkeypatch):
    no_http(monkeypatch)
    plan = make_plan(tmp_path, env={"BASE_DIR": "/opt/lib", "LD_LIBRARY_PATH": "$BASE_DIR:/usr/local/lib"})
    executor = FakeExecutor()
    assert parity.execute_config(plan, make_options(tmp_path), executor) == 0
    assert executor.launch_envs[0]["LD_LIBRARY_PATH"] == "/opt/lib:/usr/local/lib"


def test_config_env_placeholder_in_the_command_is_rendered(tmp_path, monkeypatch, capsys):
    no_http(monkeypatch)
    plan = make_plan(tmp_path, command="vllm serve org/model --host ${HOST_IP}", env={"HOST_IP": "10.0.0.5"})
    assert parity.execute_config(plan, make_options(tmp_path, dry_run=True), FakeExecutor()) == 0
    payload = json.loads(capsys.readouterr().out)
    assert "--host 10.0.0.5" in payload["backends"][0]["serve_command"]


def test_placeholder_resolves_from_os_environ(tmp_path, monkeypatch, capsys):
    no_http(monkeypatch)
    monkeypatch.setenv("ONLY_ENV", "42")
    plan = make_plan(tmp_path, command="vllm serve org/model --seed ${ONLY_ENV}")
    assert parity.execute_config(plan, make_options(tmp_path, dry_run=True), FakeExecutor()) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["backends"][0]["serve_command"].endswith("--seed 42")
    assert "ONLY_ENV" in payload["resolved_from_environ"]


def test_bare_command_variable_resolves_from_os_environ(tmp_path, monkeypatch, capsys):
    no_http(monkeypatch)
    monkeypatch.setenv("LOCAL_IP", "10.0.0.1")
    plan = make_plan(tmp_path, command="vllm serve org/model --host $LOCAL_IP")
    assert parity.execute_config(plan, make_options(tmp_path, dry_run=True), FakeExecutor()) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["unresolved_variables"] == []
    assert "LOCAL_IP" in payload["resolved_from_environ"]


def test_placeholder_error_carries_the_location(tmp_path, monkeypatch):
    no_http(monkeypatch)
    plan = make_plan(tmp_path, command="vllm serve org/model --tp ${TP_SIZE}")
    assert parity.execute_config(plan, make_options(tmp_path), FakeExecutor()) == 2
    reason = read_manifest(plan)["abort_reason"]
    assert "TP_SIZE" in reason
    assert '"location": "command"' in reason


def test_var_cannot_override_a_reserved_name(tmp_path, capsys):
    with pytest.raises(parity.InputError, match="reserved"):
        parity.parse_variables(["SERVER_PORT=1"])
    argv = [
        "run",
        "--configs",
        "glm-5.2-W4A8C8-DSpark-PCP",
        "--dest",
        str(tmp_path / "dest"),
        "--data-dir",
        str(tmp_path / "data"),
        "--dry-run",
        "--var",
        f"{parity.RUNNER_ENV}=1",
    ]
    assert parity.main(argv) == 2
    assert "reserved" in capsys.readouterr().err


def test_manifest_records_the_variables_read_from_the_environment(tmp_path, monkeypatch):
    no_http(monkeypatch)
    monkeypatch.setenv("ONLY_ENV", "42")
    plan = make_plan(tmp_path, command="vllm serve org/model --seed ${ONLY_ENV}")
    assert parity.execute_config(plan, make_options(tmp_path), FakeExecutor()) == 0
    assert read_manifest(plan)["resolved_from_environ"] == ["ONLY_ENV"]


def test_readiness_uses_the_config_timeout(tmp_path, monkeypatch):
    no_http(monkeypatch)
    plan = make_plan(tmp_path)
    plan.engine_ready_timeout = 1830.0
    options = replace(make_options(tmp_path), readiness_timeout=None)
    assert parity.execute_config(plan, options, FakeExecutor()) == 0
    manifest = read_manifest(plan)
    assert manifest["readiness_timeout"] == 1830.0
    assert manifest["readiness_source"] == "config"
    assert manifest["backends"]["mrv1"]["readiness_timeout"] == 1830.0


def test_cli_readiness_wins_only_when_larger(tmp_path, monkeypatch):
    no_http(monkeypatch)
    plan = make_plan(tmp_path)
    plan.engine_ready_timeout = 1830.0
    larger = replace(make_options(tmp_path), readiness_timeout=2400.0)
    assert parity.execute_config(plan, larger, FakeExecutor()) == 0
    manifest = read_manifest(plan)
    assert manifest["readiness_timeout"] == 2400.0
    assert manifest["readiness_source"] == "cli"

    second = tmp_path / "second"
    second.mkdir()
    plan2 = make_plan(second)
    plan2.engine_ready_timeout = 1830.0
    smaller = replace(make_options(tmp_path), readiness_timeout=300.0)
    assert parity.execute_config(plan2, smaller, FakeExecutor()) == 0
    manifest2 = read_manifest(plan2)
    assert manifest2["readiness_timeout"] == 1830.0
    assert manifest2["readiness_source"] == "config"


def test_default_readiness_is_1800_and_dry_run_echoes_it(tmp_path, monkeypatch, capsys):
    no_http(monkeypatch)
    plan = make_plan(tmp_path)
    options = replace(make_options(tmp_path, dry_run=True), readiness_timeout=None)
    assert parity.execute_config(plan, options, FakeExecutor()) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["readiness_timeout"] == 1800.0
    assert payload["readiness_source"] == "default"
    assert parity.DEFAULT_READINESS_TIMEOUT == 1800.0


def test_non_empty_data_dir_without_reuse_exits_two(tmp_path, monkeypatch, capsys):
    no_http(monkeypatch)
    plan = make_plan(tmp_path)
    plan.data_dir.mkdir(parents=True)
    (plan.data_dir / "stale.txt").write_text("old\n", encoding="utf-8")
    executor = FakeExecutor()
    assert parity.execute_config(plan, make_options(tmp_path), executor) == 2
    assert executor.launches == []
    assert executor.calls == []
    assert "pass --reuse-dir" in capsys.readouterr().err


def test_reuse_dir_archives_the_previous_comparison(tmp_path, monkeypatch):
    no_http(monkeypatch)
    plan = make_plan(tmp_path)
    plan.data_dir.mkdir(parents=True)
    (plan.data_dir / "compare.md").write_text("old report\n", encoding="utf-8")
    options = replace(make_options(tmp_path), reuse_dir=True)
    assert parity.execute_config(plan, options, FakeExecutor()) == 0
    previous = plan.data_dir / "compare.prev.md"
    assert previous.read_text(encoding="utf-8") == "old report\n"
    manifest = read_manifest(plan)
    assert manifest["reused_dir"] is True
    assert manifest["archived_compare"] == [str(previous)]


def test_manifest_self_description(tmp_path, monkeypatch):
    no_http(monkeypatch)
    plan = make_plan(tmp_path)
    assert parity.execute_config(plan, make_options(tmp_path), FakeExecutor()) == 0
    manifest = read_manifest(plan)
    assert manifest["run_mode"] == "real"
    assert manifest["warning"] is None
    assert manifest["identity_source"] == "unset"
    for identity_field in parity.IDENTITY_FIELDS:
        assert manifest[identity_field] is None
    backend = manifest["backends"]["mrv1"]
    assert backend["serve_command"].startswith("vllm serve org/model")
    assert set(backend["child_env_subset"]) <= set(parity.CHILD_ENV_KEYS)
    assert backend["child_env_subset"][parity.RUNNER_ENV] == parity.BACKEND_VALUES["mrv1"]
    assert "SERVER_PORT" in backend["child_env_subset"]
    assert "HCCL_BUFFSIZE" not in backend["child_env_subset"]


def test_manifest_records_cli_identity_and_stub_warning(tmp_path, monkeypatch):
    no_http(monkeypatch)
    plan = make_plan(tmp_path)
    plan.run_mode = "stub"
    options = replace(make_options(tmp_path), image_tag="img:1", hardware="Atlas A2")
    assert parity.execute_config(plan, options, FakeExecutor()) == 0
    manifest = read_manifest(plan)
    assert manifest["run_mode"] == "stub"
    assert manifest["warning"] == parity.STUB_WARNING
    assert manifest["identity_source"] == "cli"
    assert manifest["image_tag"] == "img:1"
    assert manifest["hardware"] == "Atlas A2"
    assert manifest["torch_npu_version"] is None


def test_install_signal_handlers_registers_available_signals():
    import signal

    original = {name: signal.getsignal(getattr(signal, name)) for name in ("SIGTERM", "SIGINT")}
    try:
        installed = parity.install_signal_handlers()
        assert "SIGTERM" in installed
        assert "SIGINT" in installed
        assert signal.getsignal(signal.SIGTERM) is parity.handle_termination
        assert signal.getsignal(signal.SIGINT) is parity.handle_termination
    finally:
        for name, handler in original.items():
            signal.signal(getattr(signal, name), handler)


def test_terminate_process_groups_uses_the_child_group_on_posix(tmp_path, monkeypatch):
    calls: list[tuple[int, int]] = []

    def fake_killpg(pgid, sig):
        calls.append((pgid, sig))
        if sig == parity.signal.SIGTERM:
            process.returncode = 0

    monkeypatch.setattr(parity, "is_windows", lambda: False)
    monkeypatch.setattr(parity.os, "killpg", fake_killpg, raising=False)
    monkeypatch.setattr(
        parity.os, "getpgid", lambda pid: pytest.fail("handler must use the stored pgid"), raising=False
    )
    monkeypatch.setattr(parity.subprocess, "run", lambda *args, **kwargs: pytest.fail("taskkill on POSIX"))
    process = FakeProcess()
    process.pid = 123
    parity.LIVE_PROCESSES.add(process)
    parity.LIVE_GROUPS[process.pid] = 5555
    try:
        parity.terminate_process_groups()
    finally:
        parity.LIVE_PROCESSES.discard(process)
        parity.LIVE_GROUPS.pop(process.pid, None)
    # 5555 is the child's recorded pgid; the executor's own group (getpgid(0)) is never used.
    assert calls == [(5555, parity.signal.SIGTERM)]


def test_terminate_process_groups_uses_taskkill_on_windows(tmp_path, monkeypatch):
    commands: list[list[str]] = []
    monkeypatch.setattr(parity, "is_windows", lambda: True)
    monkeypatch.setattr(parity.os, "killpg", lambda pid, sig: pytest.fail("killpg on Windows"), raising=False)
    monkeypatch.setattr(
        parity.subprocess,
        "run",
        lambda argv, **kwargs: commands.append(list(argv)) or subprocess.CompletedProcess(argv, 0),
    )
    process = FakeProcess()
    process.pid = 23732
    parity.LIVE_PROCESSES.add(process)
    try:
        parity.terminate_process_groups()
    finally:
        parity.LIVE_PROCESSES.discard(process)
    assert ["taskkill", "/F", "/T", "/PID", "23732"] in commands


def test_stub_server_text_words_match_token_count(tmp_path):
    port = free_port_pair()
    tool = Path(parity.__file__).resolve()
    process = subprocess.Popen(
        [sys.executable, str(tool), "_stub-server", "--port", str(port), "--tokens", "4"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as response:
                    assert response.status == 200
                break
            except OSError:
                time.sleep(0.1)
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/completions",
            data=b'{"prompt":"hi"}',
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            payload = json.loads(response.read())
        tokens = payload["choices"][0]["logprobs"]["tokens"]
        text = payload["choices"][0]["text"]
        assert len(tokens) == 4
        assert text.split() == tokens
    finally:
        process.terminate()
        process.wait(timeout=20)


def test_rehearse_four_tokens_is_inconclusive_and_keeps_its_reports(tmp_path):
    data = tmp_path / "data4"
    port = free_port_pair()
    prompts = write_prompts(tmp_path)
    assert (
        parity.main(
            [
                "rehearse",
                "--stub",
                "--stub-tokens",
                "4",
                "--data-dir",
                str(data),
                "--requests",
                str(prompts),
                "--port-base",
                str(port),
            ]
        )
        == 2
    )
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["compare_verdict"] == "inconclusive"
    assert manifest["abort_reason"] is None
    assert manifest["compare_exit"] == 2
    assert manifest["compare_md"] is not None and manifest["compare_json"] is not None
    assert (data / "compare.md").is_file() and (data / "compare.json").is_file()


def test_commands_env_is_whitelisted(tmp_path, monkeypatch):
    no_http(monkeypatch)
    plan = make_plan(tmp_path, env={"HCCL_BUFFSIZE": "1024", "SERVER_PORT": "8080"})
    assert parity.execute_config(plan, make_options(tmp_path), FakeExecutor()) == 0
    manifest = read_manifest(plan)
    for record in manifest["commands"]:
        assert set(record["env"]) <= set(parity.CHILD_ENV_KEYS), record["stage"]
    assert all("HCCL_BUFFSIZE" not in record["env"] for record in manifest["commands"])


def test_old_report_is_archived_and_never_accepted(tmp_path, monkeypatch, capsys):
    no_http(monkeypatch)
    plan = make_plan(tmp_path)
    plan.data_dir.mkdir(parents=True)
    old = plan.data_dir / "compare.json"
    old.write_text('{"verdict": "PASS"}\n', encoding="utf-8")
    os.utime(old, (time.time() - 1.5, time.time() - 1.5))
    options = replace(make_options(tmp_path), reuse_dir=True)
    executor = FakeExecutor(compare_writes=False)
    assert parity.execute_config(plan, options, executor) == 2
    manifest = read_manifest(plan)
    # The 1.5 s old PASS json was moved away before the run and cannot be accepted.
    assert not (plan.data_dir / "compare.json").exists()
    assert (plan.data_dir / "compare.prev.json").read_text(encoding="utf-8") == '{"verdict": "PASS"}\n'
    assert manifest["compare_json"] is None
    assert manifest["compare_note"]


def test_reuse_dir_archives_both_reports(tmp_path, monkeypatch):
    no_http(monkeypatch)
    plan = make_plan(tmp_path)
    plan.data_dir.mkdir(parents=True)
    (plan.data_dir / "compare.md").write_text("old md\n", encoding="utf-8")
    (plan.data_dir / "compare.json").write_text("old json\n", encoding="utf-8")
    options = replace(make_options(tmp_path), reuse_dir=True)
    assert parity.execute_config(plan, options, FakeExecutor()) == 0
    assert (plan.data_dir / "compare.prev.md").read_text(encoding="utf-8") == "old md\n"
    assert (plan.data_dir / "compare.prev.json").read_text(encoding="utf-8") == "old json\n"
    manifest = read_manifest(plan)
    assert len(manifest["archived_compare"]) == 2
    assert any(path.endswith("compare.prev.json") for path in manifest["archived_compare"])


def test_stop_skips_a_reaped_pid_on_windows(tmp_path, monkeypatch):
    commands: list[list[str]] = []
    monkeypatch.setattr(parity, "is_windows", lambda: True)
    monkeypatch.setattr(
        parity.subprocess,
        "run",
        lambda argv, **kwargs: commands.append(list(argv)) or subprocess.CompletedProcess(argv, 0),
    )
    process = FakeProcess()
    process.pid = 23732
    process.returncode = 1
    launch = parity.Launch(process, tmp_path / "server.log")
    parity.Executor().stop(launch, 0.1)
    assert ["taskkill", "/F", "/T", "/PID", "23732"] not in commands


def test_stop_taskkills_a_live_pid_on_windows(tmp_path, monkeypatch):
    commands: list[list[str]] = []
    monkeypatch.setattr(parity, "is_windows", lambda: True)
    monkeypatch.setattr(parity, "kill_port_owners", lambda port: False)
    monkeypatch.setattr(
        parity.subprocess,
        "run",
        lambda argv, **kwargs: commands.append(list(argv)) or subprocess.CompletedProcess(argv, 0),
    )

    class LiveProcess(FakeProcess):
        def wait(self, timeout=None):
            self.returncode = 0
            return 0

    process = LiveProcess()
    process.pid = 23732
    launch = parity.Launch(process, tmp_path / "server.log")
    parity.Executor().stop(launch, 0.1)
    assert ["taskkill", "/F", "/T", "/PID", "23732"] in commands


def test_residual_server_report_names_the_owned_pid(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(parity, "owned_listeners", lambda port: ([23732], []))
    assert parity.report_residual_servers([8000]) is False
    err = capsys.readouterr().err
    assert "port 8000" in err and "pid 23732" in err


def test_foreign_listener_is_only_a_warning(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(parity, "owned_listeners", lambda port: ([], [9999]))
    assert parity.report_residual_servers([8000]) is True
    err = capsys.readouterr().err
    assert "warning:" in err and "9999" in err


def test_readiness_must_be_positive(tmp_path, capsys):
    prompts = write_prompts(tmp_path)
    for value in ("0", "-5"):
        code = parity.main(
            [
                "rehearse",
                "--stub",
                "--data-dir",
                str(tmp_path / f"d{value}"),
                "--requests",
                str(prompts),
                "--readiness-timeout",
                value,
            ]
        )
        assert code == 2
        assert "--readiness-timeout must be positive" in capsys.readouterr().err


def test_data_dir_that_is_a_regular_file_is_rejected(tmp_path, capsys):
    blocker = tmp_path / "blocker"
    blocker.write_text("not a dir\n", encoding="utf-8")
    code = parity.main(["rehearse", "--stub", "--data-dir", str(blocker), "--requests", str(write_prompts(tmp_path))])
    assert code == 2
    assert "regular file" in capsys.readouterr().err


def test_stub_tokens_must_be_within_range(tmp_path, capsys):
    for value in ("0", "-1", "33"):
        code = parity.main(["rehearse", "--stub", "--data-dir", str(tmp_path / f"d{value}"), "--stub-tokens", value])
        assert code == 2
        assert "must be between 1 and 32" in capsys.readouterr().err


def test_signal_abort_during_load_plans_exits_cleanly(tmp_path, monkeypatch, capsys):
    def explode(*args, **kwargs):
        raise parity.SignalAbort("terminated by SIGINT")

    monkeypatch.setattr(parity, "load_plans", explode)
    code = parity.main(
        ["run", "--configs", "x", "--dest", str(tmp_path / "dest"), "--data-dir", str(tmp_path / "data")]
    )
    assert code == 2
    err = capsys.readouterr().err
    assert "terminated by SIGINT" in err
    assert "Traceback" not in err


def test_run_entry_rejects_a_file_data_dir_without_a_traceback(tmp_path, capsys):
    blocker = tmp_path / "blocker"
    blocker.write_text("not a dir\n", encoding="utf-8")
    for extra in ((), ("--dry-run",)):
        code = parity.main(
            ["run", "--configs", "x", "--dest", str(tmp_path / "dest"), "--data-dir", str(blocker), *extra]
        )
        assert code == 2
        err = capsys.readouterr().err
        assert "regular file" in err
        assert "Traceback" not in err


def test_listening_query_asks_for_listen_state_only(monkeypatch):
    seen: list[list[str]] = []

    def fake_run(argv, **kwargs):
        seen.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, "1048\n1049\n", "")

    monkeypatch.setattr(parity, "is_windows", lambda: False)
    monkeypatch.setattr(parity.subprocess, "run", fake_run)
    assert parity.listening_pids(24560) == [1048, 1049]
    assert seen[0][:2] == ["lsof", "-nP"]
    assert "-sTCP:LISTEN" in seen[0]


def test_lsof_missing_degrades_to_a_warning(monkeypatch, capsys):
    def fake_run(argv, **kwargs):
        raise FileNotFoundError("lsof")

    monkeypatch.setattr(parity, "is_windows", lambda: False)
    monkeypatch.setattr(parity.subprocess, "run", fake_run)
    assert parity.listening_pids(24560) == []
    assert "lsof is unavailable" in capsys.readouterr().err


def test_posix_fallback_kills_the_listener_but_not_the_client(monkeypatch):
    killed: list[tuple[int, int]] = []
    monkeypatch.setattr(parity, "is_windows", lambda: False)
    monkeypatch.setattr(parity, "listening_pids", lambda port: [1048, 1049])
    monkeypatch.setattr(parity, "pid_owned_by_server", lambda pid: pid == 1048)
    monkeypatch.setattr(parity.os, "kill", lambda pid, sig: killed.append((pid, sig)), raising=False)
    assert parity.kill_port_owners(24560) is True
    assert killed == [(1048, parity.signal.SIGTERM), (1048, parity.FORCE_KILL_SIGNAL)]


def test_unknown_listener_is_never_killed(monkeypatch, capsys):
    monkeypatch.setattr(parity, "is_windows", lambda: False)
    monkeypatch.setattr(parity, "listening_pids", lambda port: [9999])
    monkeypatch.setattr(parity, "pid_owned_by_server", lambda pid: False)
    monkeypatch.setattr(parity.os, "kill", lambda pid, sig: pytest.fail("foreign listener killed"), raising=False)
    assert parity.kill_port_owners(24560) is False
    err = capsys.readouterr().err
    assert "outside this run" in err and "9999" in err


def test_handler_waits_grace_before_sigkill(monkeypatch):
    events: list[tuple[str, int, float]] = []
    started = time.monotonic()

    class IgnoringServer(FakeProcess):
        def terminate(self):
            pass

    process = IgnoringServer()
    process.pid = 123

    def fake_killpg(pgid, sig):
        events.append(("signal", sig, time.monotonic() - started))
        if sig == parity.FORCE_KILL_SIGNAL:
            process.returncode = -9

    monkeypatch.setattr(parity, "is_windows", lambda: False)
    monkeypatch.setattr(parity.os, "killpg", fake_killpg, raising=False)
    parity.LIVE_PROCESSES.add(process)
    parity.LIVE_GROUPS[process.pid] = 5555
    try:
        parity.terminate_process_groups(grace=0.2)
    finally:
        parity.LIVE_PROCESSES.discard(process)
        parity.LIVE_GROUPS.pop(process.pid, None)
    kinds = [(sig, elapsed) for kind, sig, elapsed in events]
    assert kinds[0][0] == parity.signal.SIGTERM
    assert kinds[-1][0] == parity.FORCE_KILL_SIGNAL
    assert kinds[-1][1] - kinds[0][1] >= 0.15


def test_external_listener_does_not_change_the_exit_code(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(parity, "execute_config", lambda *args, **kwargs: 0)
    monkeypatch.setattr(parity, "owned_listeners", lambda port: ([], [9999]))
    code = parity.main(["rehearse", "--stub", "--data-dir", str(tmp_path / "data")])
    assert code == 0
    assert "warning:" in capsys.readouterr().err


def test_manifest_records_report_identity(tmp_path, monkeypatch):
    no_http(monkeypatch)
    plan = make_plan(tmp_path)
    assert parity.execute_config(plan, make_options(tmp_path), FakeExecutor()) == 0
    manifest = read_manifest(plan)
    reports = manifest["compare_reports"]
    assert set(reports) == {"md", "json"}
    for entry in reports.values():
        assert entry["size"] > 0
        assert entry["mtime_ns"] > 0
        assert "inode" in entry


def test_compare_crash_with_rc3_reports_no_output(tmp_path, monkeypatch, capsys):
    no_http(monkeypatch)
    plan = make_plan(tmp_path)
    executor = FakeExecutor(compare_rc=3, compare_writes=False)
    assert parity.execute_config(plan, make_options(tmp_path), executor) == 2
    manifest = read_manifest(plan)
    assert "crashed with 3 (no output)" in manifest["abort_reason"]
    assert manifest["compare_exit"] is None


def test_second_plan_does_not_inherit_the_first_compare_exit(tmp_path):
    options = make_options(tmp_path)
    plan = make_plan(tmp_path)
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    plan.data_dir.mkdir(parents=True)
    records = [
        {
            "stage": "compare",
            "backend": None,
            "argv": ["python", "compare", "--out", str(other_dir / "compare.md")],
            "returncode": 0,
            "env": {},
            "cwd": "",
        },
        {
            "stage": "compare",
            "backend": None,
            "argv": ["python", "compare", "--out", str(other_dir / "compare.json")],
            "returncode": 0,
            "env": {},
            "cwd": "",
        },
    ]
    manifest = parity.build_manifest(
        plan,
        {"mrv1": parity.BackendRun("mrv1")},
        records,
        {"exit": None, "verdict": None, "md": None, "json": None, "reports": {}},
        "compare aborted",
        options,
    )
    assert manifest["compare_exit"] is None
    assert "did not run for this plan" in manifest["compare_note"]


def test_default_stub_token_count_is_pinned():
    args = parity.build_parser().parse_args(["rehearse", "--stub", "--data-dir", "x"])
    assert args.stub_tokens == 32
    assert args.stub_tokens == len(parity.STUB_PROMPT_TOKENS) == parity.MAX_STUB_TOKENS


def test_handle_termination_sigint_records_and_aborts(monkeypatch):
    cleaned: list[bool] = []
    monkeypatch.setattr(parity, "terminate_process_groups", lambda: cleaned.append(True))
    with pytest.raises(parity.SignalAbort) as excinfo:
        parity.handle_termination(parity.signal.SIGINT, None)
    assert parity.SIGNAL_ABORT_REASON == "terminated by SIGINT"
    assert "SIGINT" in str(excinfo.value)
    assert cleaned == [True]
