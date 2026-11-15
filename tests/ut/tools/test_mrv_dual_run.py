import ast
import hashlib
import json
import shlex
import subprocess
from pathlib import Path

import pytest
import yaml

from tools import mrv_dual_run as dual

REPO_ROOT = Path(__file__).resolve().parents[3]
ENV_VAR = dual.BACKEND_ENV_VAR
GLM_DIR = "tests/e2e/cases/models/configs/GLM"
FEATURE_PCP = "tests/e2e/cases/features/context_parallel/glm-5.2-W4A8C8-DSpark-PCP.yaml"
FEATURE_PP = "tests/e2e/cases/features/pp/GLM-5.3-W8A8C8-A3-200k-PP-PD.yaml"

EXPECTED_PINS = {
    FEATURE_PCP: (9, "1"),
    FEATURE_PP: (72, "1"),
    f"{GLM_DIR}/GLM-5.1-W8A8C8-128k-1k-90-50-PD.yaml": (54, "1"),
    f"{GLM_DIR}/GLM-5.1-W8A8C8-A3_128k_90_50.yaml": (7, "1"),
    f"{GLM_DIR}/GLM-5.1-W8A8C8-A3_198k_function.yaml": (8, "1"),
    f"{GLM_DIR}/GLM-5.2-W4A4C8-DSpark-GQA-A5-V2.yaml": (19, "1"),
    f"{GLM_DIR}/GLM-5.2-W4A8-DSpark-A3-V2.yaml": (15, "1"),
    f"{GLM_DIR}/GLM-5.2-W4A8-MTP-A3-V2.yaml": (15, "1"),
    f"{GLM_DIR}/GLM-5.2-W4A8C8-SFA-DCP.yaml": (23, "1"),
    f"{GLM_DIR}/GLM-5.2-W8A8C8-A3_128k-static-super-kernels.yaml": (8, "1"),
    f"{GLM_DIR}/GLM-5.2-W8A8C8-MTP-A5-dual-nodes.yaml": (29, "0"),
    f"{GLM_DIR}/GLM-5.2-W8A8C8-layerwise-PD.yaml": (59, "0"),
    f"{GLM_DIR}/GLM-5.3-Flash-W8A8-130k-PD.yaml": (41, "1"),
    f"{GLM_DIR}/GLM5_1_W4A4_A5.yaml": (9, "1"),
    f"{GLM_DIR}/GLM5_1-W8A8-A2-single-node.yaml": (49, "1"),
}
EXPECTED_CARRIERS = {
    "test_cases": {
        FEATURE_PCP,
        f"{GLM_DIR}/GLM-5.2-W4A4C8-DSpark-GQA-A5-V2.yaml",
        f"{GLM_DIR}/GLM-5.2-W4A8-DSpark-A3-V2.yaml",
        f"{GLM_DIR}/GLM-5.2-W4A8-MTP-A3-V2.yaml",
        f"{GLM_DIR}/GLM-5.2-W4A8C8-SFA-DCP.yaml",
        f"{GLM_DIR}/GLM5_1_W4A4_A5.yaml",
        f"{GLM_DIR}/GLM5_1-W8A8-A2-single-node.yaml",
    },
    "deployment": {
        f"{GLM_DIR}/GLM-5.1-W8A8C8-A3_128k_90_50.yaml",
        f"{GLM_DIR}/GLM-5.1-W8A8C8-A3_198k_function.yaml",
        f"{GLM_DIR}/GLM-5.2-W8A8C8-A3_128k-static-super-kernels.yaml",
        f"{GLM_DIR}/GLM-5.2-W8A8C8-MTP-A5-dual-nodes.yaml",
    },
    "external_templates": {
        f"{GLM_DIR}/GLM-5.1-W8A8C8-128k-1k-90-50-PD.yaml",
        f"{GLM_DIR}/GLM-5.2-W8A8C8-layerwise-PD.yaml",
        f"{GLM_DIR}/GLM-5.3-Flash-W8A8-130k-PD.yaml",
        FEATURE_PP,
    },
}

INLINE_CASE = """test_cases:
  - name: "case-a"
    model: "org/model"
    envs:
      VLLM_USE_V2_MODEL_RUNNER: "1"
    server_cmd:
      - "--port"
      - "$SERVER_PORT"
"""
DEPLOYMENT_CONFIG = """test_name: "multi-node"
model: "org/model"
env_common: &env_common
  VLLM_USE_V2_MODEL_RUNNER: "0"
deployment:
  - envs:
      <<: *env_common
    server_cmd: >
      vllm serve org/model
      --port 8000
"""
ANCHOR_CHAIN = """test_name: "external"
model: "org/model"
env_common: &env_common
  VLLM_USE_V2_MODEL_RUNNER: "1"
env_prefill: &env_prefill
  <<: *env_common
  VLLM_USE_FASTOKENS: "1"
templates:
  - node_index: 0
    envs: *env_prefill
    server_cmd_template:
      - --host
      - 0.0.0.0
"""
QUOTED_FIXTURE = """test_name: "quoted"
env_common: &env_common
  VLLM_USE_V2_MODEL_RUNNER: "1"
  SERVER_PORT: "8080"
deployment:
  - envs:
      <<: *env_common
    server_cmd: >
      vllm serve org/model
"""
SINGLE_QUOTED_FIXTURE = """test_name: "single"
env_common: &env_common
  VLLM_USE_V2_MODEL_RUNNER: '0'
  SERVER_PORT: '8080'
deployment:
  - envs:
      <<: *env_common
    server_cmd: >
      vllm serve org/model
"""
COMMENTED_FIXTURE = """test_name: "commented"
env_common: &env_common
  VLLM_USE_V2_MODEL_RUNNER: "0"  # pinned, keep me
  SERVER_PORT: "8080"
deployment:
  - envs:
      <<: *env_common
    server_cmd: >
      vllm serve org/model
"""
NO_COMMAND_FIXTURE = """test_cases:
  - name: "no-command"
    model: "org/model"
    envs:
      VLLM_USE_V2_MODEL_RUNNER: "1"
"""
LOG_MRV1 = "INFO (EngineCore_0 pid=1) VLLM_USE_V2_MODEL_RUNNER=0 is set; using Model Runner V1.\n"
LOG_MRV2 = "INFO (EngineCore_0 pid=1) VLLM_USE_V2_MODEL_RUNNER=1 is set; using Model Runner V2.\n"
LOG_FALLBACK = "WARNING Model Runner V2 does not yet support draft_model; using the V1 model runner instead.\n"
LOG_UNSET = "INFO (EngineCore_0 pid=1) VLLM_USE_V2_MODEL_RUNNER is unset; using Model Runner V2 by default.\n"
LOG_INCONSISTENT = "INFO (EngineCore_0 pid=1) VLLM_USE_V2_MODEL_RUNNER=0 is set; using Model Runner V2.\n"
FLOW_FIXTURE = """test_cases:
  - name: "flow"
    envs: {VLLM_USE_V2_MODEL_RUNNER: "1"}
    server_cmd:
      - "--port"
"""


def write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def fake_repo(tmp_path: Path, name: str, config: str) -> tuple[Path, Path]:
    root = tmp_path / name
    config_path = write_text(root / GLM_DIR / "fixture.yaml", config)
    return root, config_path


def cli_json(capsys, argv: list[str]) -> dict:
    assert dual.main(argv) == dual.EXIT_OK
    return json.loads(capsys.readouterr().out)


def test_discover_finds_all_pinned_glm_configs():
    configs = dual.discover_configs(REPO_ROOT)
    assert len(configs) == 15


def test_discover_paths_match_the_audited_matrix():
    discovered = {info.rel for info in dual.discover_configs(REPO_ROOT)}
    assert discovered == set(EXPECTED_PINS)


def test_discover_reports_line_and_value_for_every_pin():
    assert {info.rel: (info.pin.line, info.pin.value) for info in dual.discover_configs(REPO_ROOT)} == EXPECTED_PINS


def test_discover_carrier_counts_match_the_matrix(capsys):
    payload = cli_json(capsys, ["discover", "--repo-root", str(REPO_ROOT)])
    assert payload["total"] == 15
    assert payload["counts"] == {"test_cases": 7, "deployment": 4, "external_templates": 4}


def test_discover_assigns_every_config_to_the_expected_carrier():
    by_carrier = {carrier: set() for carrier in dual.CARRIERS}
    for info in dual.discover_configs(REPO_ROOT):
        by_carrier[info.carrier].add(info.rel)
    assert by_carrier == EXPECTED_CARRIERS


def test_discover_classifies_the_single_quoted_layerwise_config():
    info = next(item for item in dual.discover_configs(REPO_ROOT) if item.rel.endswith("layerwise-PD.yaml"))
    assert (info.carrier, info.pin.value, info.pin.line) == ("external_templates", "0", 59)


def test_discover_includes_the_a2_single_node_config():
    info = next(
        item for item in dual.discover_configs(REPO_ROOT) if item.rel.endswith("GLM5_1-W8A8-A2-single-node.yaml")
    )
    assert (info.carrier, info.pin.value) == ("test_cases", "1")


def test_discover_ignores_a_commented_pin(tmp_path):
    root, _ = fake_repo(
        tmp_path,
        "commented",
        """test_cases:
  - name: "commented"
    envs:
      # VLLM_USE_V2_MODEL_RUNNER: "1"
    server_cmd:
      - "--port"
""",
    )
    assert dual.discover_configs(root) == []


def test_carrier_of_an_inline_test_cases_env_block():
    assert dual.classify_carrier(INLINE_CASE) == "test_cases"


def test_carrier_of_a_deployment_env_common_anchor():
    assert dual.classify_carrier(DEPLOYMENT_CONFIG) == "deployment"


def test_carrier_of_an_anchor_chain_merged_by_templates():
    assert dual.classify_carrier(ANCHOR_CHAIN) == "external_templates"


def test_carrier_of_an_unmerged_anchor_raises():
    orphan = 'env_common: &env_common\n  VLLM_USE_V2_MODEL_RUNNER: "1"\nrouting:\n  - node_index: 0\n'
    with pytest.raises(dual.InputError, match="not merged"):
        dual.classify_carrier(orphan)


def test_rewrite_of_a_double_quoted_value_changes_one_line():
    rewritten, changed = dual.rewrite_pin_value(QUOTED_FIXTURE, "0")
    assert changed == [3]
    assert dual.find_pins(rewritten)[0].value == "0"
    assert rewritten.splitlines()[:2] == QUOTED_FIXTURE.splitlines()[:2]
    assert rewritten.splitlines()[3:] == QUOTED_FIXTURE.splitlines()[3:]
    assert dual.verify_rewritten(QUOTED_FIXTURE, rewritten, "0", True) == [3]


def test_rewrite_of_a_single_quoted_value_preserves_the_quote():
    rewritten, changed = dual.rewrite_pin_value(SINGLE_QUOTED_FIXTURE, "1")
    assert changed == [3]
    assert "  VLLM_USE_V2_MODEL_RUNNER: '1'" in rewritten.splitlines()
    assert dual.find_pins(rewritten)[0].value == "1"


def test_rewrite_keeps_the_trailing_comment_and_indent():
    rewritten, _ = dual.rewrite_pin_value(COMMENTED_FIXTURE, "1")
    assert '  VLLM_USE_V2_MODEL_RUNNER: "1"  # pinned, keep me' in rewritten.splitlines()


def test_rewrite_reports_every_pin_line_when_a_file_has_two():
    two_pins = INLINE_CASE + '  - name: "case-b"\n    envs:\n      VLLM_USE_V2_MODEL_RUNNER: "1"\n'
    _, changed = dual.rewrite_pin_value(two_pins, "0")
    assert changed == [5, 11]


def test_rewrite_cli_writes_a_verified_copy(tmp_path):
    source = REPO_ROOT / GLM_DIR / "GLM-5.2-W8A8C8-layerwise-PD.yaml"
    before = source.read_text(encoding="utf-8")
    destination = tmp_path / "copy"
    assert dual.main(["rewrite", "--src", str(source), "--dest", str(destination), "--value", "1"]) == dual.EXIT_OK
    copy = destination / source.name
    assert dual.find_pins(copy.read_text(encoding="utf-8"))[0].value == "1"
    assert source.read_text(encoding="utf-8") == before


def test_rewrite_cli_refuses_a_config_without_the_key(tmp_path):
    root, config = fake_repo(tmp_path, "no-key", 'test_name: "plain"\nmodel: "org/model"\n')
    assert dual.main(["rewrite", "--src", str(config), "--dest", str(tmp_path / "out"), "--value", "0"]) == 2
    assert not (tmp_path / "out" / config.name).exists()


def test_rewrite_cli_refuses_a_copy_that_would_not_change(tmp_path):
    root, config = fake_repo(tmp_path, "no-change", QUOTED_FIXTURE)
    assert dual.main(["rewrite", "--src", str(config), "--dest", str(tmp_path / "out"), "--value", "1"]) == 2


def test_rewrite_cli_refuses_to_overwrite_the_source(tmp_path):
    root, config = fake_repo(tmp_path, "overwrite", QUOTED_FIXTURE)
    before = config.read_text(encoding="utf-8")
    assert dual.main(["rewrite", "--src", str(config), "--dest", str(config.parent), "--value", "0"]) == 2
    assert config.read_text(encoding="utf-8") == before


def test_plan_print_only_covers_all_configs_with_two_command_sets(capsys):
    dest = REPO_ROOT / "tests/e2e/cases/.should-not-exist"
    payload = cli_json(capsys, ["plan", "--print-only", "--dest", str(dest)])
    assert len(payload["configs"]) == 15
    assert not dest.exists()
    for entry in payload["configs"]:
        assert set(entry["targets"]) == {"mrv1", "mrv2"}
        for target, state in entry["targets"].items():
            assert state["value"] == dual.BACKEND_VALUES[target]
            assert state["dest"] is None
            assert state["commands"]
            assert all(command["display"].startswith("vllm serve") for command in state["commands"])


def test_plan_marks_no_change_needed_per_target(capsys):
    payload = cli_json(capsys, ["plan", "--print-only", "--dest", str(REPO_ROOT / "ignored")])
    actions = {target: [entry["targets"][target]["action"] for entry in payload["configs"]] for target in dual.BACKENDS}
    assert actions["mrv1"].count("rewrite") == 13
    assert actions["mrv1"].count("no-change-needed") == 2
    assert actions["mrv2"].count("rewrite") == 2
    assert actions["mrv2"].count("no-change-needed") == 13
    unchanged = {
        entry["path"] for entry in payload["configs"] if entry["targets"]["mrv2"]["action"] == "no-change-needed"
    }
    assert unchanged == {rel for rel, (_, value) in EXPECTED_PINS.items() if value == "1"}
    rewritten = {entry["path"] for entry in payload["configs"] if entry["targets"]["mrv2"]["action"] == "rewrite"}
    assert rewritten == {rel for rel, (_, value) in EXPECTED_PINS.items() if value == "0"}


def test_plan_writes_two_trees_and_never_touches_the_repo(tmp_path):
    before = {rel: (REPO_ROOT / rel).read_text(encoding="utf-8") for rel in EXPECTED_PINS}
    dest = tmp_path / "dual"
    assert dual.main(["plan", "--dest", str(dest)]) == dual.EXIT_OK
    copies = sorted(dest.rglob("*.yaml"))
    assert len(copies) == 30
    for rel, (pin_line, source_value) in EXPECTED_PINS.items():
        source_lines = before[rel].splitlines(keepends=True)
        for target, expected in dual.BACKEND_VALUES.items():
            copy_text = (dest / target / rel).read_text(encoding="utf-8")
            assert dual.find_pins(copy_text)[0].value == expected
            copy_lines = copy_text.splitlines(keepends=True)
            changed = [number for number, (old, new) in enumerate(zip(source_lines, copy_lines), start=1) if old != new]
            assert changed == ([] if source_value == expected else [pin_line])
    for rel in EXPECTED_PINS:
        assert (REPO_ROOT / rel).read_text(encoding="utf-8") == before[rel]


def test_plan_refuses_a_config_without_a_serve_command(tmp_path, capsys):
    root, config = fake_repo(tmp_path, "no-command", NO_COMMAND_FIXTURE)
    assert dual.main(["plan", "--repo-root", str(root), "--dest", str(tmp_path / "out")]) == 2
    assert "no server_cmd" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_plan_refuses_a_dest_inside_the_case_tree():
    dest = REPO_ROOT / "tests/e2e/cases/tmp-dual"
    assert dual.main(["plan", "--dest", str(dest)]) == 2
    assert not (dest / "mrv1").exists()


def test_plan_only_filters_to_one_config(capsys):
    payload = cli_json(capsys, ["plan", "--print-only", "--dest", str(REPO_ROOT / "ignored"), "--only", FEATURE_PP])
    assert [entry["path"] for entry in payload["configs"]] == [FEATURE_PP]


def run_verify(tmp_path, capsys, log_text: str, expect: str) -> tuple[int, str, str]:
    log = write_text(tmp_path / "server.log", log_text)
    code = dual.main(["verify-log", "--log", str(log), "--expect", expect])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_verify_log_explicit_v2_passes_for_mrv2(tmp_path, capsys):
    code, out, err = run_verify(tmp_path, capsys, LOG_MRV2, "mrv2")
    assert (code, err) == (0, "")
    assert "VLLM_USE_V2_MODEL_RUNNER=1 is set" in out


def test_verify_log_explicit_v2_fails_for_mrv1(tmp_path, capsys):
    code, _, err = run_verify(tmp_path, capsys, LOG_MRV2, "mrv1")
    assert code == 1
    assert "expected V1" in err


def test_verify_log_unset_default_v2_fails_for_mrv2(tmp_path, capsys):
    """Regression: 'is unset; using Model Runner V2 by default' used to pass == false green."""
    code, _, err = run_verify(tmp_path, capsys, LOG_UNSET, "mrv2")
    assert code == 1
    assert "NOT applied" in err


def test_verify_log_unset_default_v2_fails_for_mrv1(tmp_path, capsys):
    code, _, err = run_verify(tmp_path, capsys, LOG_UNSET, "mrv1")
    assert code == 1
    assert "NOT applied" in err


def test_verify_log_blacklist_fallback_fails_instead_of_missing(tmp_path, capsys):
    code, _, err = run_verify(tmp_path, capsys, LOG_FALLBACK, "mrv1")
    assert code == 1
    assert "fallback" in err and "NOT applied" in err


def test_verify_log_conflicting_explicit_lines_fail(tmp_path, capsys):
    code, _, err = run_verify(tmp_path, capsys, LOG_MRV1 + LOG_MRV2, "mrv1")
    assert code == 1
    assert "contradictory" in err


def test_verify_log_empty_log_is_an_input_error(tmp_path, capsys):
    code, _, err = run_verify(tmp_path, capsys, "nothing to see here\n", "mrv2")
    assert code == 2
    assert "not a pass" in err


def test_verify_log_fallback_line_is_not_read_as_explicit():
    modes = [item["mode"] for item in dual.scan_runner_log(LOG_FALLBACK.splitlines())]
    assert modes == ["fallback"]
    assert all(item["mode"] != "explicit" for item in dual.scan_runner_log(LOG_FALLBACK.splitlines()))


def test_verify_log_inconsistent_explicit_value_fails(tmp_path, capsys):
    code, _, err = run_verify(tmp_path, capsys, LOG_INCONSISTENT, "mrv2")
    assert code == 1
    assert "contradicts" in err


def test_verify_log_missing_file_is_an_input_error(tmp_path):
    assert dual.main(["verify-log", "--log", str(tmp_path / "absent.log"), "--expect", "mrv1"]) == 2


def test_manifest_unknown_fields_stay_null_and_the_tag_is_read(tmp_path, capsys):
    out = tmp_path / "manifest.json"
    assert dual.main(["manifest", "--out", str(out), "--repo-root", str(REPO_ROOT), "--repo-commit", "abc123"]) == 0
    manifest = json.loads(out.read_text(encoding="utf-8"))
    assert tuple(manifest) == dual.MANIFEST_FIELDS
    assert manifest["vllm_release_tag"] == "v0.33.0"
    assert manifest["repo_commit"] == "abc123"
    for field in (
        "image_tag",
        "cann_version",
        "torch_npu_version",
        "model_id",
        "model_revision",
        "hardware",
        "graph_mode",
        "config_yaml",
        "backend_expected",
        "server_log_sha256",
        "commands",
    ):
        assert manifest[field] is None
    assert manifest["generated_at"].endswith("Z")


def test_manifest_records_provided_values_and_the_log_hash(tmp_path, capsys):
    log = write_text(tmp_path / "server.log", LOG_MRV2)
    out = tmp_path / "manifest.json"
    argv = [
        "manifest",
        "--out",
        str(out),
        "--repo-root",
        str(REPO_ROOT),
        "--repo-commit",
        "deadbeef",
        "--image-tag",
        "img:1",
        "--cann-version",
        "8.2.RC1",
        "--torch-npu-version",
        "2.7.1",
        "--model-id",
        "Eco-Tech/GLM-5.1",
        "--model-revision",
        "main",
        "--hardware",
        "A3-16x16",
        "--graph-mode",
        "eager",
        "--config-yaml",
        "GLM5_1_W4A4_A5.yaml",
        "--backend-expected",
        "mrv2",
        "--server-log",
        str(log),
    ]
    assert dual.main(argv) == 0
    manifest = json.loads(out.read_text(encoding="utf-8"))
    assert manifest["image_tag"] == "img:1" and manifest["cann_version"] == "8.2.RC1"
    assert manifest["backend_expected"] == "mrv2"
    assert manifest["server_log_sha256"] == hashlib.sha256(log.read_bytes()).hexdigest()


def test_manifest_requires_a_json_out(tmp_path):
    out = tmp_path / "manifest.md"
    assert dual.main(["manifest", "--out", str(out), "--repo-root", str(REPO_ROOT)]) == 2
    assert not out.exists()


def test_manifest_refuses_a_missing_server_log(tmp_path):
    assert dual.main(["manifest", "--out", str(tmp_path / "m.json"), "--server-log", str(tmp_path / "absent.log")]) == 2


def test_tool_imports_only_the_standard_library_and_the_regex_whitelist():
    """The repository lint bans 'import re'; 'import regex as re' is the allowed spelling."""
    allowed = {
        "__future__",
        "argparse",
        "hashlib",
        "json",
        "regex",
        "shlex",
        "subprocess",
        "sys",
        "dataclasses",
        "datetime",
        "pathlib",
        "typing",
    }
    tree = ast.parse(Path(dual.__file__).read_text(encoding="utf-8"))
    modules = {node.module.split(".")[0] for node in tree.body if isinstance(node, ast.ImportFrom) and node.module}
    modules |= {alias.name.split(".")[0] for node in tree.body if isinstance(node, ast.Import) for alias in node.names}
    assert modules <= allowed
    assert not modules & {"re", "vllm", "torch", "yaml", "vllm_ascend"}


def test_ut_run_leaves_the_yaml_tree_clean():
    completed = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--", "tests/e2e/cases"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    assert completed.stdout.strip() == ""


def test_discover_refuses_a_flow_style_pin(tmp_path, capsys):
    """A flow mapping mentions the variable but cannot be read: fail, never skip silently."""
    root, _ = fake_repo(tmp_path, "flow", FLOW_FIXTURE)
    assert dual.main(["discover", "--repo-root", str(root)]) == 2
    err = capsys.readouterr().err
    assert "cannot be read" in err
    assert "fixture.yaml" in err and "line(s) 3" in err


def test_stray_pin_lines_ignores_comments():
    assert dual.stray_pin_lines('  # VLLM_USE_V2_MODEL_RUNNER: "1"\n') == []
    assert dual.stray_pin_lines('    envs: {VLLM_USE_V2_MODEL_RUNNER: "1"}\n') == [1]


def test_command_record_returns_argv_when_the_display_round_trips():
    record = dual.command_record("node-a", "vllm serve org/model --port 8000 --seed 1024", "test_cases[0].server_cmd")
    assert record["argv"] == ["vllm", "serve", "org/model", "--port", "8000", "--seed", "1024"]
    assert shlex.join(record["argv"]) == record["display"]
    assert record["requires_runner_render"] is False
    assert record["placeholders"] == []


def test_command_record_refuses_argv_for_a_json_argument():
    record = dual.command_record("node-a", 'vllm serve org/model --cudagraph-mode {"mode": "FULL"}', "fixture")
    assert record["argv"] is None
    assert record["requires_runner_render"] is True
    assert record["placeholders"] == []


def test_command_record_refuses_argv_and_lists_placeholders():
    record = dual.command_record("node-a", "vllm serve org/model --port ${PORT} --tp ${TP_SIZE}", "fixture")
    assert record["argv"] is None
    assert record["placeholders"] == ["PORT", "TP_SIZE"]
    assert record["requires_runner_render"] is True


def test_plan_keeps_test_cases_json_arguments_as_single_tokens(capsys):
    payload = cli_json(capsys, ["plan", "--print-only", "--dest", str(REPO_ROOT / "ignored")])
    entry = next(item for item in payload["configs"] if item["path"].endswith("SFA-DCP.yaml"))
    command = entry["targets"]["mrv1"]["commands"][0]
    tokens = shlex.split(command["display"])
    case = yaml.safe_load((REPO_ROOT / entry["path"]).read_text(encoding="utf-8"))["test_cases"][0]
    assert tokens == ["vllm", "serve", case["model"], *case["server_cmd"]]
    assert len(tokens) == 45
    assert command["argv"] == tokens
    assert command["requires_runner_render"] is True


def test_a2_single_node_display_has_thirty_shell_tokens(capsys):
    payload = cli_json(capsys, ["plan", "--print-only", "--dest", str(REPO_ROOT / "ignored")])
    entry = next(item for item in payload["configs"] if item["path"].endswith("GLM5_1-W8A8-A2-single-node.yaml"))
    command = entry["targets"]["mrv1"]["commands"][0]
    tokens = shlex.split(command["display"])
    case = yaml.safe_load((REPO_ROOT / entry["path"]).read_text(encoding="utf-8"))["test_cases"][0]
    assert tokens == ["vllm", "serve", case["model"], *case["server_cmd"]]
    assert len(tokens) == 30
    assert command["argv"] == tokens
    assert tokens.count("--tensor-parallel-size") == 1
    assert any(token.startswith('{"cudagraph_mode"') and " " in token for token in tokens)


def test_deployment_and_external_display_behavior_is_unchanged(capsys):
    payload = cli_json(capsys, ["plan", "--print-only", "--dest", str(REPO_ROOT / "ignored")])
    for name in ("GLM-5.2-W8A8C8-MTP-A5-dual-nodes.yaml", "GLM-5.1-W8A8C8-128k-1k-90-50-PD.yaml"):
        command = next(item for item in payload["configs"] if item["path"].endswith(name))["targets"]["mrv1"][
            "commands"
        ][0]
        assert command["argv"] is None
        assert command["requires_runner_render"] is True
        assert command["display"].startswith("vllm serve")


def test_plan_exposes_placeholders_for_the_four_external_configs(capsys):
    payload = cli_json(capsys, ["plan", "--print-only", "--dest", str(REPO_ROOT / "ignored")])
    external = [entry for entry in payload["configs"] if entry["carrier"] == "external_templates"]
    assert len(external) == 4
    for entry in external:
        for command in entry["targets"]["mrv1"]["commands"]:
            assert command["placeholders"]
            assert command["requires_runner_render"] is True
            assert command["argv"] is None


def test_plan_never_claims_an_argv_that_does_not_round_trip(capsys):
    payload = cli_json(capsys, ["plan", "--print-only", "--dest", str(REPO_ROOT / "ignored")])
    for entry in payload["configs"]:
        for state in entry["targets"].values():
            for command in state["commands"]:
                assert set(command) == {
                    "node",
                    "display",
                    "argv",
                    "requires_runner_render",
                    "placeholders",
                    "source",
                    "model",
                    "notes",
                }
                if command["argv"] is None:
                    assert command["requires_runner_render"] is True
                else:
                    assert shlex.join(command["argv"]) == " ".join(command["display"].split())
