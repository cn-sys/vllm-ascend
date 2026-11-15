import ast
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

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
}
EXPECTED_CARRIERS = {
    "test_cases": {
        FEATURE_PCP,
        f"{GLM_DIR}/GLM-5.2-W4A4C8-DSpark-GQA-A5-V2.yaml",
        f"{GLM_DIR}/GLM-5.2-W4A8-DSpark-A3-V2.yaml",
        f"{GLM_DIR}/GLM-5.2-W4A8-MTP-A3-V2.yaml",
        f"{GLM_DIR}/GLM-5.2-W4A8C8-SFA-DCP.yaml",
        f"{GLM_DIR}/GLM5_1_W4A4_A5.yaml",
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


def test_discover_finds_exactly_fourteen_configs():
    configs = dual.discover_configs(REPO_ROOT)
    assert len(configs) == 14


def test_discover_paths_match_the_audited_matrix():
    discovered = {info.rel for info in dual.discover_configs(REPO_ROOT)}
    assert discovered == set(EXPECTED_PINS)


def test_discover_reports_line_and_value_for_every_pin():
    assert {info.rel: (info.pin.line, info.pin.value) for info in dual.discover_configs(REPO_ROOT)} == EXPECTED_PINS


def test_discover_carrier_counts_are_six_four_four(capsys):
    payload = cli_json(capsys, ["discover", "--repo-root", str(REPO_ROOT)])
    assert payload["total"] == 14
    assert payload["counts"] == {"test_cases": 6, "deployment": 4, "external_templates": 4}


def test_discover_assigns_every_config_to_the_expected_carrier():
    by_carrier = {carrier: set() for carrier in dual.CARRIERS}
    for info in dual.discover_configs(REPO_ROOT):
        by_carrier[info.carrier].add(info.rel)
    assert by_carrier == EXPECTED_CARRIERS


def test_discover_classifies_the_single_quoted_layerwise_config():
    info = next(item for item in dual.discover_configs(REPO_ROOT) if item.rel.endswith("layerwise-PD.yaml"))
    assert (info.carrier, info.pin.value, info.pin.line) == ("external_templates", "0", 59)


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
    assert len(payload["configs"]) == 14
    assert not dest.exists()
    for entry in payload["configs"]:
        assert set(entry["targets"]) == {"mrv1", "mrv2"}
        for target, state in entry["targets"].items():
            assert state["value"] == dual.BACKEND_VALUES[target]
            assert state["dest"] is None
            assert state["commands"]
            assert all(command["command"].startswith("vllm serve") for command in state["commands"])


def test_plan_marks_no_change_needed_per_target(capsys):
    payload = cli_json(capsys, ["plan", "--print-only", "--dest", str(REPO_ROOT / "ignored")])
    actions = {target: [entry["targets"][target]["action"] for entry in payload["configs"]] for target in dual.BACKENDS}
    assert actions["mrv1"].count("rewrite") == 12
    assert actions["mrv1"].count("no-change-needed") == 2
    assert actions["mrv2"].count("rewrite") == 2
    assert actions["mrv2"].count("no-change-needed") == 12
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
    assert len(copies) == 28
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


@pytest.mark.parametrize(
    ("log_text", "expected", "code"),
    [
        (LOG_MRV1, "mrv1", 0),
        (LOG_MRV2, "mrv2", 0),
        (LOG_MRV1, "mrv2", 1),
        (LOG_FALLBACK, "mrv1", 2),
        (LOG_MRV1 + LOG_MRV2, "mrv1", 1),
    ],
)
def test_verify_log_three_state_matrix(tmp_path, capsys, log_text, expected, code):
    log = write_text(tmp_path / "server.log", log_text)
    assert dual.main(["verify-log", "--log", str(log), "--expect", expected]) == code
    captured = capsys.readouterr()
    if code == 0:
        assert "using Model Runner" in captured.out
    else:
        assert captured.err


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


def test_tool_imports_only_the_standard_library():
    allowed = {
        "__future__",
        "argparse",
        "hashlib",
        "json",
        "re",
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
    assert not modules & {"vllm", "torch", "yaml", "vllm_ascend"}


def test_ut_run_leaves_the_yaml_tree_clean():
    completed = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--", "tests/e2e/cases"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    assert completed.stdout.strip() == ""
