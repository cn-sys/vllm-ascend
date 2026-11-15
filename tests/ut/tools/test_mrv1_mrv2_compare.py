import json
from pathlib import Path

import pytest

from tools import mrv1_mrv2_compare as compare_tool

REPO_ROOT = Path(__file__).resolve().parents[3]
PROMPT_SET = REPO_ROOT / "tools" / "mrv1_mrv2_prompts.jsonl"
ENV_VAR = compare_tool.BACKEND_ENV_VAR


def make_record(
    record_id: str,
    tokens: list[str],
    logprobs: list[float | None] | None = None,
    backend: str = "mrv1",
    temperature: float = 0.0,
    parsed_ok: bool = True,
    prompt: str = "same prompt",
) -> dict:
    if logprobs is None:
        logprobs = [-0.1 * (position + 1) for position in range(len(tokens))]
    return {
        "id": record_id,
        "backend": backend,
        "prompt_sha1": compare_tool.prompt_sha1(prompt),
        "tokens": list(tokens),
        "logprobs": list(logprobs),
        "finish_reason": "length",
        "parsed_ok": parsed_ok,
        "temperature": temperature,
    }


def write_jsonl(path: Path, records: list[dict]) -> Path:
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    return path


def write_pair(tmp_path: Path, records_a: list[dict], records_b: list[dict]) -> tuple[Path, Path]:
    return write_jsonl(tmp_path / "mrv1.jsonl", records_a), write_jsonl(tmp_path / "mrv2.jsonl", records_b)


def run_compare(tmp_path: Path, path_a: Path, path_b: Path, *extra: str) -> tuple[int, str]:
    out = tmp_path / "report.md"
    code = compare_tool.main(["compare", "--a", str(path_a), "--b", str(path_b), "--out", str(out), *extra])
    report = out.read_text(encoding="utf-8") if out.exists() else ""
    return code, report


def test_identical_runs_pass(tmp_path: Path):
    tokens = ["Hel", "lo", " wor", "ld"]
    logprobs = [-0.1, -0.2, -0.3, -0.4]
    records_a = [make_record("p1", tokens, logprobs, backend="mrv1")]
    records_b = [make_record("p1", tokens, logprobs, backend="mrv2")]

    code, report = run_compare(tmp_path, *write_pair(tmp_path, records_a, records_b))

    assert code == compare_tool.EXIT_OK
    assert "## Verdict: PASS" in report
    assert "| records compared | 1 |" in report
    assert "| max logprob delta | 0.000000 |" in report


def test_summary_metrics_for_identical_runs(tmp_path: Path):
    path_a, path_b = write_pair(
        tmp_path,
        [make_record("p1", ["a", "b"], backend="mrv1")],
        [make_record("p1", ["a", "b"], backend="mrv2")],
    )

    report = compare_tool.compare_files(path_a, path_b)
    metrics = compare_tool.summarize(report)

    assert report.passed is True
    assert metrics["records_compared"] == 1
    assert metrics["exact_token_match_rate"] == 1.0
    assert metrics["mean_token_match_rate"] == 1.0
    assert metrics["max_logprob_delta"] == 0.0
    assert metrics["mean_logprob_delta"] == 0.0
    assert metrics["parsed_ok_rate"] == 1.0


def test_token_divergence_fails_and_reports_position(tmp_path: Path):
    records_a = [
        make_record("p1", ["a", "b", "c"], backend="mrv1"),
        make_record("p2", ["x", "y", "z"], backend="mrv1"),
    ]
    records_b = [
        make_record("p1", ["a", "b", "c"], backend="mrv2"),
        make_record("p2", ["x", "other", "z"], backend="mrv2"),
    ]

    code, report = run_compare(tmp_path, *write_pair(tmp_path, records_a, records_b))

    assert code == compare_tool.EXIT_MISMATCH
    assert "## Verdict: FAIL" in report
    assert "p2: greedy generation diverged at token 1" in report
    assert "| `p2` | 1 |" in report
    path_a, path_b = write_pair(tmp_path, records_a, records_b)
    diffs = {diff.record_id: diff for diff in compare_tool.compare_files(path_a, path_b).diffs}
    assert diffs["p1"].exact_tokens is True
    assert diffs["p2"].exact_tokens is False
    assert diffs["p2"].divergence_at == 1


def test_logprob_delta_over_threshold_fails(tmp_path: Path):
    tokens = ["a", "b"]
    records_a = [make_record("p1", tokens, [-0.1, -0.2], backend="mrv1")]
    records_b = [make_record("p1", tokens, [-0.1, -0.5], backend="mrv2")]

    code, report = run_compare(tmp_path, *write_pair(tmp_path, records_a, records_b))

    assert code == compare_tool.EXIT_MISMATCH
    assert "logprob delta 0.300000 at position 1 exceeds" in report
    assert "| `p1` | yes | 2/2 | yes | 100.00% | 0.300000 |" in report


def test_logprob_delta_within_threshold_passes(tmp_path: Path):
    tokens = ["a", "b"]
    records_a = [make_record("p1", tokens, [-0.1, -0.2], backend="mrv1")]
    records_b = [make_record("p1", tokens, [-0.1, -0.5], backend="mrv2")]

    code, _ = run_compare(tmp_path, *write_pair(tmp_path, records_a, records_b), "--max-logprob-delta", "0.5")

    assert code == compare_tool.EXIT_OK


@pytest.mark.parametrize(
    "payload, expected",
    [
        pytest.param("not json\n", "invalid JSON", id="invalid-json"),
        pytest.param("[]\n", "expected a JSON object", id="json-array"),
        pytest.param("", "no records found", id="empty-file"),
    ],
)
def test_malformed_json_is_input_error(tmp_path: Path, payload: str, expected: str, capsys: pytest.CaptureFixture):
    good = write_jsonl(tmp_path / "mrv2.jsonl", [make_record("p1", ["a"], backend="mrv2")])
    broken = tmp_path / "mrv1.jsonl"
    broken.write_text(payload, encoding="utf-8")

    code, _ = run_compare(tmp_path, broken, good)

    assert code == compare_tool.EXIT_INPUT_ERROR
    assert expected in capsys.readouterr().err


def test_missing_record_field_is_input_error(tmp_path: Path, capsys: pytest.CaptureFixture):
    broken_record = make_record("p1", ["a"], backend="mrv1")
    del broken_record["tokens"]
    good = write_jsonl(tmp_path / "mrv2.jsonl", [make_record("p1", ["a"], backend="mrv2")])
    broken = write_jsonl(tmp_path / "mrv1.jsonl", [broken_record])

    code, _ = run_compare(tmp_path, broken, good)

    assert code == compare_tool.EXIT_INPUT_ERROR
    assert "'tokens'" in capsys.readouterr().err


def test_missing_id_is_input_error(tmp_path: Path, capsys: pytest.CaptureFixture):
    broken_record = make_record("p1", ["a"], backend="mrv1")
    del broken_record["id"]
    good = write_jsonl(tmp_path / "mrv2.jsonl", [make_record("p1", ["a"], backend="mrv2")])
    broken = write_jsonl(tmp_path / "mrv1.jsonl", [broken_record])

    code, _ = run_compare(tmp_path, broken, good)

    assert code == compare_tool.EXIT_INPUT_ERROR
    assert "'id'" in capsys.readouterr().err


def test_duplicate_id_is_input_error(tmp_path: Path, capsys: pytest.CaptureFixture):
    duplicated = [make_record("p1", ["a"], backend="mrv1"), make_record("p1", ["a"], backend="mrv1")]
    good = write_jsonl(tmp_path / "mrv2.jsonl", [make_record("p1", ["a"], backend="mrv2")])

    code, _ = run_compare(tmp_path, write_jsonl(tmp_path / "mrv1.jsonl", duplicated), good)

    assert code == compare_tool.EXIT_INPUT_ERROR
    assert "duplicate id" in capsys.readouterr().err


def test_missing_input_file_is_input_error(tmp_path: Path, capsys: pytest.CaptureFixture):
    good = write_jsonl(tmp_path / "mrv2.jsonl", [make_record("p1", ["a"], backend="mrv2")])

    code, _ = run_compare(tmp_path, tmp_path / "does_not_exist.jsonl", good)

    assert code == compare_tool.EXIT_INPUT_ERROR
    assert "cannot read" in capsys.readouterr().err


def test_unknown_report_suffix_is_input_error(tmp_path: Path, capsys: pytest.CaptureFixture):
    path_a, path_b = write_pair(tmp_path, [make_record("p1", ["a"])], [make_record("p1", ["a"], backend="mrv2")])
    out = tmp_path / "report.txt"

    code = compare_tool.main(["compare", "--a", str(path_a), "--b", str(path_b), "--out", str(out)])

    assert code == compare_tool.EXIT_INPUT_ERROR
    assert "must end with one of" in capsys.readouterr().err
    assert not out.exists()


def test_greedy_exact_check_applies_only_to_temperature_zero(tmp_path: Path):
    sampled = 0.7
    records_a = [make_record("p1", ["a", "b"], backend="mrv1", temperature=sampled)]
    records_b = [make_record("p1", ["a", "c"], backend="mrv2", temperature=sampled)]

    code, report = run_compare(tmp_path, *write_pair(tmp_path, records_a, records_b))

    assert code == compare_tool.EXIT_OK
    assert "| `p1` | 1 |" in report
    assert "non-greedy records are listed for information only" in report

    records_a = [make_record("p1", ["a", "b"], backend="mrv1", temperature=0.0)]
    records_b = [make_record("p1", ["a", "c"], backend="mrv2", temperature=0.0)]

    code, report = run_compare(tmp_path, *write_pair(tmp_path, records_a, records_b))

    assert code == compare_tool.EXIT_MISMATCH
    assert "greedy generation diverged" in report


def test_greedy_check_can_be_disabled(tmp_path: Path):
    records_a = [make_record("p1", ["a", "b"], backend="mrv1", temperature=0.0)]
    records_b = [make_record("p1", ["a", "c"], backend="mrv2", temperature=0.0)]

    code, report = run_compare(tmp_path, *write_pair(tmp_path, records_a, records_b), "--no-greedy-require-exact")

    assert code == compare_tool.EXIT_OK
    assert "greedy exact tokens required: no" in report


def test_greedy_policy_needs_temperature_metadata(tmp_path: Path):
    records_a = [make_record("p1", ["a", "b"], backend="mrv1", temperature=0.0)]
    records_b = [make_record("p1", ["a", "b"], backend="mrv2", temperature=0.0)]
    records_a[0].pop("temperature")

    code, report = run_compare(tmp_path, *write_pair(tmp_path, records_a, records_b))

    assert code == compare_tool.EXIT_MISMATCH
    assert "has no numeric 'temperature'" in report


def test_parsed_ok_false_fails(tmp_path: Path):
    records_a = [make_record("p1", [], backend="mrv1", parsed_ok=False)]
    records_b = [make_record("p1", [], backend="mrv2", parsed_ok=True)]

    code, report = run_compare(tmp_path, *write_pair(tmp_path, records_a, records_b))

    assert code == compare_tool.EXIT_MISMATCH
    assert "parsed_ok is false" in report


def test_id_present_in_only_one_run_fails(tmp_path: Path):
    records_a = [make_record("p1", ["a"], backend="mrv1"), make_record("p2", ["b"], backend="mrv1")]
    records_b = [make_record("p1", ["a"], backend="mrv2")]

    code, report = run_compare(tmp_path, *write_pair(tmp_path, records_a, records_b))

    assert code == compare_tool.EXIT_MISMATCH
    assert "id(s) collected for MRV1 but missing from MRV2: p2" in report


def test_same_backend_on_both_sides_fails(tmp_path: Path):
    records_a = [make_record("p1", ["a"], backend="mrv2")]
    records_b = [make_record("p1", ["a"], backend="mrv2")]

    code, report = run_compare(tmp_path, *write_pair(tmp_path, records_a, records_b))

    assert code == compare_tool.EXIT_MISMATCH
    assert "expected backend mrv1 in --a and mrv2 in --b" in report


def test_different_prompt_hash_fails(tmp_path: Path):
    records_a = [make_record("p1", ["a"], backend="mrv1", prompt="left prompt")]
    records_b = [make_record("p1", ["a"], backend="mrv2", prompt="right prompt")]

    code, report = run_compare(tmp_path, *write_pair(tmp_path, records_a, records_b))

    assert code == compare_tool.EXIT_MISMATCH
    assert "prompt_sha1 differs" in report


def test_json_report_shape(tmp_path: Path):
    records_a = [make_record("p1", ["a", "b"], backend="mrv1")]
    records_b = [make_record("p1", ["a", "b"], backend="mrv2")]
    path_a, path_b = write_pair(tmp_path, records_a, records_b)
    out = tmp_path / "report.json"

    code = compare_tool.main(["compare", "--a", str(path_a), "--b", str(path_b), "--out", str(out)])

    assert code == compare_tool.EXIT_OK
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["policy"] == {"max_logprob_delta": 0.05, "greedy_require_exact": True}
    assert payload["metrics"]["records_compared"] == 1
    assert payload["records"][0]["id"] == "p1"
    assert payload["records"][0]["exact_tokens"] is True


def test_check_backend_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    unset_message = compare_tool.check_backend_env("mrv1")
    assert unset_message is not None
    assert "is unset" in unset_message

    monkeypatch.setenv(ENV_VAR, "1")
    assert compare_tool.check_backend_env("mrv2") is None
    mismatch = compare_tool.check_backend_env("mrv1")
    assert mismatch is not None
    assert "contradicts --backend mrv1" in mismatch

    monkeypatch.setenv(ENV_VAR, "0")
    assert compare_tool.check_backend_env("mrv1") is None
    assert compare_tool.check_backend_env("mrv2") is not None


def test_collect_refuses_mismatched_env_without_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
):
    prompts = write_jsonl(tmp_path / "prompts.jsonl", [{"id": "p1", "prompt": "hello", "max_tokens": 8}])
    out = tmp_path / "collected.jsonl"
    monkeypatch.setenv(ENV_VAR, "1")

    code = compare_tool.main(
        ["collect", "--backend", "mrv1", "--model", "GLM-5.1", "--requests", str(prompts), "--out", str(out)]
    )

    assert code == compare_tool.EXIT_INPUT_ERROR
    assert "contradicts --backend mrv1" in capsys.readouterr().err
    assert not out.exists()


def test_collect_dry_run_builds_payloads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
):
    prompts = write_jsonl(
        tmp_path / "prompts.jsonl",
        [
            {"id": "p1", "prompt": "hello", "max_tokens": 8, "temperature": 0.0, "top_p": 0.9, "seed": 1234},
            {"id": "p2", "prompt": "world"},
        ],
    )
    monkeypatch.delenv(ENV_VAR, raising=False)

    code = compare_tool.main(
        ["collect", "--backend", "mrv1", "--model", "GLM-5.1", "--requests", str(prompts), "--dry-run"]
    )

    assert code == compare_tool.EXIT_OK
    payloads = [compare_tool.json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert payloads[0]["id"] == "p1"
    assert payloads[0]["payload"] == {
        "model": "GLM-5.1",
        "prompt": "hello",
        "max_tokens": 8,
        "logprobs": 5,
        "temperature": 0.0,
        "top_p": 0.9,
        "seed": 1234,
    }
    assert payloads[1]["payload"]["max_tokens"] == compare_tool.DEFAULT_MAX_TOKENS
    assert "seed" not in payloads[1]["payload"]


def test_collect_requires_out_without_dry_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
):
    prompts = write_jsonl(tmp_path / "prompts.jsonl", [{"id": "p1", "prompt": "hello"}])
    monkeypatch.setenv(ENV_VAR, "0")

    code = compare_tool.main(["collect", "--backend", "mrv1", "--model", "GLM-5.1", "--requests", str(prompts)])

    assert code == compare_tool.EXIT_INPUT_ERROR
    assert "--out is required" in capsys.readouterr().err


def test_collect_rejects_duplicate_prompt_ids(tmp_path: Path, capsys: pytest.CaptureFixture):
    prompts = write_jsonl(
        tmp_path / "prompts.jsonl",
        [{"id": "p1", "prompt": "hello"}, {"id": "p1", "prompt": "hello again"}],
    )

    code = compare_tool.main(
        ["collect", "--backend", "mrv1", "--model", "GLM-5.1", "--requests", str(prompts), "--dry-run"]
    )

    assert code == compare_tool.EXIT_INPUT_ERROR
    assert "duplicate id" in capsys.readouterr().err


def test_extract_choice_aligns_logprobs():
    response = {
        "choices": [
            {
                "logprobs": {"tokens": ["a", "b", "c"], "token_logprobs": [-0.5, None]},
                "finish_reason": "stop",
            }
        ]
    }

    tokens, logprobs, finish_reason = compare_tool.extract_choice(response)

    assert tokens == ["a", "b", "c"]
    assert logprobs == [-0.5, None, None]
    assert finish_reason == "stop"


def test_extract_choice_without_choices_raises():
    with pytest.raises(compare_tool.CollectError, match="no 'choices'"):
        compare_tool.extract_choice({"choices": []})


def test_prompt_set_covers_lengths_languages_and_sampling_modes():
    records = compare_tool.load_jsonl(PROMPT_SET)

    assert len(records) >= 9
    ids = [record["id"] for record in records]
    assert len(ids) == len(set(ids))

    combinations = set()
    modes = set()
    for record in records:
        assert set(record) == {"id", "prompt", "max_tokens", "temperature", "top_p", "seed"}
        assert isinstance(record["prompt"], str) and record["prompt"]
        assert isinstance(record["max_tokens"], int) and record["max_tokens"] > 0
        length, language, mode = record["id"].split("-")
        combinations.add((length, language))
        modes.add(mode)
        if length == "long":
            assert len(record["prompt"]) >= 2000, record["id"]
        if mode == "greedy":
            assert record["temperature"] == 0, record["id"]
        else:
            assert record["temperature"] == 0.7, record["id"]
            assert record["top_p"] == 0.95, record["id"]
            assert record["seed"] == 1234, record["id"]

    assert combinations == {
        ("short", "zh"),
        ("short", "en"),
        ("short", "code"),
        ("medium", "zh"),
        ("medium", "en"),
        ("medium", "code"),
        ("long", "zh"),
        ("long", "en"),
        ("long", "code"),
    }
    assert modes == {"greedy", "sampled"}


def test_prompt_set_is_replayable_by_collect():
    requests = compare_tool.load_requests(PROMPT_SET)

    assert len(requests) >= 9
    for request in requests:
        payload = compare_tool.build_completion_payload(request, "GLM-5.1")
        assert payload["logprobs"] == compare_tool.LOGPROB_TOP_K
        assert payload["prompt"] == request["prompt"]
