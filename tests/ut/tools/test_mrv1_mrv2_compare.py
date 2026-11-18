import json
import urllib.error
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
    assert "| `p1` | yes | 2/2 | 100.00% | 2/2 (100.00%) | 2/2 (100.00%) | 0 | 0.300000 | yes | FAIL |" in report
    assert "## Verdict: FAIL" in report


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


def test_exact_match_rule_applies_only_to_non_greedy_records(tmp_path: Path):
    sampled = 0.7
    records_a = [make_record("p1", ["a", "b"], backend="mrv1", temperature=sampled)]
    records_b = [make_record("p1", ["a", "c"], backend="mrv2", temperature=sampled)]

    code, report = run_compare(tmp_path, *write_pair(tmp_path, records_a, records_b))

    # 1 of 2 token positions match: 50% < the 90% non-greedy exact-rate default.
    assert code == compare_tool.EXIT_MISMATCH
    assert "| `p1` | 1 |" in report
    assert "non-greedy exact token match rate 50.00% is below --min-exact-match-rate 90%" in report
    assert "judged by --min-exact-match-rate" in report

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


def test_non_greedy_divergence_skips_logprobs_of_that_position(tmp_path: Path):
    # R1: the two runs emitted different tokens at position 1, so the two logprobs
    # belong to different random variables and must not be compared.
    tokens_a = ["m", "n", "o"]
    tokens_b = ["m", "NN", "o"]
    records_a = [make_record("s1", tokens_a, [-0.1, -0.2, -0.3], backend="mrv1", temperature=0.7)]
    records_b = [make_record("s1", tokens_b, [-0.1, -0.7, -0.3], backend="mrv2", temperature=0.7)]

    code, report = run_compare(tmp_path, *write_pair(tmp_path, records_a, records_b), "--min-exact-match-rate", "0.5")

    assert code == compare_tool.EXIT_OK
    assert "non-comparable positions (tokens differ) | 1 |" in report
    assert "| `s1` | 1 | `n` | `NN` |" in report
    assert "## Verdict: PASS" in report


def test_same_token_position_over_threshold_still_fails(tmp_path: Path):
    tokens = ["m", "n"]
    records_a = [make_record("s1", tokens, [-0.1, -0.2], backend="mrv1", temperature=0.7)]
    records_b = [make_record("s1", tokens, [-0.1, -0.7], backend="mrv2", temperature=0.7)]

    code, report = run_compare(tmp_path, *write_pair(tmp_path, records_a, records_b))

    assert code == compare_tool.EXIT_MISMATCH
    assert "logprob delta 0.500000 at position 1 exceeds" in report
    assert "comparable position(s) over threshold" in report
    assert "non-comparable positions (tokens differ) | 0 |" in report


def test_incomparable_positions_reach_the_json_report(tmp_path: Path):
    records_a = [make_record("s1", ["m", "n"], [-0.1, -0.2], backend="mrv1", temperature=0.7)]
    records_b = [make_record("s1", ["m", "NN"], [-0.1, -0.9], backend="mrv2", temperature=0.7)]
    path_a, path_b = write_pair(tmp_path, records_a, records_b)
    out = tmp_path / "report.json"

    code = compare_tool.main(
        [
            "compare",
            "--a",
            str(path_a),
            "--b",
            str(path_b),
            "--out",
            str(out),
            "--min-exact-match-rate",
            "0.4",
        ]
    )

    assert code == compare_tool.EXIT_OK
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["metrics"]["incomparable_positions"] == 1
    assert payload["records"][0]["incomparable"] == [{"position": 1, "a": "n", "b": "NN"}]
    assert payload["records"][0]["exact_rate"] == 0.5
    assert payload["records"][0]["comparable_positions"] == 1
    assert payload["records"][0]["logprob_coverage"] == 1.0
    assert payload["records"][0]["verdict"] == "PASS"


def test_token_length_mismatch_under_greedy_fails(tmp_path: Path):
    records_a = [make_record("g1", ["a", "b", "c"], backend="mrv1", temperature=0.0)]
    records_b = [make_record("g1", ["a", "b"], backend="mrv2", temperature=0.0)]

    code, report = run_compare(tmp_path, *write_pair(tmp_path, records_a, records_b))

    assert code == compare_tool.EXIT_MISMATCH
    assert "g1: greedy generation diverged at token 2" in report


def test_token_length_mismatch_under_sampling_fails_exact_rate(tmp_path: Path):
    records_a = [make_record("s1", ["a", "b", "c"], backend="mrv1", temperature=0.7)]
    records_b = [make_record("s1", ["a", "b"], backend="mrv2", temperature=0.7)]

    code, report = run_compare(tmp_path, *write_pair(tmp_path, records_a, records_b))

    assert code == compare_tool.EXIT_MISMATCH
    assert "| 3/2 |" in report
    assert "non-greedy exact token match rate 66.67% is below --min-exact-match-rate 90%" in report
    assert "## Verdict: FAIL" in report


def test_temperature_mismatch_fails(tmp_path: Path):
    records_a = [make_record("p1", ["a", "b"], backend="mrv1", temperature=0.0)]
    records_b = [make_record("p1", ["a", "b"], backend="mrv2", temperature=0.7)]

    code, report = run_compare(tmp_path, *write_pair(tmp_path, records_a, records_b))

    assert code == compare_tool.EXIT_MISMATCH
    assert "temperature differs (--a=0.0, --b=0.7)" in report
    assert "the two runs are not the same configuration" in report


def test_temperature_mismatch_fails_even_without_greedy_policy(tmp_path: Path):
    records_a = [make_record("p1", ["a", "b"], backend="mrv1", temperature=0.0)]
    records_b = [make_record("p1", ["a", "b"], backend="mrv2", temperature=0.7)]

    code, report = run_compare(tmp_path, *write_pair(tmp_path, records_a, records_b), "--no-greedy-require-exact")

    assert code == compare_tool.EXIT_MISMATCH
    assert "temperature differs" in report


def test_collect_request_failure_writes_out_and_exits_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
):
    prompts = write_jsonl(tmp_path / "prompts.jsonl", [{"id": "p1", "prompt": "hello", "max_tokens": 8}])
    out = tmp_path / "collected.jsonl"
    monkeypatch.setenv(ENV_VAR, "0")

    def boom(*args, **kwargs):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(compare_tool, "post_completion", boom)

    code = compare_tool.main(
        ["collect", "--backend", "mrv1", "--model", "GLM-5.1", "--requests", str(prompts), "--out", str(out)]
    )

    assert code == compare_tool.EXIT_MISMATCH
    assert "produced no tokens" in capsys.readouterr().err
    records = compare_tool.load_jsonl(out)
    assert len(records) == 1
    assert records[0]["parsed_ok"] is False
    assert records[0]["tokens"] == []
    assert records[0]["logprobs"] == []
    assert "URLError" in records[0]["error"]


def test_negative_max_logprob_delta_is_input_error(tmp_path: Path, capsys: pytest.CaptureFixture):
    path_a, path_b = write_pair(tmp_path, [make_record("p1", ["a"])], [make_record("p1", ["a"], backend="mrv2")])
    out = tmp_path / "report.md"

    code = compare_tool.main(
        ["compare", "--a", str(path_a), "--b", str(path_b), "--out", str(out), "--max-logprob-delta", "-0.1"]
    )

    assert code == compare_tool.EXIT_INPUT_ERROR
    assert "must not be negative" in capsys.readouterr().err
    assert not out.exists()


def test_non_greedy_zero_exact_match_fails(tmp_path: Path):
    records_a = [make_record("p1", ["a", "b", "c"], backend="mrv1", temperature=0.7)]
    records_b = [make_record("p1", ["x", "y", "z"], backend="mrv2", temperature=0.7)]

    code, report = run_compare(tmp_path, *write_pair(tmp_path, records_a, records_b))

    assert code == compare_tool.EXIT_MISMATCH
    assert "non-greedy exact token match rate 0.00% is below --min-exact-match-rate 90%" in report
    assert "## Verdict: FAIL" in report


def test_both_sides_missing_logprobs_is_inconclusive(tmp_path: Path):
    tokens = ["a", "b"]
    records_a = [make_record("p1", tokens, [None, None], backend="mrv1")]
    records_b = [make_record("p1", tokens, [None, None], backend="mrv2")]

    code, report = run_compare(tmp_path, *write_pair(tmp_path, records_a, records_b))

    assert code == compare_tool.EXIT_INPUT_ERROR
    assert "## Verdict: INCONCLUSIVE" in report
    assert "## Inconclusive records" in report
    assert "logprob coverage 0/2" in report
    assert "both missing 2, one side missing 0" in report


def test_one_side_missing_logprobs_is_inconclusive(tmp_path: Path):
    tokens = ["a", "b"]
    records_a = [make_record("p1", tokens, [-0.1, -0.2], backend="mrv1")]
    records_b = [make_record("p1", tokens, [None, None], backend="mrv2")]

    code, report = run_compare(tmp_path, *write_pair(tmp_path, records_a, records_b))

    assert code == compare_tool.EXIT_INPUT_ERROR
    assert "both missing 0, one side missing 2" in report
    assert "## Verdict: INCONCLUSIVE" in report


def test_comparable_ratio_below_threshold_is_inconclusive(tmp_path: Path):
    records_a = [make_record("g1", ["a", "x", "y"], backend="mrv1", temperature=0.0)]
    records_b = [make_record("g1", ["a", "p", "q"], backend="mrv2", temperature=0.0)]

    # Greedy exactness is disabled on purpose: the comparable-ratio rule must
    # catch the record even though greedy divergence would otherwise fail it.
    code, report = run_compare(tmp_path, *write_pair(tmp_path, records_a, records_b), "--no-greedy-require-exact")

    assert code == compare_tool.EXIT_INPUT_ERROR
    assert "evidence insufficient: only 1/3 comparable position(s)" in report
    assert "## Verdict: INCONCLUSIVE" in report


def test_exact_rate_at_the_threshold_passes(tmp_path: Path):
    tokens_a = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
    tokens_b = list(tokens_a[:-1]) + ["Z"]
    records_a = [make_record("p1", tokens_a, backend="mrv1", temperature=0.7)]
    records_b = [make_record("p1", tokens_b, backend="mrv2", temperature=0.7)]

    code, report = run_compare(tmp_path, *write_pair(tmp_path, records_a, records_b))

    assert code == compare_tool.EXIT_OK
    assert "| 90.00% | 9/10 (90.00%) |" in report
    assert "## Verdict: PASS" in report


def test_exact_rate_below_the_threshold_fails(tmp_path: Path):
    tokens_a = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
    tokens_b = list(tokens_a[:-2]) + ["Y", "Z"]
    records_a = [make_record("p1", tokens_a, backend="mrv1", temperature=0.7)]
    records_b = [make_record("p1", tokens_b, backend="mrv2", temperature=0.7)]

    code, report = run_compare(tmp_path, *write_pair(tmp_path, records_a, records_b))

    assert code == compare_tool.EXIT_MISMATCH
    assert "non-greedy exact token match rate 80.00% is below --min-exact-match-rate 90%" in report


def test_empty_tokens_are_inconclusive(tmp_path: Path):
    records_a = [make_record("p1", [], [], backend="mrv1")]
    records_b = [make_record("p1", [], [], backend="mrv2")]

    code, report = run_compare(tmp_path, *write_pair(tmp_path, records_a, records_b))

    assert code == compare_tool.EXIT_INPUT_ERROR
    assert "both runs produced no tokens" in report
    assert "## Verdict: INCONCLUSIVE" in report


def test_inconclusive_aggregation_beats_pass(tmp_path: Path):
    records_a = [
        make_record("pass1", ["a"], backend="mrv1"),
        make_record("inc1", ["b"], [None], backend="mrv1"),
    ]
    records_b = [
        make_record("pass1", ["a"], backend="mrv2"),
        make_record("inc1", ["b"], [None], backend="mrv2"),
    ]

    code, report = run_compare(tmp_path, *write_pair(tmp_path, records_a, records_b))

    assert code == compare_tool.EXIT_INPUT_ERROR
    assert "## Verdict: INCONCLUSIVE" in report
    assert "| records PASS / FAIL / INCONCLUSIVE | 1 / 0 / 1 |" in report


def test_fail_aggregation_beats_inconclusive(tmp_path: Path):
    records_a = [
        make_record("inc1", ["b"], [None], backend="mrv1"),
        make_record("fail1", ["a", "b"], backend="mrv1", temperature=0.7),
    ]
    records_b = [
        make_record("inc1", ["b"], [None], backend="mrv2"),
        make_record("fail1", ["x", "y"], backend="mrv2", temperature=0.7),
    ]

    code, report = run_compare(tmp_path, *write_pair(tmp_path, records_a, records_b))

    assert code == compare_tool.EXIT_MISMATCH
    assert "## Verdict: FAIL" in report
    assert "| records PASS / FAIL / INCONCLUSIVE | 0 / 1 / 1 |" in report


def test_thresholds_are_echoed_in_markdown(tmp_path: Path):
    records_a = [make_record("p1", ["a", "b"], backend="mrv1")]
    records_b = [make_record("p1", ["a", "b"], backend="mrv2")]

    code, report = run_compare(
        tmp_path,
        *write_pair(tmp_path, records_a, records_b),
        "--min-exact-match-rate",
        "0.5",
        "--min-comparable-ratio",
        "0.4",
        "--min-logprob-coverage",
        "0.6",
    )

    assert code == compare_tool.EXIT_OK
    assert "- min exact match rate (non-greedy): 0.5" in report
    assert "- min comparable ratio: 0.4" in report
    assert "- min logprob coverage: 0.6" in report
    assert "| min exact match rate (non-greedy) | 50.00% |" in report
    assert "| min comparable ratio | 40.00% |" in report
    assert "| min logprob coverage | 60.00% |" in report


def test_thresholds_are_echoed_in_json(tmp_path: Path):
    records_a = [make_record("p1", ["a", "b"], backend="mrv1")]
    records_b = [make_record("p1", ["a", "b"], backend="mrv2")]
    path_a, path_b = write_pair(tmp_path, records_a, records_b)
    out = tmp_path / "report.json"

    code = compare_tool.main(["compare", "--a", str(path_a), "--b", str(path_b), "--out", str(out)])

    assert code == compare_tool.EXIT_OK
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["verdict"] == "PASS"
    assert payload["thresholds"] == {
        "min_exact_match_rate": 0.90,
        "min_comparable_ratio": 0.50,
        "min_logprob_coverage": 0.90,
    }
    assert payload["metrics"]["thresholds"] == payload["thresholds"]
    assert payload["metrics"]["exact_match_rate"] == 1.0
    assert payload["metrics"]["comparable_ratio"] == 1.0
    assert payload["metrics"]["logprob_coverage"] == 1.0
    assert payload["metrics"]["inconclusive_records"] == []
    assert payload["metrics"]["records_passed"] == 1
    assert payload["records"][0]["verdict"] == "PASS"
    assert payload["records"][0]["exact_rate"] == 1.0
    assert payload["records"][0]["comparable_positions"] == 2


def test_inconclusive_json_metrics(tmp_path: Path):
    tokens = ["a", "b"]
    records_a = [make_record("p1", tokens, [None, None], backend="mrv1")]
    records_b = [make_record("p1", tokens, [None, None], backend="mrv2")]
    path_a, path_b = write_pair(tmp_path, records_a, records_b)
    out = tmp_path / "report.json"

    code = compare_tool.main(["compare", "--a", str(path_a), "--b", str(path_b), "--out", str(out)])

    assert code == compare_tool.EXIT_INPUT_ERROR
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["verdict"] == "INCONCLUSIVE"
    assert payload["metrics"]["inconclusive_records"] == ["p1"]
    assert payload["metrics"]["records_inconclusive"] == 1
    assert payload["records"][0]["missing_both_logprobs"] == 2
    assert payload["records"][0]["missing_one_side_logprobs"] == 0
    assert payload["records"][0]["logprob_coverage"] == 0.0
    assert "evidence insufficient" in payload["records"][0]["verdict_reason"]


def test_comparable_ratio_threshold_can_be_lowered(tmp_path: Path):
    records_a = [make_record("g1", ["a", "x", "y"], backend="mrv1", temperature=0.0)]
    records_b = [make_record("g1", ["a", "p", "q"], backend="mrv2", temperature=0.0)]

    code, report = run_compare(
        tmp_path,
        *write_pair(tmp_path, records_a, records_b),
        "--no-greedy-require-exact",
        "--min-comparable-ratio",
        "0.3",
    )

    assert code == compare_tool.EXIT_OK
    assert "## Verdict: PASS" in report


def test_logprob_coverage_threshold_can_be_lowered(tmp_path: Path):
    tokens = ["a", "b"]
    records_a = [make_record("p1", tokens, [None, None], backend="mrv1")]
    records_b = [make_record("p1", tokens, [None, None], backend="mrv2")]

    code, report = run_compare(tmp_path, *write_pair(tmp_path, records_a, records_b), "--min-logprob-coverage", "0.0")

    assert code == compare_tool.EXIT_OK
    assert "## Verdict: PASS" in report


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--min-exact-match-rate", "-0.1"),
        ("--min-exact-match-rate", "1.5"),
        ("--min-comparable-ratio", "-0.1"),
        ("--min-comparable-ratio", "1.5"),
        ("--min-logprob-coverage", "-0.1"),
        ("--min-logprob-coverage", "1.5"),
    ],
)
def test_threshold_outside_unit_interval_is_input_error(
    tmp_path: Path, capsys: pytest.CaptureFixture, flag: str, value: str
):
    path_a, path_b = write_pair(tmp_path, [make_record("p1", ["a"])], [make_record("p1", ["a"], backend="mrv2")])
    out = tmp_path / "report.md"

    code = compare_tool.main(["compare", "--a", str(path_a), "--b", str(path_b), "--out", str(out), flag, value])

    assert code == compare_tool.EXIT_INPUT_ERROR
    assert "must be within [0, 1]" in capsys.readouterr().err
    assert not out.exists()


def test_stdout_last_line_is_the_verdict(tmp_path: Path, capsys: pytest.CaptureFixture):
    tokens = ["a", "b"]
    records_a = [make_record("p1", tokens, [None, None], backend="mrv1")]
    records_b = [make_record("p1", tokens, [None, None], backend="mrv2")]
    path_a, path_b = write_pair(tmp_path, records_a, records_b)
    out = tmp_path / "report.md"

    code = compare_tool.main(["compare", "--a", str(path_a), "--b", str(path_b), "--out", str(out)])

    assert code == compare_tool.EXIT_INPUT_ERROR
    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert lines[-1] == "verdict: INCONCLUSIVE"
