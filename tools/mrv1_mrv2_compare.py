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
"""Request-level parity harness for the Ascend model runner V1 vs V2.

``VLLM_USE_V2_MODEL_RUNNER`` is read once per process, so MRV1 and MRV2 cannot
coexist inside a single server. The comparison is therefore two-stage:

1. ``collect`` replays one prompt set against a running server and stores the
   generated token strings plus their logprobs as JSONL. Run it twice: once
   against a server started with ``VLLM_USE_V2_MODEL_RUNNER=0`` and once
   against a server started with ``VLLM_USE_V2_MODEL_RUNNER=1``.
2. ``compare`` diffs the two JSONL files and emits a Markdown or JSON report.

Only the Python standard library is imported, so ``compare`` runs on a
CPU-only host without vLLM or torch_npu.

Records are JSONL objects. ``collect`` writes ``{id, backend,
prompt_sha1, temperature, tokens, logprobs, finish_reason, parsed_ok}`` (plus
``error`` when a request failed); ``tokens`` are the decoded token strings from
``logprobs.tokens`` because the completions API returns no token ids, and
``logprobs`` is aligned with them. ``compare`` also reads ``temperature`` to decide
whether the greedy exact-match rule applies, so it treats a record without one
as unusable input.

Logprobs are compared only at positions where both runs emitted the same token;
a position whose tokens differ is reported as non-comparable and never fails a
run, because a non-greedy sample legitimately diverges there. A comparable
position that lacks a logprob on both sides (or on one side) is counted as
missing evidence instead of being compared.

``compare`` decides one of three verdicts per record and aggregates them:

* ``PASS`` (exit 0) - the evidence is sufficient and every threshold holds.
* ``FAIL`` (exit 1) - a structural failure (prompt/backend/temperature/parsed_ok
  mismatch, greedy token divergence) or a violated decision threshold.
* ``INCONCLUSIVE`` (exit 2) - the evidence is insufficient to judge: no tokens,
  too few comparable positions, or too few logprobs. Insufficient evidence never
  counts as a pass.

Decision thresholds (also written into ``--help`` and echoed by both report
formats):

* ``--min-exact-match-rate`` (default 0.90): every non-greedy record must match
  this share of its token positions exactly, otherwise it fails.
* ``--max-single-sided-missing-ratio`` (default 0.00): a position where exactly
  one run carries a logprob is asymmetric evidence; above this share the record
  fails. The default fails on any single-sided position.
* ``--min-comparable-positions`` (default 16): absolute lower bound on the number
  of token-equal (comparable) positions. Fewer positions cannot carry a verdict,
  however consistent they look, so the record is inconclusive.
* ``--min-logprob-coverage`` (default 0.90): of the comparable positions, this
  share must carry a logprob on both sides, otherwise the record is inconclusive.
  This only constrains positions where both sides are missing; single-sided
  gaps are handled by ``--max-single-sided-missing-ratio``. The threshold is a
  lower bound compared with a strict less-than: a coverage exactly equal to it
  passes (e.g. 0.90 with the default); only strictly less is inconclusive.

Metric denominators: ``position_total = max(len(tokens_a), len(tokens_b))``;
``position_matches`` counts positions whose tokens are equal (the comparable
positions, and also the numerator of ``exact_rate = position_matches /
position_total``); ``single_sided_missing_ratio = missing_one_side /
position_matches``; ``logprob_coverage = both_present / position_matches``.
Decision order per record: structural failure, no tokens, non-greedy exact rate,
single-sided missing ratio, comparable position count, logprob coverage, PASS.

Failure semantics:

* ``collect`` unusable input or environment (``VLLM_USE_V2_MODEL_RUNNER`` missing or
  contradicting ``--backend``, unreadable prompt file, missing ``--out``) ->
  exit 2, **no output file is written**.
* ``collect`` request-level failure (HTTP error, timeout, response without
  tokens or logprobs) -> the JSONL **is** written, the record keeps
  ``parsed_ok: false`` plus an ``error`` message, exit 1.
* ``compare`` unusable input (unreadable file, malformed JSONL, missing or
  duplicate field/id, unknown ``--out`` suffix, negative ``--max-logprob-delta``,
  threshold outside [0, 1]) -> exit 2, **no report file is written**.
* ``compare`` FAIL or INCONCLUSIVE -> the report **is** written, exit 1 or 2.

Exit codes: ``0`` parity holds, ``1`` parity is broken, ``2`` the input is
unusable or the evidence is inconclusive.
"""

# ruff: noqa: E402

from __future__ import annotations

import os
import sys

# Running 'python tools/mrv1_mrv2_compare.py' puts tools/ first on sys.path,
# where the tools/bisect subpackage shadows the standard library bisect module
# that urllib.request needs, so this script would die before main() on a plain
# script invocation. Drop that entry so the standard library always wins;
# 'python -m tools.mrv1_mrv2_compare' (the form the docs use for the other
# tools) never hits this. The import order below is load bearing: the guard
# must run before urllib is imported.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if sys.path and os.path.abspath(sys.path[0] or ".") == _SCRIPT_DIR:
    sys.path.pop(0)

import argparse
import hashlib
import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

EXIT_OK = 0
EXIT_MISMATCH = 1
EXIT_INPUT_ERROR = 2

BACKEND_ENV_VAR = "VLLM_USE_V2_MODEL_RUNNER"
BACKEND_ENV_VALUES: dict[str, frozenset[str]] = {
    "mrv1": frozenset({"0", "false", "off", "no"}),
    "mrv2": frozenset({"1", "true", "on", "yes"}),
}
BACKENDS = ("mrv1", "mrv2")
DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_MAX_LOGPROB_DELTA = 0.05
DEFAULT_MAX_TOKENS = 128
DEFAULT_TIMEOUT_SECONDS = 600.0
GREEDY_TEMPERATURE = 0.0
LOGPROB_TOP_K = 5
MAX_LISTED_ITEMS = 5
FRAGMENT_RADIUS = 2
DEFAULT_MIN_EXACT_MATCH_RATE = 0.90
DEFAULT_MIN_COMPARABLE_POSITIONS = 16
DEFAULT_MIN_LOGPROB_COVERAGE = 0.90
DEFAULT_MAX_SINGLE_SIDED_MISSING_RATIO = 0.0
VERDICT_PASS = "PASS"
VERDICT_FAIL = "FAIL"
VERDICT_INCONCLUSIVE = "INCONCLUSIVE"
VERDICT_EXIT_CODES = {
    VERDICT_PASS: EXIT_OK,
    VERDICT_FAIL: EXIT_MISMATCH,
    VERDICT_INCONCLUSIVE: EXIT_INPUT_ERROR,
}
RECORD_FIELDS = ("id", "backend", "prompt_sha1", "tokens", "logprobs", "finish_reason", "parsed_ok")
REPORT_SUFFIXES = (".md", ".json")

EXAMPLES = """\
examples:
  # MRV1 server on :8000, MRV2 server on :8001, same weights and same serve flags
  VLLM_USE_V2_MODEL_RUNNER=0 python tools/mrv1_mrv2_compare.py collect \
      --backend mrv1 --base-url http://127.0.0.1:8000 --model GLM-5.1 \
      --requests tools/mrv1_mrv2_prompts.jsonl --out mrv1.jsonl
  VLLM_USE_V2_MODEL_RUNNER=1 python tools/mrv1_mrv2_compare.py collect \
      --backend mrv2 --base-url http://127.0.0.1:8001 --model GLM-5.1 \
      --requests tools/mrv1_mrv2_prompts.jsonl --out mrv2.jsonl
  python tools/mrv1_mrv2_compare.py compare --a mrv1.jsonl --b mrv2.jsonl --out report.md
"""


class InputError(ValueError):
    """A JSONL input file or a report path cannot be used."""


class CollectError(RuntimeError):
    """A completion response cannot be turned into a comparable record."""


@dataclass(frozen=True)
class Thresholds:
    """Decision thresholds for the three-state verdict.

    Evidence strictly below a threshold makes a record INCONCLUSIVE (or FAIL
    for the exact-match rule; the single-sided ratio is a strict upper bound),
    never PASS: a run whose logprobs are missing everywhere, or whose two
    samples share too few comparable positions, says nothing about parity and
    must not be reported as a pass. Exactly at a lower-bound threshold passes.
    """

    min_exact_match_rate: float = DEFAULT_MIN_EXACT_MATCH_RATE
    min_comparable_positions: int = DEFAULT_MIN_COMPARABLE_POSITIONS
    min_logprob_coverage: float = DEFAULT_MIN_LOGPROB_COVERAGE
    max_single_sided_missing_ratio: float = DEFAULT_MAX_SINGLE_SIDED_MISSING_RATIO

    def to_dict(self) -> dict[str, float | int]:
        """JSON form of the thresholds, echoed by both report formats."""
        return {
            "min_exact_match_rate": self.min_exact_match_rate,
            "min_comparable_positions": self.min_comparable_positions,
            "min_logprob_coverage": self.min_logprob_coverage,
            "max_single_sided_missing_ratio": self.max_single_sided_missing_ratio,
        }


def prompt_sha1(prompt: str) -> str:
    """Stable fingerprint that proves both runs received the same prompt text."""
    return hashlib.sha1(prompt.encode("utf-8")).hexdigest()


def _preview(items: list[str]) -> str:
    """Comma separated head of a list, truncated so messages stay readable."""
    head = ", ".join(str(item) for item in items[:MAX_LISTED_ITEMS])
    if len(items) > MAX_LISTED_ITEMS:
        return f"{head}, ... ({len(items)} total)"
    return head


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _format_rate(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2%}"


def _format_delta(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6f}"


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file, rejecting malformed lines and non-object records."""
    file_path = Path(path)
    try:
        text = file_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise InputError(f"cannot read {file_path}: {exc}") from exc
    records: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise InputError(f"{file_path}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(item, dict):
            raise InputError(f"{file_path}:{line_number}: expected a JSON object, got {type(item).__name__}")
        records.append(item)
    if not records:
        raise InputError(f"{file_path}: no records found")
    return records


def validate_record(record: dict[str, Any], origin: str) -> None:
    """Reject a record that :func:`compare_record` cannot reason about."""
    missing = [name for name in RECORD_FIELDS if name not in record]
    if missing:
        raise InputError(f"{origin}: record is missing field(s) {missing}")
    record_id = record["id"]
    if not isinstance(record_id, str) or not record_id:
        raise InputError(f"{origin}: 'id' must be a non-empty string, got {record_id!r}")
    if record["backend"] not in BACKENDS:
        raise InputError(f"{origin}: record {record_id!r} has unknown backend {record['backend']!r}")
    if not isinstance(record["prompt_sha1"], str):
        raise InputError(f"{origin}: record {record_id!r} has a non-string 'prompt_sha1'")
    if not isinstance(record["parsed_ok"], bool):
        raise InputError(f"{origin}: record {record_id!r} has a non-boolean 'parsed_ok'")
    tokens = record["tokens"]
    logprobs = record["logprobs"]
    if not isinstance(tokens, list) or not all(isinstance(token, str) for token in tokens):
        raise InputError(f"{origin}: record {record_id!r} 'tokens' must be a list of token strings")
    if not isinstance(logprobs, list):
        raise InputError(f"{origin}: record {record_id!r} 'logprobs' must be a list")
    if len(logprobs) != len(tokens):
        raise InputError(
            f"{origin}: record {record_id!r} has {len(logprobs)} logprob(s) for {len(tokens)} token(s); "
            "the two lists must be aligned"
        )
    for position, value in enumerate(logprobs):
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise InputError(f"{origin}: record {record_id!r} logprob[{position}] is not a number: {value!r}")


def index_records(records: list[dict[str, Any]], origin: str) -> dict[str, dict[str, Any]]:
    """Validate records and key them by id, rejecting duplicates."""
    index: dict[str, dict[str, Any]] = {}
    for record in records:
        validate_record(record, origin)
        record_id = record["id"]
        if record_id in index:
            raise InputError(f"{origin}: duplicate id {record_id!r}")
        index[record_id] = record
    return index


def has_temperature(record: dict[str, Any]) -> bool:
    """Whether the record carries a numeric sampling temperature."""
    temperature = record.get("temperature")
    return isinstance(temperature, (int, float)) and not isinstance(temperature, bool)


def is_greedy(record: dict[str, Any]) -> bool:
    """True only when the record proves it was generated with temperature 0."""
    if not has_temperature(record):
        return False
    return float(record["temperature"]) == GREEDY_TEMPERATURE


def first_divergence(tokens_a: list[str], tokens_b: list[str]) -> int:
    """Index of the first differing token, or the shorter length for a prefix."""
    for position, (token_a, token_b) in enumerate(zip(tokens_a, tokens_b, strict=False)):
        if token_a != token_b:
            return position
    return min(len(tokens_a), len(tokens_b))


def escape_token(token: str) -> str:
    """Render one token so it cannot break a Markdown table cell."""
    return token.replace("|", "\\|").replace("\n", "\\n")


def token_fragment(tokens: list[str], position: int) -> str:
    """Short window of decoded tokens around a divergence, safe inside a Markdown table."""
    start = max(0, position - FRAGMENT_RADIUS)
    stop = min(len(tokens), position + FRAGMENT_RADIUS + 1)
    window = " | ".join(escape_token(token) for token in tokens[start:stop])
    prefix = "..." if start > 0 else ""
    suffix = "..." if stop < len(tokens) else ""
    return f"{prefix}{window}{suffix}"


@dataclass
class IncomparablePosition:
    """One aligned position whose tokens differ, so its logprobs are not comparable."""

    position: int
    token_a: str
    token_b: str


@dataclass
class LogprobStats:
    """Logprob agreement measured on the positions both runs call comparable."""

    positions: int = 0
    delta_sum: float = 0.0
    max_delta: float | None = None
    missing_both: int = 0
    missing_one_side: int = 0
    incomparable: list[IncomparablePosition] = field(default_factory=list)


def compare_logprobs(
    record_id: str,
    tokens_a: list[str],
    tokens_b: list[str],
    logprobs_a: list[float | None],
    logprobs_b: list[float | None],
    max_logprob_delta: float,
    failures: list[str],
) -> LogprobStats:
    """Compare logprobs only where both runs emitted the same token.

    A position whose tokens differ is not comparable: the two values belong to
    different random variables, so their difference says nothing about parity.
    Those positions are reported instead of failing the run, which is what makes
    a legitimately diverging non-greedy sample comparable at all.
    """
    stats = LogprobStats()
    deltas: list[tuple[int, float]] = []
    for position, (token_a, token_b, value_a, value_b) in enumerate(
        zip(tokens_a, tokens_b, logprobs_a, logprobs_b, strict=False)
    ):
        if token_a != token_b:
            stats.incomparable.append(IncomparablePosition(position, token_a, token_b))
            continue
        if value_a is None and value_b is None:
            stats.missing_both += 1
            continue
        if value_a is None or value_b is None:
            stats.missing_one_side += 1
            continue
        deltas.append((position, abs(float(value_a) - float(value_b))))
    oversized = [(position, delta) for position, delta in deltas if delta > max_logprob_delta]
    if oversized:
        position, delta = oversized[0]
        failures.append(
            f"{record_id}: logprob delta {delta:.6f} at position {position} exceeds "
            f"--max-logprob-delta {max_logprob_delta} ({len(oversized)} comparable position(s) over threshold)"
        )
    stats.positions = len(deltas)
    stats.delta_sum = sum(delta for _, delta in deltas)
    stats.max_delta = max((delta for _, delta in deltas), default=None)
    return stats


@dataclass
class TokenDiff:
    """Per-record result of comparing the MRV1 and MRV2 generation."""

    record_id: str
    greedy: bool
    tokens_a: int
    tokens_b: int
    exact_tokens: bool
    position_total: int
    position_matches: int
    logprob_positions: int
    logprob_delta_sum: float
    max_logprob_delta: float | None
    parsed_ok: bool
    exact_rate: float | None = None
    comparable_ratio: float | None = None
    single_sided_missing_ratio: float | None = None
    logprob_coverage: float | None = None
    comparable_positions: int = 0
    missing_both_logprobs: int = 0
    missing_one_side_logprobs: int = 0
    verdict: str = VERDICT_PASS
    verdict_reason: str = ""
    divergence_at: int | None = None
    fragment_a: str = ""
    fragment_b: str = ""
    incomparable: list[IncomparablePosition] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)


def compare_record(
    record_a: dict[str, Any],
    record_b: dict[str, Any],
    max_logprob_delta: float,
    greedy_require_exact: bool,
    thresholds: Thresholds | None = None,
) -> TokenDiff:
    """Diff one MRV1 record against the MRV2 record that carries the same id."""
    thresholds = Thresholds() if thresholds is None else thresholds
    record_id = record_a["id"]
    tokens_a = record_a["tokens"]
    tokens_b = record_b["tokens"]
    exact_tokens = tokens_a == tokens_b
    greedy = is_greedy(record_a) and is_greedy(record_b)
    parsed_ok = bool(record_a["parsed_ok"]) and bool(record_b["parsed_ok"])
    position_total = max(len(tokens_a), len(tokens_b))
    position_matches = sum(1 for token_a, token_b in zip(tokens_a, tokens_b, strict=False) if token_a == token_b)

    failures: list[str] = []
    if record_a["prompt_sha1"] != record_b["prompt_sha1"]:
        failures.append(f"{record_id}: prompt_sha1 differs, the two runs did not receive the same prompt")
    if record_a["backend"] != "mrv1" or record_b["backend"] != "mrv2":
        failures.append(
            f"{record_id}: expected backend mrv1 in --a and mrv2 in --b, got "
            f"{record_a['backend']!r}/{record_b['backend']!r}"
        )
    if (
        has_temperature(record_a)
        and has_temperature(record_b)
        and float(record_a["temperature"]) != float(record_b["temperature"])
    ):
        failures.append(
            f"{record_id}: temperature differs (--a={record_a['temperature']!r}, "
            f"--b={record_b['temperature']!r}); the two runs are not the same configuration"
        )
    if greedy_require_exact:
        for label, record in (("--a", record_a), ("--b", record_b)):
            if not has_temperature(record):
                failures.append(
                    f"{record_id}: {label} record has no numeric 'temperature'; the greedy exact check needs it "
                    "(collect always writes it)"
                )
    if not parsed_ok:
        failures.append(f"{record_id}: at least one run failed to parse the completion (parsed_ok is false)")

    divergence_at: int | None = None
    fragment_a = ""
    fragment_b = ""
    if not exact_tokens:
        divergence_at = first_divergence(tokens_a, tokens_b)
        fragment_a = token_fragment(tokens_a, divergence_at)
        fragment_b = token_fragment(tokens_b, divergence_at)
        if greedy_require_exact and greedy:
            failures.append(
                f"{record_id}: greedy generation diverged at token {divergence_at} (a={fragment_a!r}, b={fragment_b!r})"
            )

    logprob_stats = compare_logprobs(
        record_id,
        tokens_a,
        tokens_b,
        record_a["logprobs"],
        record_b["logprobs"],
        max_logprob_delta,
        failures,
    )
    exact_rate = position_matches / position_total if position_total else None
    comparable_ratio = exact_rate
    single_sided_ratio = logprob_stats.missing_one_side / position_matches if position_matches else 0.0
    coverage = logprob_stats.positions / position_matches if position_matches else (0.0 if position_total else None)
    verdict = VERDICT_PASS
    verdict_reason = ""
    if failures:
        verdict = VERDICT_FAIL
        verdict_reason = failures[0]
    elif position_total == 0 or (not tokens_a and not tokens_b):
        verdict = VERDICT_INCONCLUSIVE
        verdict_reason = "both runs produced no tokens; there is nothing to compare"
    elif not greedy and exact_rate is not None and exact_rate < thresholds.min_exact_match_rate:
        verdict = VERDICT_FAIL
        verdict_reason = (
            f"non-greedy exact token match rate {exact_rate:.2%} is below "
            f"--min-exact-match-rate {thresholds.min_exact_match_rate:.0%}"
        )
    elif position_matches and single_sided_ratio > thresholds.max_single_sided_missing_ratio:
        verdict = VERDICT_FAIL
        verdict_reason = (
            f"single-sided missing logprobs {logprob_stats.missing_one_side}/{position_matches} "
            f"({single_sided_ratio:.2%}) exceed "
            f"--max-single-sided-missing-ratio {thresholds.max_single_sided_missing_ratio:.2%}"
        )
    elif position_matches < thresholds.min_comparable_positions:
        verdict = VERDICT_INCONCLUSIVE
        verdict_reason = (
            f"evidence insufficient: only {position_matches} comparable position(s), "
            f"--min-comparable-positions is {thresholds.min_comparable_positions}"
        )
    elif coverage is not None and coverage < thresholds.min_logprob_coverage:
        verdict = VERDICT_INCONCLUSIVE
        verdict_reason = (
            f"evidence insufficient: logprob coverage {logprob_stats.positions}/{position_matches} "
            f"({coverage:.2%}) < --min-logprob-coverage {thresholds.min_logprob_coverage:.0%} "
            f"(both missing {logprob_stats.missing_both}, one side missing {logprob_stats.missing_one_side})"
        )
    return TokenDiff(
        record_id=record_id,
        greedy=greedy,
        tokens_a=len(tokens_a),
        tokens_b=len(tokens_b),
        exact_tokens=exact_tokens,
        position_total=position_total,
        position_matches=position_matches,
        logprob_positions=logprob_stats.positions,
        logprob_delta_sum=logprob_stats.delta_sum,
        max_logprob_delta=logprob_stats.max_delta,
        parsed_ok=parsed_ok,
        exact_rate=exact_rate,
        comparable_ratio=comparable_ratio,
        single_sided_missing_ratio=single_sided_ratio,
        logprob_coverage=coverage,
        comparable_positions=position_matches,
        missing_both_logprobs=logprob_stats.missing_both,
        missing_one_side_logprobs=logprob_stats.missing_one_side,
        verdict=verdict,
        verdict_reason=verdict_reason,
        divergence_at=divergence_at,
        fragment_a=fragment_a,
        fragment_b=fragment_b,
        incomparable=logprob_stats.incomparable,
        failures=failures,
    )


@dataclass
class ComparisonReport:
    """Everything ``compare`` learned about one MRV1/MRV2 pair of runs."""

    a_path: str
    b_path: str
    max_logprob_delta: float
    greedy_require_exact: bool
    records_a: int = 0
    records_b: int = 0
    diffs: list[TokenDiff] = field(default_factory=list)
    missing_ids: list[str] = field(default_factory=list)
    unexpected_ids: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    thresholds: Thresholds = field(default_factory=Thresholds)

    @property
    def verdict(self) -> str:
        """Aggregate the record verdicts: FAIL wins, then INCONCLUSIVE, then PASS."""
        if self.failures or any(diff.verdict == VERDICT_FAIL for diff in self.diffs):
            return VERDICT_FAIL
        if any(diff.verdict == VERDICT_INCONCLUSIVE for diff in self.diffs):
            return VERDICT_INCONCLUSIVE
        return VERDICT_PASS

    @property
    def passed(self) -> bool:
        return self.verdict == VERDICT_PASS

    def verdict_counts(self) -> dict[str, int]:
        """How many records landed in each verdict bucket."""
        counts = {VERDICT_PASS: 0, VERDICT_FAIL: 0, VERDICT_INCONCLUSIVE: 0}
        for diff in self.diffs:
            counts[diff.verdict] = counts.get(diff.verdict, 0) + 1
        return counts


def compare_files(
    path_a: str | Path,
    path_b: str | Path,
    max_logprob_delta: float = DEFAULT_MAX_LOGPROB_DELTA,
    greedy_require_exact: bool = True,
    thresholds: Thresholds | None = None,
) -> ComparisonReport:
    """Load two collected JSONL files and diff them record by record."""
    thresholds = Thresholds() if thresholds is None else thresholds
    records_a = load_jsonl(path_a)
    records_b = load_jsonl(path_b)
    index_a = index_records(records_a, str(path_a))
    index_b = index_records(records_b, str(path_b))
    report = ComparisonReport(
        a_path=str(path_a),
        b_path=str(path_b),
        max_logprob_delta=max_logprob_delta,
        greedy_require_exact=greedy_require_exact,
        records_a=len(records_a),
        records_b=len(records_b),
        thresholds=thresholds,
    )
    report.missing_ids = [record_id for record_id in index_a if record_id not in index_b]
    report.unexpected_ids = [record_id for record_id in index_b if record_id not in index_a]
    if report.missing_ids:
        report.failures.append(
            f"{len(report.missing_ids)} id(s) collected for MRV1 but missing from MRV2: {_preview(report.missing_ids)}"
        )
    if report.unexpected_ids:
        report.failures.append(
            f"{len(report.unexpected_ids)} id(s) collected for MRV2 but missing from MRV1: "
            f"{_preview(report.unexpected_ids)}"
        )
    for record_id, record_a in index_a.items():
        record_b = index_b.get(record_id)
        if record_b is None:
            continue
        diff = compare_record(record_a, record_b, max_logprob_delta, greedy_require_exact, thresholds)
        report.diffs.append(diff)
        report.failures.extend(diff.failures)
    if report.records_a != report.records_b:
        report.failures.append(f"record count differs: --a has {report.records_a}, --b has {report.records_b}")
    return report


def summarize(report: ComparisonReport) -> dict[str, Any]:
    """Aggregate the per-record diffs into the parity metrics the report shows."""
    greedy = [diff for diff in report.diffs if diff.greedy]
    sampled = [diff for diff in report.diffs if not diff.greedy]
    position_total = sum(diff.position_total for diff in report.diffs)
    position_matches = sum(diff.position_matches for diff in report.diffs)
    logprob_positions = sum(diff.logprob_positions for diff in report.diffs)
    single_sided_missing = sum(diff.missing_one_side_logprobs for diff in report.diffs)
    logprob_delta_sum = sum(diff.logprob_delta_sum for diff in report.diffs)
    max_delta = max(
        (diff.max_logprob_delta for diff in report.diffs if diff.max_logprob_delta is not None),
        default=None,
    )
    return {
        "records_a": report.records_a,
        "records_b": report.records_b,
        "records_compared": len(report.diffs),
        "missing_ids": report.missing_ids,
        "unexpected_ids": report.unexpected_ids,
        "exact_token_match_rate": _rate(sum(1 for diff in report.diffs if diff.exact_tokens), len(report.diffs)),
        "greedy_records": len(greedy),
        "greedy_exact_token_match_rate": _rate(sum(1 for diff in greedy if diff.exact_tokens), len(greedy)),
        "non_greedy_records": len(sampled),
        "non_greedy_exact_token_match_rate": _rate(sum(1 for diff in sampled if diff.exact_tokens), len(sampled)),
        "mean_token_match_rate": _rate(position_matches, position_total),
        "incomparable_positions": sum(len(diff.incomparable) for diff in report.diffs),
        "max_logprob_delta": max_delta,
        "mean_logprob_delta": _rate(logprob_delta_sum, logprob_positions),
        "parsed_ok_rate": _rate(sum(1 for diff in report.diffs if diff.parsed_ok), len(report.diffs)),
        "records_passed": report.verdict_counts()[VERDICT_PASS],
        "records_failed": report.verdict_counts()[VERDICT_FAIL],
        "records_inconclusive": report.verdict_counts()[VERDICT_INCONCLUSIVE],
        "inconclusive_records": [diff.record_id for diff in report.diffs if diff.verdict == VERDICT_INCONCLUSIVE],
        "failed_records": [diff.record_id for diff in report.diffs if diff.verdict == VERDICT_FAIL],
        "exact_match_rate": _rate(position_matches, position_total),
        "comparable_positions_total": position_matches,
        "single_sided_missing_ratio": _rate(single_sided_missing, position_matches),
        "logprob_coverage": _rate(logprob_positions, position_matches),
        "thresholds": report.thresholds.to_dict(),
        "verdict": report.verdict,
        "passed": report.passed,
    }


def report_to_dict(report: ComparisonReport, metrics: dict[str, Any]) -> dict[str, Any]:
    """JSON form of the report, written when ``--out`` ends with `.json`."""
    return {
        "a": report.a_path,
        "b": report.b_path,
        "policy": {
            "max_logprob_delta": report.max_logprob_delta,
            "greedy_require_exact": report.greedy_require_exact,
        },
        "metrics": metrics,
        "records": [
            {
                "id": diff.record_id,
                "greedy": diff.greedy,
                "tokens_a": diff.tokens_a,
                "tokens_b": diff.tokens_b,
                "exact_tokens": diff.exact_tokens,
                "position_total": diff.position_total,
                "position_matches": diff.position_matches,
                "max_logprob_delta": diff.max_logprob_delta,
                "parsed_ok": diff.parsed_ok,
                "exact_rate": diff.exact_rate,
                "comparable_ratio": diff.comparable_ratio,
                "comparable_positions": diff.comparable_positions,
                "single_sided_missing_ratio": diff.single_sided_missing_ratio,
                "logprob_coverage": diff.logprob_coverage,
                "missing_both_logprobs": diff.missing_both_logprobs,
                "missing_one_side_logprobs": diff.missing_one_side_logprobs,
                "verdict": diff.verdict,
                "verdict_reason": diff.verdict_reason,
                "divergence_at": diff.divergence_at,
                "fragment_a": diff.fragment_a,
                "fragment_b": diff.fragment_b,
                "incomparable": [
                    {"position": item.position, "a": item.token_a, "b": item.token_b} for item in diff.incomparable
                ],
                "failures": diff.failures,
            }
            for diff in report.diffs
        ],
        "failures": report.failures,
        "verdict": report.verdict,
        "thresholds": report.thresholds.to_dict(),
        "passed": report.passed,
    }


def render_markdown(report: ComparisonReport, metrics: dict[str, Any]) -> str:
    """Render the human readable report with the summary table and the divergence list."""
    lines = [
        "# MRV1 vs MRV2 request-level comparison",
        "",
        f"- MRV1 input (`--a`): `{report.a_path}` ({report.records_a} record(s))",
        f"- MRV2 input (`--b`): `{report.b_path}` ({report.records_b} record(s))",
        f"- max logprob delta allowed: {report.max_logprob_delta}",
        f"- greedy exact tokens required: {'yes' if report.greedy_require_exact else 'no'}",
        f"- min exact match rate (non-greedy): {report.thresholds.min_exact_match_rate}",
        f"- min comparable positions: {report.thresholds.min_comparable_positions}",
        f"- min logprob coverage: {report.thresholds.min_logprob_coverage}",
        f"- max single-sided missing ratio: {report.thresholds.max_single_sided_missing_ratio}",
        f"- verdict: {report.verdict}",
        "",
        "## Summary",
        "",
        "| metric | value |",
        "|---|---|",
        f"| records compared | {metrics['records_compared']} |",
        f"| records PASS / FAIL / INCONCLUSIVE | {metrics['records_passed']} / {metrics['records_failed']} "
        f"/ {metrics['records_inconclusive']} |",
        f"| exact token match rate | {_format_rate(metrics['exact_token_match_rate'])} |",
        f"| greedy exact match rate ({metrics['greedy_records']} record(s)) "
        f"| {_format_rate(metrics['greedy_exact_token_match_rate'])} |",
        f"| non-greedy exact match rate ({metrics['non_greedy_records']} record(s)) "
        f"| {_format_rate(metrics['non_greedy_exact_token_match_rate'])} |",
        f"| mean per-token match rate | {_format_rate(metrics['mean_token_match_rate'])} |",
        f"| non-comparable positions (tokens differ) | {metrics['incomparable_positions']} |",
        f"| max logprob delta | {_format_delta(metrics['max_logprob_delta'])} |",
        f"| mean logprob delta | {_format_delta(metrics['mean_logprob_delta'])} |",
        f"| parsed_ok rate | {_format_rate(metrics['parsed_ok_rate'])} |",
        f"| comparable positions (all records) | {metrics['comparable_positions_total']} |",
        f"| single-sided missing ratio (all records) | {_format_rate(metrics['single_sided_missing_ratio'])} |",
        f"| logprob coverage (all records) | {_format_rate(metrics['logprob_coverage'])} |",
        f"| min exact match rate (non-greedy) | {_format_rate(report.thresholds.min_exact_match_rate)} |",
        f"| min comparable positions | {report.thresholds.min_comparable_positions} |",
        f"| min logprob coverage | {_format_rate(report.thresholds.min_logprob_coverage)} |",
        f"| max single-sided missing ratio | {_format_rate(report.thresholds.max_single_sided_missing_ratio)} |",
        f"| missing ids | {_preview(report.missing_ids) if report.missing_ids else 'none'} |",
        f"| unexpected ids | {_preview(report.unexpected_ids) if report.unexpected_ids else 'none'} |",
        "",
        "Logprob deltas are measured on comparable positions only: positions where the two runs",
        "emitted different tokens are listed under 'Non-comparable positions' and never fail a run.",
        "",
        "## Records",
        "",
        "| id | greedy | tokens (a/b) | exact% | comparable | logprob coverage "
        "| non-comparable | max logprob delta | parsed_ok | verdict |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for diff in report.diffs:
        comparable = f"{diff.position_matches} pos ({_format_rate(diff.comparable_ratio)} of {diff.position_total})"
        coverage = f"{diff.logprob_positions}/{diff.position_matches} ({_format_rate(diff.logprob_coverage)})"
        lines.append(
            f"| `{diff.record_id}` | {'yes' if diff.greedy else 'no'} | {diff.tokens_a}/{diff.tokens_b} "
            f"| {_format_rate(diff.exact_rate)} | {comparable} | {coverage} | {len(diff.incomparable)} "
            f"| {_format_delta(diff.max_logprob_delta)} | {'yes' if diff.parsed_ok else 'no'} | {diff.verdict} |"
        )
    lines += ["", "## Token divergences", ""]
    divergent = [diff for diff in report.diffs if not diff.exact_tokens]
    if divergent:
        lines += ["| id | first divergence | a fragment | b fragment |", "|---|---|---|---|"]
        lines += [
            f"| `{diff.record_id}` | {diff.divergence_at} | `{diff.fragment_a}` | `{diff.fragment_b}` |"
            for diff in divergent
        ]
        if report.greedy_require_exact:
            lines += [
                "",
                "Greedy records must match exactly; non-greedy records are judged by --min-exact-match-rate.",
            ]
    else:
        lines.append("No token divergence between the two runs.")
    lines += ["", "## Non-comparable positions", ""]
    incomparable = [(diff.record_id, item) for diff in report.diffs for item in diff.incomparable]
    if incomparable:
        lines += ["| id | position | a token | b token |", "|---|---|---|---|"]
        lines += [
            f"| `{record_id}` | {item.position} | `{escape_token(item.token_a)}` | `{escape_token(item.token_b)}` |"
            for record_id, item in incomparable
        ]
        lines += [
            "",
            "Tokens differ at these positions, so the two logprobs are not the same random variable;",
            "the delta is skipped and the position never makes a run fail.",
        ]
    else:
        lines.append("Every aligned position carries the same token in both runs.")
    lines += ["", "## Failed records", ""]
    failed = [diff for diff in report.diffs if diff.verdict == VERDICT_FAIL]
    if failed:
        lines += ["| id | reason |", "|---|---|"]
        lines += [f"| `{diff.record_id}` | {diff.verdict_reason} |" for diff in failed]
    else:
        lines.append("None.")
    lines += ["", "## Inconclusive records", ""]
    inconclusive = [diff for diff in report.diffs if diff.verdict == VERDICT_INCONCLUSIVE]
    if inconclusive:
        lines += ["| id | reason |", "|---|---|"]
        lines += [f"| `{diff.record_id}` | {diff.verdict_reason} |" for diff in inconclusive]
    else:
        lines.append("None.")
    lines += ["", "## Failures", ""]
    lines += [f"- {failure}" for failure in report.failures] if report.failures else ["None."]
    lines += ["", f"## Verdict: {report.verdict}", ""]
    return "\n".join(lines)


def build_thresholds(args: argparse.Namespace) -> Thresholds:
    """Validate the three rate thresholds and package them for the report."""
    values = {
        "--min-exact-match-rate": args.min_exact_match_rate,
        "--min-logprob-coverage": args.min_logprob_coverage,
        "--max-single-sided-missing-ratio": args.max_single_sided_missing_ratio,
    }
    for name, value in values.items():
        if not 0.0 <= float(value) <= 1.0:
            raise InputError(f"{name} must be within [0, 1], got {value!r}")
    if int(args.min_comparable_positions) < 0:
        raise InputError(f"--min-comparable-positions must not be negative, got {args.min_comparable_positions!r}")
    return Thresholds(
        min_exact_match_rate=float(args.min_exact_match_rate),
        min_comparable_positions=int(args.min_comparable_positions),
        min_logprob_coverage=float(args.min_logprob_coverage),
        max_single_sided_missing_ratio=float(args.max_single_sided_missing_ratio),
    )


def run_compare(args: argparse.Namespace) -> int:
    """Entry point of the ``compare`` subcommand."""
    out_path = Path(args.out)
    if out_path.suffix.lower() not in REPORT_SUFFIXES:
        print(f"error: --out must end with one of {REPORT_SUFFIXES}, got {out_path.name!r}", file=sys.stderr)
        return EXIT_INPUT_ERROR
    if args.max_logprob_delta < 0:
        print("error: --max-logprob-delta must not be negative", file=sys.stderr)
        return EXIT_INPUT_ERROR
    try:
        thresholds = build_thresholds(args)
    except InputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INPUT_ERROR
    try:
        report = compare_files(args.a, args.b, args.max_logprob_delta, args.greedy_require_exact, thresholds)
        metrics = summarize(report)
        if out_path.suffix.lower() == ".json":
            text = json.dumps(report_to_dict(report, metrics), indent=2, ensure_ascii=False) + "\n"
        else:
            text = render_markdown(report, metrics)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
    except (InputError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INPUT_ERROR
    print(
        f"records compared: {metrics['records_compared']} "
        f"(a={metrics['records_a']}, b={metrics['records_b']}); "
        f"PASS {metrics['records_passed']} / FAIL {metrics['records_failed']} "
        f"/ INCONCLUSIVE {metrics['records_inconclusive']}; "
        f"exact match rate {_format_rate(metrics['exact_match_rate'])}; "
        f"comparable positions {metrics['comparable_positions_total']}; "
        f"single-sided missing ratio {_format_rate(metrics['single_sided_missing_ratio'])}; "
        f"logprob coverage {_format_rate(metrics['logprob_coverage'])}; "
        f"max logprob delta {_format_delta(metrics['max_logprob_delta'])} "
        f"(allowed {report.max_logprob_delta}); "
        f"{metrics['incomparable_positions']} non-comparable position(s)"
    )
    if report.failures:
        print(f"failures: {len(report.failures)}")
        for failure in report.failures:
            print(f"  - {failure}")
    bad = [diff for diff in report.diffs if diff.verdict != VERDICT_PASS]
    if bad:
        print(f"records not passing: {len(bad)}")
        for diff in bad:
            print(f"  - {diff.verdict} {diff.record_id}: {diff.verdict_reason}")
    print(f"report: {out_path}")
    print(f"verdict: {report.verdict}")
    return VERDICT_EXIT_CODES[report.verdict]


def _require_optional_number(request: dict[str, Any], name: str, origin: str, integer: bool = False) -> None:
    value = request.get(name)
    if value is None:
        return
    valid = isinstance(value, int) if integer else isinstance(value, (int, float))
    if isinstance(value, bool) or not valid:
        expected = "an integer" if integer else "a number"
        raise InputError(f"{origin}: {name!r} must be {expected} or null, got {value!r}")


def load_requests(path: str | Path) -> list[dict[str, Any]]:
    """Read and validate the prompt set that ``collect`` replays."""
    requests = load_jsonl(path)
    seen: set[str] = set()
    for number, request in enumerate(requests, start=1):
        origin = f"{path}:record {number}"
        record_id = request.get("id")
        if not isinstance(record_id, str) or not record_id:
            raise InputError(f"{origin}: 'id' must be a non-empty string, got {record_id!r}")
        if record_id in seen:
            raise InputError(f"{origin}: duplicate id {record_id!r}")
        seen.add(record_id)
        prompt = request.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise InputError(f"{origin}: 'prompt' must be a non-empty string")
        _require_optional_number(request, "max_tokens", origin, integer=True)
        _require_optional_number(request, "seed", origin, integer=True)
        _require_optional_number(request, "temperature", origin)
        _require_optional_number(request, "top_p", origin)
    return requests


def build_completion_payload(request: dict[str, Any], model: str) -> dict[str, Any]:
    """Turn one prompt entry into a ``/v1/completions`` request body."""
    payload: dict[str, Any] = {
        "model": model,
        "prompt": request["prompt"],
        "max_tokens": int(request.get("max_tokens", DEFAULT_MAX_TOKENS)),
        "logprobs": LOGPROB_TOP_K,
    }
    for name in ("temperature", "top_p"):
        value = request.get(name)
        if value is not None:
            payload[name] = float(value)
    seed = request.get("seed")
    if seed is not None:
        payload["seed"] = int(seed)
    return payload


def extract_choice(response: dict[str, Any]) -> tuple[list[str], list[float | None], str | None]:
    """Pull aligned token strings, logprobs and finish_reason out of a completion response."""
    choices = response.get("choices") or []
    if not choices:
        raise CollectError("response has no 'choices'")
    choice = choices[0]
    logprobs = choice.get("logprobs") or {}
    tokens = [str(token) for token in logprobs.get("tokens") or []]
    raw_values = logprobs.get("token_logprobs") or []
    aligned: list[float | None] = [None] * len(tokens)
    for position, value in enumerate(raw_values[: len(tokens)]):
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            aligned[position] = float(value)
    return tokens, aligned, choice.get("finish_reason")


def post_completion(base_url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    """POST one completion request and decode the JSON body."""
    url = base_url.rstrip("/") + "/v1/completions"
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        decoded = json.loads(response.read().decode("utf-8"))
    if not isinstance(decoded, dict):
        raise CollectError(f"expected a JSON object from {url}, got {type(decoded).__name__}")
    return decoded


def collect_one(
    request: dict[str, Any],
    payload: dict[str, Any],
    backend: str,
    base_url: str,
    timeout: float,
) -> dict[str, Any]:
    """Send one request and return the comparable record, failures included."""
    record: dict[str, Any] = {
        "id": request["id"],
        "backend": backend,
        "prompt_sha1": prompt_sha1(request["prompt"]),
        "temperature": request.get("temperature"),
        "tokens": [],
        "logprobs": [],
        "finish_reason": None,
        "parsed_ok": False,
    }
    try:
        response = post_completion(base_url, payload, timeout)
        tokens, logprobs, finish_reason = extract_choice(response)
    except (urllib.error.URLError, OSError, ValueError, CollectError) as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        return record
    record.update(tokens=tokens, logprobs=logprobs, finish_reason=finish_reason, parsed_ok=bool(tokens))
    if not tokens:
        record["error"] = "response contained no tokens"
    return record


def write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    """Write records as UTF-8 JSONL, one object per line."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def check_backend_env(backend: str) -> str | None:
    """Return an error message when this shell would collect the wrong runner.

    The server picks its runner from its own environment, and a client cannot
    see that environment. Requiring the same variable here turns "I forgot to
    export it" into an error instead of two silently identical MRV2 runs.
    """
    raw = os.environ.get(BACKEND_ENV_VAR)
    expected = BACKEND_ENV_VALUES[backend]
    if raw is None:
        value = "0" if backend == "mrv1" else "1"
        return (
            f"error: {BACKEND_ENV_VAR} is unset. MRV2 is the default runner, so a --backend {backend} run would "
            f"measure the wrong runner. Start the server with {BACKEND_ENV_VAR}={value} and export the same "
            f"value in this shell."
        )
    if raw.strip().lower() not in expected:
        return (
            f"error: {BACKEND_ENV_VAR}={raw!r} contradicts --backend {backend}; expected one of "
            f"{sorted(expected)}. Refusing to collect mismatched data."
        )
    return None


def run_collect(args: argparse.Namespace) -> int:
    """Entry point of the ``collect`` subcommand."""
    try:
        requests = load_requests(args.requests)
    except (InputError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INPUT_ERROR
    if args.dry_run:
        for request in requests:
            payload = build_completion_payload(request, args.model)
            print(json.dumps({"id": request["id"], "payload": payload}, ensure_ascii=False))
        print(f"dry-run: {len(requests)} payload(s) built, no request was sent", file=sys.stderr)
        return EXIT_OK
    problem = check_backend_env(args.backend)
    if problem is not None:
        print(problem, file=sys.stderr)
        return EXIT_INPUT_ERROR
    if not args.out:
        print("error: --out is required unless --dry-run is set", file=sys.stderr)
        return EXIT_INPUT_ERROR
    records = []
    for request in requests:
        payload = build_completion_payload(request, args.model)
        record = collect_one(request, payload, args.backend, args.base_url, args.timeout)
        records.append(record)
        state = "ok" if record["parsed_ok"] else "failed"
        print(f"[{record['id']}] {record['backend']} {state}: {len(record['tokens'])} token(s)")
    try:
        write_jsonl(args.out, records)
    except OSError as exc:
        print(f"error: cannot write {args.out}: {exc}", file=sys.stderr)
        return EXIT_INPUT_ERROR
    failed = [record["id"] for record in records if not record["parsed_ok"]]
    if failed:
        print(f"error: {len(failed)} request(s) produced no tokens: {_preview(failed)}", file=sys.stderr)
        return EXIT_MISMATCH
    print(f"wrote {len(records)} record(s) to {args.out}")
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    """Command line definition shared by ``main`` and the unit tests."""
    parser = argparse.ArgumentParser(
        prog="mrv1_mrv2_compare",
        description="Collect and compare MRV1 vs MRV2 generations for one prompt set.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EXAMPLES,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect = subparsers.add_parser("collect", help="replay a prompt set against one running server")
    collect.add_argument("--backend", choices=BACKENDS, required=True, help="runner the server was started with")
    collect.add_argument("--base-url", default=DEFAULT_BASE_URL, help=f"server root URL (default: {DEFAULT_BASE_URL})")
    collect.add_argument("--model", required=True, help="served model name")
    collect.add_argument("--requests", required=True, help="prompt JSONL, one {id, prompt, ...} object per line")
    collect.add_argument("--out", default=None, help="where to write the collected JSONL")
    collect.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"per request timeout in seconds (default: {DEFAULT_TIMEOUT_SECONDS})",
    )
    collect.add_argument(
        "--dry-run",
        action="store_true",
        help="print the request payloads instead of calling a server",
    )
    collect.set_defaults(handler=run_collect)

    compare = subparsers.add_parser("compare", help="compare two JSONL files produced by collect")
    compare.add_argument("--a", required=True, help="JSONL collected with --backend mrv1")
    compare.add_argument("--b", required=True, help="JSONL collected with --backend mrv2")
    compare.add_argument("--out", required=True, help="report path; .md writes Markdown, .json writes JSON")
    compare.add_argument(
        "--max-logprob-delta",
        type=float,
        default=DEFAULT_MAX_LOGPROB_DELTA,
        help=f"per token logprob delta treated as a mismatch (default: {DEFAULT_MAX_LOGPROB_DELTA})",
    )
    compare.add_argument(
        "--min-exact-match-rate",
        type=float,
        default=DEFAULT_MIN_EXACT_MATCH_RATE,
        help=(
            f"non-greedy records whose exact token match rate is lower fail (default: {DEFAULT_MIN_EXACT_MATCH_RATE})"
        ),
    )
    compare.add_argument(
        "--min-comparable-positions",
        type=int,
        default=DEFAULT_MIN_COMPARABLE_POSITIONS,
        help=(
            "records with fewer token-equal (comparable) positions than this are inconclusive; "
            "a single agreeing position is not evidence "
            f"(default: {DEFAULT_MIN_COMPARABLE_POSITIONS})"
        ),
    )
    compare.add_argument(
        "--max-single-sided-missing-ratio",
        type=float,
        default=DEFAULT_MAX_SINGLE_SIDED_MISSING_RATIO,
        help=(
            "records whose comparable positions lack a logprob on exactly one side at a higher share "
            f"fail (default: {DEFAULT_MAX_SINGLE_SIDED_MISSING_RATIO})"
        ),
    )
    compare.add_argument(
        "--min-logprob-coverage",
        type=float,
        default=DEFAULT_MIN_LOGPROB_COVERAGE,
        help=(
            "records below this share of comparable positions carrying a logprob on both sides are "
            f"inconclusive (default: {DEFAULT_MIN_LOGPROB_COVERAGE})"
        ),
    )
    compare.add_argument(
        "--greedy-require-exact",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="require identical tokens for records generated with temperature 0 (default: on)",
    )
    compare.set_defaults(handler=run_compare)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse ``argv`` and dispatch to the selected subcommand."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
