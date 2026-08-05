#!/usr/bin/env python3
"""Convert optest .optest-results.json to JUnit XML for GitLab CI.

optest output structure:
  /op_test_output/flashmla/<branch>_<8char_commit>_<timestamp>/
    ├── .optest-results.json
    ├── acc/
    └── *.csv

Logs mirror the same version directory under /op_test_log/flashmla/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, tostring
import xml.dom.minidom

RESULTS_ROOT = Path("/op_test_output/flashmla")
LOG_ROOT = Path("/op_test_log/flashmla")
JUNIT_OUTPUT = (
    Path(sys.argv[1]) / "tests/flashmla_performance_output.xml"
    if len(sys.argv) > 1
    else Path("tests/flashmla_performance_output.xml")
)

_LOG_TAIL_LINES = 80


def find_latest_results() -> Path | None:
    """Return the newest .optest-results.json under RESULTS_ROOT."""
    candidates = sorted(RESULTS_ROOT.glob("*/.optest-results.json"))
    return candidates[-1] if candidates else None


def make_error_suite(message: str) -> str:
    testsuite = Element(
        "testsuite", name="flashmla.performance",
        tests="1", failures="1", errors="0", skipped="0",
    )
    tc = SubElement(testsuite, "testcase",
                    classname="flashmla.performance", name="no_results")
    SubElement(tc, "failure", message=message)
    return xml.dom.minidom.parseString(tostring(testsuite)).toprettyxml()


def _read_log_tail(log_dir: Path, log_rel: str) -> str:
    """Return the tail of a log file, safe for embedding in XML."""
    log_file = log_dir / log_rel
    if not log_file.is_file():
        return f"(log file not found: {log_file})"
    try:
        content = log_file.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"(failed to read log: {exc})"
    all_lines = content.splitlines()
    if len(all_lines) > _LOG_TAIL_LINES:
        return "...\n" + "\n".join(all_lines[-_LOG_TAIL_LINES:])
    return content


def build_junit(data: dict, log_dir: Path) -> str:
    testsuite = Element(
        "testsuite", name="flashmla.performance",
        tests="0", failures="0", errors="0", skipped="0",
    )
    total = 0
    failed = 0
    skipped = 0

    # ── per (performance) tests ──────────────────────────────────────
    per_data = data.get("per", {})
    for test in per_data.get("tests", []):
        total += 1
        name = test["name"]
        duration = test.get("duration_s")
        attrs = {"classname": "flashmla.per", "name": name}
        if duration is not None:
            attrs["time"] = str(duration)
        tc = SubElement(testsuite, "testcase", **attrs)
        if test.get("status") == "FAIL":
            failed += 1
            log_rel = test.get("log", "")
            failure = SubElement(
                tc, "failure",
                message=f"{name} failed (log: {log_rel})",
            )
            failure.text = _read_log_tail(log_dir, log_rel)

    per_summary = per_data.get("summary", {})
    per_failed = per_summary.get("failed", 0)
    if per_failed > failed:
        failed = per_failed

    # ── acc (accuracy / pytest) tests ─────────────────────────────────
    # flashmla does not currently support accuracy tests; keep the parser
    # in place so the CI doesn't break if acc is added later.
    acc_data = data.get("acc", {})
    for row in acc_data.get("rows", []):
        total += 1
        name = row["name"]
        row_passed = row.get("passed", 0)
        row_failed = row.get("failed", 0)
        row_skipped = row.get("skipped", 0)

        tc = SubElement(
            testsuite, "testcase",
            classname="flashmla.acc",
            name=f"{name} (p:{row_passed} f:{row_failed} s:{row_skipped})",
        )
        if row_failed > 0:
            failed += 1
            failure = SubElement(
                tc, "failure",
                message=f"{name}: {row_failed}/{row.get('total', 0)} failed",
            )
            failure.text = _read_log_tail(log_dir, f"acc/{name}.log")
        if row_skipped > 0:
            skipped += 1
            SubElement(tc, "skipped",
                       message=f"{name}: {row_skipped}/{row.get('total', 0)} skipped")

    testsuite.set("tests", str(total))
    testsuite.set("failures", str(failed))
    testsuite.set("skipped", str(skipped))

    return xml.dom.minidom.parseString(tostring(testsuite)).toprettyxml()


def main() -> int:
    results_file = find_latest_results()
    if results_file is None:
        xml_str = make_error_suite("No optest results found under " + str(RESULTS_ROOT))
        JUNIT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        JUNIT_OUTPUT.write_text(xml_str, encoding="utf-8")
        print(f"No optest results found; wrote empty JUnit report to {JUNIT_OUTPUT}")
        return 0

    data = json.loads(results_file.read_text(encoding="utf-8"))
    branch = data.get("branch", "?")
    commit = data.get("commit", "?")[:8]
    version_dir = results_file.parent.name  # e.g. ci_mr_d6f49e99_260805114057
    log_dir = LOG_ROOT / version_dir
    print(f"Converting optest results: {branch=} commit={commit} {version_dir=}")

    xml_str = build_junit(data, log_dir)

    JUNIT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    JUNIT_OUTPUT.write_text(xml_str, encoding="utf-8")
    print(f"JUnit report generated: {JUNIT_OUTPUT}")

    # Return non-zero if any failures so GitLab can fail the job.
    per_summary = data.get("per", {}).get("summary", {})
    acc_summary = data.get("acc", {}).get("summary", {})
    total_failed = per_summary.get("failed", 0) + acc_summary.get("failed", 0)
    return 1 if total_failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
