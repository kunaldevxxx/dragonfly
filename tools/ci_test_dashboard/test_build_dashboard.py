"""Regression tests for compact dashboard summaries; run with unittest discovery."""

import contextlib
import io
import json
import tempfile
import unittest
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import build_dashboard as dashboard


class DashboardTests(unittest.TestCase):
    def metadata(self, day="2026-09-17", run="100", **overrides):
        return replace(
            dashboard.Metadata(
                suite="cpp",
                date=day,
                workflow="CI",
                run_id=run,
                attempt="1",
                job="build",
                variant="debug",
                level="gtest",
                group="fast/unit",
                report_name="report",
                relative_path=f"{day}/{run}/report.xml",
            ),
            **overrides,
        )

    def new_day(self):
        return dashboard.DayTestAggregate("test", "cpp", "gtest", "Suite", "test", "Suite::test")

    def add_day(self, compact, daily, latest="2026-09-17"):
        day = next(iter(daily.segments.values()))["date"]
        range_ids = [
            range_id
            for range_id, _, days in dashboard.RANGE_OPTIONS
            if dashboard.segments_for_range([{"date": day}], latest, days)
        ]
        compact.add_day(daily, range_ids)

    def test_daily_totals_and_inclusive_range_boundaries(self):
        compact = dashboard.TestAggregate()
        for index in range(40):
            day = (date(2026, 8, 9) + timedelta(days=index)).isoformat()
            daily = self.new_day()
            for minute, status in enumerate(["passed", "failed", "error", "skipped", "unknown"]):
                daily.add(
                    self.metadata(day, str(index)), status, f"{day}T12:0{minute}:00Z", 0.25, ""
                )
            self.add_day(compact, daily)

        self.assertEqual(40, len(compact.days))
        for range_id, _, length in dashboard.RANGE_OPTIONS:
            with self.subTest(range_id=range_id):
                days = dashboard.segments_for_range(compact.days, "2026-09-17", length)
                summary = compact.summary(days, range_id)
                count = length or 40
                self.assertEqual(count * 5, summary["total"])
                for status in ("passed", "failed", "errored", "skipped"):
                    self.assertEqual(count, summary[status])
                self.assertEqual(count * 2, summary["failures"])
                self.assertEqual(0.6667, summary["failure_rate"])
                self.assertEqual(0.25, summary["avg_time"])
                self.assertEqual(count, len(summary["active_dates"]))
                cutoff = (date(2026, 9, 17) - timedelta(days=count - 1)).isoformat()
                self.assertEqual(cutoff, summary["active_dates"][0])
                self.assertEqual(f"{cutoff}T12:00:00Z", summary["first_seen"])
                self.assertEqual("2026-09-17T12:02:00Z", summary["last_failed"])
                self.assertEqual("unknown", summary["last_status"])
                self.assertFalse(summary["is_currently_failing"])
                self.assertTrue(summary["is_flaky"])
                self.assertNotIn("date", summary)
                self.assertNotIn("total_time", summary)

    def test_started_failing_uses_runs_and_preserves_recovered_failure(self):
        daily = self.new_day()
        for run in range(12):
            status = "passed" if run < 9 else "failed"
            daily.add(self.metadata(run=str(run)), status, f"2026-09-17T12:{run:02}:00Z", 1.0, "")
        # Another report from the same run must update its status, not add a history dot.
        daily.add(self.metadata(run="11"), "passed", "2026-09-17T13:00:00Z", 1.0, "")
        compact = dashboard.TestAggregate()
        self.add_day(compact, daily)
        summary = compact.summary(compact.days, "all")
        detail = compact.details["all"]
        self.assertEqual(13, summary["total"])
        self.assertEqual(3, summary["failures"])
        self.assertTrue(summary["started_failing_in_sample"])
        self.assertFalse(summary["is_currently_failing"])
        self.assertEqual(12, len(detail["recent"]))
        self.assertEqual("passed", detail["recent"][-1]["status"])
        self.assertEqual(["11", "10", "9"], [run["run_id"] for run in detail["failure_runs"]])
        self.assertEqual([], detail["failure_examples"])

    def test_failure_runs_preserve_attempt_job_and_variant_groups(self):
        daily = self.new_day()
        variants = [{}, {"attempt": "2"}, {"job": "other"}, {"variant": "release"}]
        for index, overrides in enumerate(variants):
            meta = self.metadata(relative_path=f"report-{index}.xml", **overrides)
            for minute, status in enumerate(["failed", "error", "passed"]):
                daily.add(meta, status, f"2026-09-17T12:0{minute}:00Z", 0.5, "")
        compact = dashboard.TestAggregate()
        self.add_day(compact, daily)
        detail = compact.details["all"]
        summary = compact.summary(compact.days, "all")
        self.assertEqual(4, len(detail["recent"]))
        self.assertEqual(4, len(detail["failure_runs"]))
        self.assertTrue(all(run["failures"] == 2 for run in detail["failure_runs"]))
        self.assertEqual({"1", "2"}, {run["run_attempt"] for run in detail["failure_runs"]})
        self.assertEqual(
            {f"report-{index}.xml" for index in range(4)},
            {run["report"] for run in detail["failure_runs"]},
        )
        self.assertEqual(["debug", "release"], summary["active_variants"])
        self.assertEqual(8, summary["failures"])

    def test_bounded_samples_belong_to_their_range(self):
        compact = dashboard.TestAggregate()
        for index in range(40):
            day = (date(2026, 8, 9) + timedelta(days=index)).isoformat()
            daily = self.new_day()
            for run in range(30):
                # A report timestamp can cross midnight; membership uses the S3 day.
                timestamp_day = "2026-09-18" if index == 0 else day
                daily.add(
                    self.metadata(day, f"{index}-{run}"),
                    "failed",
                    f"{timestamp_day}T12:{run:02}:00Z",
                    1.0,
                    f"failure {index}-{run}",
                )
            self.add_day(compact, daily)
        for range_id, _, _ in dashboard.RANGE_OPTIONS:
            detail = compact.details[range_id]
            self.assertEqual(12, len(detail["recent"]))
            self.assertEqual(20, len(detail["failure_runs"]))
            self.assertEqual(4, len(detail["failure_examples"]))
            prefix = "0-" if range_id == "all" else "39-"
            self.assertTrue(all(run["run_id"].startswith(prefix) for run in detail["failure_runs"]))
            self.assertTrue(
                all(example["run_id"].startswith(prefix) for example in detail["failure_examples"])
            )

    def test_tied_timestamps_have_stable_run_order(self):
        daily = self.new_day()
        for run in reversed(range(20)):
            daily.add(
                self.metadata(run=f"{run:03}"),
                "failed" if run >= 17 else "passed",
                "2026-09-17T12:00:00Z",
                1.0,
                "",
            )
        compact = dashboard.TestAggregate()
        self.add_day(compact, daily)
        summary = compact.summary(compact.days, "all")
        detail = compact.details["all"]
        self.assertEqual("000", summary["last_run_id"])
        self.assertEqual("017", summary["last_failed_run_id"])
        self.assertEqual(
            ["passed"] * 9 + ["failed"] * 3, [item["status"] for item in detail["recent"]]
        )
        self.assertEqual(["017", "018", "019"], [run["run_id"] for run in detail["failure_runs"]])

    def build(self, source, output):
        with (
            patch("sys.argv", ["build_dashboard.py", str(source), str(output)]),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(0, dashboard.main())
        return json.loads((output / "manifest.json").read_text())

    def test_main_mixed_reports_and_parse_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input"
            suffix = Path("year=2026/month=09/day=17/CI/123/1/build/debug")
            xml_dir = source / "junit" / "regression" / suffix
            xml_dir.mkdir(parents=True)
            for index in range(3):
                failure = '<failure message="failed" />' if index < 2 else ""
                (xml_dir / f"report-{index}.xml").write_text(
                    f'<testsuite timestamp="2026-09-17T12:0{index}:00Z">'
                    f'<testcase classname="Suite" name="test" time="1">{failure}</testcase>'
                    "</testsuite>"
                )
            json_dir = source / "dashboard" / "cpp" / suffix
            json_dir.mkdir(parents=True)
            (json_dir / "gtest-summary.json").write_text(
                json.dumps(
                    {
                        "tests": [
                            {"classname": "Suite", "name": "test", "status": "failed", "time": 1},
                            {"classname": "Suite", "name": "test", "status": "passed", "time": 1},
                        ],
                        "parse_errors": [{"file": "broken.xml", "error": "invalid XML"}],
                    }
                )
            )
            malformed = json_dir / "bad" / "gtest-summary.json"
            malformed.parent.mkdir()
            malformed.write_text("{")
            output = root / "output"
            manifest = self.build(source, output)
            self.assertEqual(dashboard.SCHEMA_VERSION, manifest["schema_version"])
            self.assertEqual(
                {
                    "xml_files": 3,
                    "dashboard_json_files": 2,
                    "reports_passed": 1,
                    "reports_failed": 3,
                    "parse_errors": 2,
                    "runs": 1,
                    "unique_tests": 2,
                    "test_occurrences": 5,
                },
                manifest["totals"],
            )
            rows = json.loads((output / "ranges" / "all.json").read_text())["tests"]
            for row in rows:
                detail = json.loads((output / row["detail_file"]).read_text())
                self.assertNotIn("segments", detail)
                self.assertEqual({"all", "7", "14", "30"}, detail["ranges"].keys())
                self.assertEqual("passed", row["last_status"])
                for samples in detail["ranges"].values():
                    self.assertEqual(1, len(samples["recent"]))
                    self.assertEqual(1, len(samples["failure_runs"]))
                    self.assertEqual(row["failures"], samples["failure_runs"][0]["failures"])
                    self.assertEqual("123", samples["failure_runs"][0]["run_id"])

    def test_empty_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input"
            source.mkdir()
            output = root / "output"
            manifest = self.build(source, output)
            self.assertEqual(0, manifest["totals"]["unique_tests"])
            self.assertEqual({"first": None, "last": None, "days": []}, manifest["date_range"])
            for range_id, _, _ in dashboard.RANGE_OPTIONS:
                summary = json.loads((output / "ranges" / f"{range_id}.json").read_text())
                self.assertEqual([], summary["tests"])


if __name__ == "__main__":
    unittest.main()
