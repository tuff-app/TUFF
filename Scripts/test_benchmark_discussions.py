#!/usr/bin/env python3
"""Tests for the benchmark review and leaderboard builder. No network."""

import copy
import json
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark_discussions as bd  # noqa: E402

NOW = datetime(2026, 10, 9, 12, tzinfo=timezone.utc)


def trial(workload, number, prompt=1300, cached=0, generated=None, prefill=2.0, decode=None):
    generated = generated if generated is not None else bd.WORKLOAD_MAX_NEW[workload]
    decode = decode if decode is not None else (generated - 1) / 10
    return {"workload": workload, "trial": number, "prompt_tokens": prompt,
            "cached_tokens": cached, "generated_tokens": generated,
            "prefill_seconds": prefill, "decode_seconds": decode,
            "time_to_first_token_seconds": prefill + 0.01, "stop_reason": "maxTokens",
            "peak_memory_bytes": 1 << 30}


def result(model="gemma4-26b-a4b", rate=10.0, run_id="3f2440f6-73bf-4275-aaaf-c11b4096d5bd",
           chip="Apple M2", memory=16):
    trials = [trial("check", 1, prompt=21, generated=2)]
    trials += [trial("short", n, prompt=45, decode=127 / rate) for n in (1, 2, 3)]
    trials += [trial("long", n) for n in (1, 2, 3)]
    trials += [trial("follow-up", 1, prompt=1400, cached=1350, prefill=0.2)]
    return {
        "schema": "tuff-benchmark/1", "id": run_id,
        "suite": {"name": "tuff-bench", "version": 1, "mode": "standard",
                  "workload_sha256": bd.CONFIG["suites"]["tuff-bench/1"]},
        "app": {"version": "8.1.0", "build": "release"},
        "machine": {"chip": chip, "model_identifier": "Mac14,2", "memory_bytes": memory << 30,
                    "performance_cores": 4, "efficiency_cores": 4, "gpu_cores": 10,
                    "mac_os_version": "26.6.2"},
        "started_at": "2026-10-09T08:00:00Z", "finished_at": "2026-10-09T08:10:00Z",
        "runs": [{"model": {"id": model, "name": "x", "revision": "r", "weights": "affine",
                            "installed_bytes": 1},
                  "settings": {"context_tokens": 4096}, "status": "completed",
                  "load_seconds": 4.2, "check": {"passed": True, "answer": "Paris"},
                  "trials": trials}],
    }


def body(data):
    return "Benchmarked with TUFF.\n\n" + bd.FENCE.pattern[:0] + \
        "```json tuff-benchmark\n" + json.dumps(data) + "\n```\n"


def post(number, data=None, author="alice", age_days=400, labels=(), created=None, raw=None):
    return bd.Post(number=number, url=f"https://github.com/x/y/discussions/{number}",
                   title="t", body=raw if raw is not None else body(data or result()),
                   author=author, author_created_at=NOW - timedelta(days=age_days),
                   created_at=created or NOW - timedelta(hours=2), labels=set(labels))


class ReviewTests(unittest.TestCase):
    def test_a_normal_result_is_accepted(self):
        decision = bd.review(post(1), [], NOW)
        self.assertEqual(decision.verdict, "community", decision.reasons)

    def test_a_warm_or_battery_run_is_accepted_with_a_note(self):
        data = result()
        self.assertEqual(bd.validate_structure(data), [])
        data["runs"][0]["conditions"] = {"thermal_state_at_start": "fair",
                                         "thermal_state_at_end": "serious",
                                         "power_source": "battery", "low_power_mode": False,
                                         "cooldown_seconds": 20}
        notes = bd.validate_structure(data)
        self.assertEqual(notes, ["`gemma4-26b-a4b` started warm (fair), ended hot (serious), "
                                 "ran on battery, so it may be slower than usual."])
        self.assertEqual(bd.review(post(1, data), [], NOW).verdict, "community")
        data["runs"][0]["conditions"] = {"thermal_state_at_start": "nominal",
                                         "power_source": "ac", "low_power_mode": False}
        self.assertEqual(bd.validate_structure(data), [])

    def test_malformed_posts_are_rejected_with_a_reason(self):
        cases = {
            "no block": post(1, raw="just text"),
            "bad json": post(1, raw="```json tuff-benchmark\n{oops\n```"),
            "two blocks": post(1, raw=body(result()) + body(result())),
        }
        for name, item in cases.items():
            with self.subTest(name):
                decision = bd.review(item, [], NOW)
                self.assertEqual(decision.verdict, "rejected")
                self.assertTrue(decision.reasons[0])

    def test_a_pasted_result_file_is_read_as_is(self):
        pasted = json.dumps(result(), indent=2, separators=(",", " : "))
        decision = bd.review(post(1, raw=pasted), [], NOW)
        self.assertEqual(decision.verdict, "community", decision.reasons)
        # Any other bare JSON is still not a result.
        decision = bd.review(post(1, raw=json.dumps({"schema": "other"})), [], NOW)
        self.assertEqual(decision.verdict, "rejected")

    def test_an_unchanged_verdict_leaves_the_comment_alone(self):
        calls = []
        original_graphql, original_labels = bd.graphql, bd.ensure_labels
        bd.ensure_labels = lambda: {label: label for label in bd.LABELS}
        try:
            def fake(query, **variables):
                calls.append(query)
                if "updateDiscussionComment" in query:
                    raise RuntimeError("update refused")
                return {}
            bd.graphql = fake
            item = post(1, raw="just text", labels={bd.LABEL_REJECTED})
            decision = bd.review(item, [], NOW)
            item.bot_comment_id = "c1"
            item.bot_comment_body = bd.comment_body(decision)
            bd.apply_review(item, decision)
            self.assertEqual(calls, [])
            # A changed verdict that cannot be edited in is posted as new.
            item.bot_comment_body = "old"
            bd.apply_review(item, decision)
            self.assertTrue(any("addDiscussionComment" in query for query in calls))
        finally:
            bd.graphql, bd.ensure_labels = original_graphql, original_labels

    def test_fields_that_were_tampered_with_are_rejected(self):
        def mutate(change):
            data = result()
            change(data)
            return bd.review(post(1, data), [], NOW)

        cases = {
            "schema": lambda d: d.update(schema="tuff-benchmark/9"),
            "workload hash": lambda d: d["suite"].update(workload_sha256="0" * 64),
            "old app": lambda d: d["app"].update(version="8.0.2"),
            "chip": lambda d: d["machine"].update(chip="<script>alert(1)</script>"),
            "memory": lambda d: d["machine"].update(memory_bytes=12345),
            "model": lambda d: d["runs"][0]["model"].update(id="llama-9000"),
            "too many tokens": lambda d: d["runs"][0]["trials"][1].update(generated_tokens=5000),
            "negative time": lambda d: d["runs"][0]["trials"][1].update(decode_seconds=-1),
            "nan": lambda d: d["runs"][0]["trials"][1].update(prefill_seconds=float("nan")),
            "cold reuse": lambda d: d["runs"][0]["trials"][1].update(cached_tokens=40, prompt_tokens=45) or
                                    d["runs"][0]["trials"][4].update(cached_tokens=900),
            "time travel": lambda d: d.update(finished_at="2026-10-09T07:00:00Z"),
            "trials outlast run": lambda d: d.update(finished_at="2026-10-09T08:00:05Z"),
            "duplicate model": lambda d: d["runs"].append(copy.deepcopy(d["runs"][0])),
            "nothing completed": lambda d: d["runs"][0].update(status="failed"),
            "string number": lambda d: d["runs"][0]["trials"][1].update(prompt_tokens="45"),
            "huge number": lambda d: d["runs"][0]["trials"][1].update(prompt_tokens=10 ** 400),
            "no time zone": lambda d: d.update(started_at="2026-10-09T08:00:00"),
            "not a time": lambda d: d.update(finished_at="soon"),
            "list model id": lambda d: d["runs"][0]["model"].update(id=[]),
            "list workload": lambda d: d["runs"][0]["trials"][1].update(workload=[]),
            "string load time": lambda d: d["runs"][0].update(load_seconds="3"),
            "string peak memory": lambda d: d["runs"][0]["trials"][1].update(peak_memory_bytes="1"),
            "check not an object": lambda d: d["runs"][0].update(check=1),
            "settings not an object": lambda d: d["runs"][0].update(settings="x"),
        }
        for name, change in cases.items():
            with self.subTest(name):
                self.assertEqual(mutate(change).verdict, "rejected")

    def test_a_replayed_run_or_copied_numbers_are_rejected(self):
        first = post(1, created=NOW - timedelta(days=1), labels={bd.LABEL_COMMUNITY})
        replay = bd.review(post(2, author="bob"), [(first, result())], NOW)
        self.assertEqual(replay.verdict, "rejected")
        self.assertIn("#1", replay.reasons[0])
        copied = result(run_id="11111111-2222-3333-4444-555555555555")
        decision = bd.review(post(3, copied, author="bob"), [(first, result())], NOW)
        self.assertEqual(decision.verdict, "rejected")

    def test_implausible_or_new_accounts_are_held_for_review(self):
        fast = bd.review(post(1, result(model="gemma4-26b-a4b", rate=900)), [], NOW)
        self.assertEqual(fast.verdict, "needs review")
        new = bd.review(post(1, age_days=2), [], NOW)
        self.assertEqual(new.verdict, "needs review")
        broken = result()
        broken["runs"][0]["check"]["passed"] = False
        self.assertEqual(bd.review(post(1, broken), [], NOW).verdict, "needs review")

    def test_too_many_posts_in_a_day_are_held(self):
        others = [(post(n, result(run_id=f"00000000-0000-0000-0000-00000000000{n}", rate=10 + n)),
                   result(run_id=f"00000000-0000-0000-0000-00000000000{n}", rate=10 + n))
                  for n in range(1, 6)]
        decision = bd.review(post(9, result(rate=30)), others, NOW)
        self.assertEqual(decision.verdict, "needs review")

    def test_outliers_against_other_peoples_results_are_held(self):
        peers = []
        for n, person in enumerate(["bob", "carol", "dave"], start=1):
            data = result(run_id=f"00000000-0000-0000-0000-00000000000{n}", rate=10 + n * 0.1)
            peers.append((post(n, data, author=person, labels={bd.LABEL_COMMUNITY}), data))
        normal = bd.review(post(9, result(rate=11)), peers, NOW)
        self.assertEqual(normal.verdict, "community", normal.reasons)
        outlier = bd.review(post(9, result(rate=40)), peers, NOW)
        self.assertEqual(outlier.verdict, "needs review")
        # A different Mac is not compared.
        other_mac = bd.review(post(9, result(rate=40, chip="Apple M4 Max", memory=64)), peers, NOW)
        self.assertEqual(other_mac.verdict, "community", other_mac.reasons)

    def test_comments_never_echo_post_content_as_markup(self):
        decision = bd.review(post(1, raw="<img src=x onerror=alert(1)>"), [], NOW)
        self.assertNotIn("<img", bd.comment_body(decision))


class DatasetTests(unittest.TestCase):
    def test_one_person_cannot_outvote_others(self):
        posts = []
        for n in range(1, 11):
            data = result(run_id=f"00000000-0000-0000-0000-0000000000{n:02d}", rate=50)
            posts.append((post(n, data, author="spammer", labels={bd.LABEL_COMMUNITY}), data))
        for n, person in enumerate(["bob", "carol"], start=20):
            data = result(run_id=f"00000000-0000-0000-0000-0000000000{n}", rate=10)
            posts.append((post(n, data, author=person, labels={bd.LABEL_COMMUNITY}), data))
        dataset = bd.build_dataset(posts, NOW)
        group = dataset["groups"][0]
        self.assertEqual(group["contributors"], 3)
        self.assertEqual(group["results"], 12)
        self.assertEqual(group["decode"], 10.0)

    def test_held_results_are_listed_but_not_ranked(self):
        data = result()
        dataset = bd.build_dataset([(post(1, data, labels={bd.LABEL_REVIEW}), data)], NOW)
        self.assertEqual(dataset["entries"][0]["trust"], "needs-review")
        self.assertEqual(dataset["groups"], [])

    def test_entries_carry_the_derived_rates(self):
        data = result(rate=12)
        entry = bd.build_dataset([(post(1, data, labels={bd.LABEL_VERIFIED}), data)], NOW)["entries"][0]
        self.assertEqual(entry["trust"], "verified")
        self.assertEqual(entry["decode"], 12.0)
        self.assertEqual(entry["memory_gb"], 16)
        self.assertEqual(entry["followup_reused"], 1350)
        self.assertNotIn("body", entry)


if __name__ == "__main__":
    unittest.main()
