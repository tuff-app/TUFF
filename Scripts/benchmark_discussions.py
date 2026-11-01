#!/usr/bin/env python3
"""Reviews benchmark posts in GitHub Discussions and builds the leaderboard.

Two commands: review runs in .github/workflows/benchmark-review.yml, and build
runs in the website's deploy workflow (tuff-app/tuff-app.github.io):

  review --discussion N   Validate one post, label it and leave a comment.
  build --output FILE     Collect every accepted post into the leaderboard data.

Posts are untrusted. Their JSON is parsed as data and checked against the
published suite; nothing in a post is ever executed or rendered as HTML.
Results live only in Discussions and in the deployed site, never in the
repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG = json.loads((ROOT / ".github" / "benchmark-suite.json").read_text())

FENCE = re.compile(r"```json tuff-benchmark\s*\n(.*?)\n```", re.DOTALL)
BOT_MARKER = "<!-- tuff-benchmark-review -->"

LABEL_COMMUNITY = "benchmark: community"
LABEL_REVIEW = "benchmark: needs review"
LABEL_REJECTED = "benchmark: rejected"
LABEL_VERIFIED = "benchmark: verified"
LABELS = {
    LABEL_COMMUNITY: ("0e8a16", "Benchmark that passed the automatic checks"),
    LABEL_REVIEW: ("fbca04", "Benchmark held for a maintainer to look at"),
    LABEL_REJECTED: ("d73a4a", "Benchmark the automatic checks could not accept"),
    LABEL_VERIFIED: ("1d76db", "Benchmark a maintainer reproduced or checked"),
}
BOT_LABELS = {LABEL_COMMUNITY, LABEL_REVIEW, LABEL_REJECTED}

MAX_BODY = 65_000
MAX_RUNS = 12
MAX_TRIALS = 20
MIN_ACCOUNT_AGE = timedelta(days=7)
MAX_POSTS_PER_DAY = 5
# How far from comparable results a decode rate can be before a person looks.
OUTLIER_HIGH = 2.0
OUTLIER_LOW = 0.33
OUTLIER_MIN_SAMPLES = 3
MAX_PREFILL_TOKENS_PER_SECOND = 20_000
WORKLOAD_MAX_NEW = {"check": 96, "short": 128, "long": 48, "follow-up": 48}

CHIP = re.compile(r"^Apple M\d{1,2}( (Pro|Max|Ultra))?$")
MODEL_IDENTIFIER = re.compile(r"^[A-Za-z]{2,20}\d{1,3},\d{1,3}$")
MACOS = re.compile(r"^\d{2}(\.\d{1,2}){0,2}$")
UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
VERSION = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")


# ---------------------------------------------------------------------------
# Parsing and validation (pure; covered by Scripts/test_benchmark_discussions.py)


class Rejected(Exception):
    """The post cannot be accepted, for the reason given."""


@dataclass
class Post:
    number: int
    url: str
    title: str
    body: str
    author: str
    author_created_at: datetime | None
    created_at: datetime
    labels: set[str] = field(default_factory=set)
    discussion_id: str = ""
    bot_comment_id: str | None = None
    bot_comment_body: str = ""


@dataclass
class Review:
    verdict: str  # "community", "needs review" or "rejected"
    reasons: list[str]
    result: dict | None = None

    @property
    def label(self) -> str:
        return {"community": LABEL_COMMUNITY, "needs review": LABEL_REVIEW,
                "rejected": LABEL_REJECTED}[self.verdict]


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def extract_result(body: str) -> dict:
    if len(body) > MAX_BODY:
        raise Rejected("The post is too long.")
    matches = FENCE.findall(body)
    if not matches:
        # People paste the saved result file instead of the shared post. The
        # data is the same, so a body that is only that file is read as is.
        try:
            pasted = json.loads(body.strip())
        except json.JSONDecodeError:
            pasted = None
        if isinstance(pasted, dict) and pasted.get("schema") == "tuff-benchmark/1":
            return pasted
        raise Rejected("No result data found. Share from TUFF's Benchmarks screen or "
                       "`tuff bench --share` and paste the post it copies, keeping the "
                       "`json tuff-benchmark` block as it is.")
    if len(matches) > 1:
        raise Rejected("The post has more than one result block. Post one run per discussion.")
    try:
        result = json.loads(matches[0])
    except json.JSONDecodeError as error:
        raise Rejected(f"The result data is not valid JSON ({error.msg}). "
                       "It may have been edited after sharing.") from None
    if not isinstance(result, dict):
        raise Rejected("The result data is not a JSON object.")
    return result


def number(value, name: str, minimum: float = 0, maximum: float = math.inf,
           integer: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Rejected(f"`{name}` is not a number.")
    if integer and not isinstance(value, int):
        raise Rejected(f"`{name}` is not a whole number.")
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise Rejected(f"`{name}` is out of range.")
    return value


def text(value, name: str, pattern: re.Pattern | None = None, limit: int = 120) -> str:
    if not isinstance(value, str) or not value or len(value) > limit:
        raise Rejected(f"`{name}` is missing or too long.")
    if pattern and not pattern.match(value):
        raise Rejected(f"`{name}` has an unexpected value.")
    return value


def version_tuple(value: str) -> tuple[int, int, int]:
    match = VERSION.match(value)
    if not match:
        raise Rejected("`app.version` is not a release version.")
    return tuple(int(part) for part in match.groups())  # type: ignore[return-value]


def decode_rate(trial: dict) -> float | None:
    generated = trial["generated_tokens"]
    seconds = trial["decode_seconds"]
    if generated <= 1 or seconds <= 0:
        return None
    return (generated - 1) / seconds


def prefill_rate(trial: dict) -> float | None:
    processed = trial["prompt_tokens"] - trial["cached_tokens"]
    if processed <= 0 or trial["prefill_seconds"] <= 0:
        return None
    return processed / trial["prefill_seconds"]


def median(values: list[float]) -> float | None:
    values = [value for value in values if value is not None]
    return statistics.median(values) if values else None


def validate_structure(result: dict) -> list[str]:
    """Raises Rejected for anything malformed; returns notes for a person."""
    notes: list[str] = []
    if result.get("schema") != "tuff-benchmark/1":
        raise Rejected("Unknown result format. Update TUFF and run the benchmark again.")
    text(result.get("id"), "id", UUID)

    suite = result.get("suite")
    if not isinstance(suite, dict):
        raise Rejected("`suite` is missing.")
    key = f"{suite.get('name')}/{suite.get('version')}"
    if key not in CONFIG["suites"]:
        raise Rejected(f"Unknown benchmark suite `{key}`.")
    if suite.get("workload_sha256") != CONFIG["suites"][key]:
        raise Rejected("The benchmark prompts do not match the published suite.")
    if suite.get("mode") not in ("quick", "standard"):
        raise Rejected("Unknown benchmark mode.")

    app = result.get("app")
    if not isinstance(app, dict):
        raise Rejected("`app` is missing.")
    if version_tuple(text(app.get("version"), "app.version")) < version_tuple(CONFIG["minimum_app_version"]):
        raise Rejected(f"Benchmarks need TUFF {CONFIG['minimum_app_version']} or newer.")
    if app.get("build") not in ("release", "source"):
        raise Rejected("`app.build` has an unexpected value.")
    if app["build"] == "source":
        notes.append("Built from source, so the code may differ from a release.")

    machine = result.get("machine")
    if not isinstance(machine, dict):
        raise Rejected("`machine` is missing.")
    text(machine.get("chip"), "machine.chip", CHIP)
    text(machine.get("model_identifier"), "machine.model_identifier", MODEL_IDENTIFIER)
    text(machine.get("mac_os_version"), "machine.mac_os_version", MACOS)
    memory = number(machine.get("memory_bytes"), "machine.memory_bytes",
                    8 << 30, 1 << 40, integer=True)
    if memory % (1 << 30):
        raise Rejected("`machine.memory_bytes` is not a whole number of gigabytes.")
    for key in ("performance_cores", "efficiency_cores", "gpu_cores"):
        if machine.get(key) is not None:
            number(machine[key], f"machine.{key}", 0, 256, integer=True)

    started = parse_time(text(result.get("started_at"), "started_at"))
    finished = parse_time(text(result.get("finished_at"), "finished_at"))
    if finished < started:
        raise Rejected("The run finishes before it starts.")
    if started > datetime.now(timezone.utc) + timedelta(days=1):
        raise Rejected("The run is dated in the future.")

    runs = result.get("runs")
    if not isinstance(runs, list) or not 1 <= len(runs) <= MAX_RUNS:
        raise Rejected("`runs` is missing or has too many entries.")
    seen: set[str] = set()
    measured = 0.0
    completed = 0
    for run in runs:
        if not isinstance(run, dict) or not isinstance(run.get("model"), dict):
            raise Rejected("A run is malformed.")
        model_id = run["model"].get("id")
        if model_id not in CONFIG["models"]:
            raise Rejected(f"Unknown model `{model_id}`.")
        if model_id in seen:
            raise Rejected(f"`{model_id}` appears twice. Each model runs once per post.")
        seen.add(model_id)
        if run.get("status") not in ("completed", "failed", "skipped", "cancelled"):
            raise Rejected("A run has an unknown status.")
        trials = run.get("trials")
        if not isinstance(trials, list) or len(trials) > MAX_TRIALS:
            raise Rejected("A run's trials are malformed.")
        for trial in trials:
            measured += validate_trial(trial, model_id)
        if run["status"] == "completed":
            completed += 1
            workloads = [trial["workload"] for trial in trials]
            if "short" not in workloads or "long" not in workloads:
                raise Rejected(f"`{model_id}` is marked completed but is missing trials.")
    if completed == 0:
        raise Rejected("No model completed, so there is nothing to compare.")
    notes += condition_notes(runs)
    if (finished - started).total_seconds() + 1 < measured:
        raise Rejected("The trials take longer than the whole run.")
    return notes


def condition_notes(runs: list) -> list[str]:
    """TUFF 10.0.2 and later record how the Mac was running for each model."""
    notes = []
    for run in runs:
        conditions = run.get("conditions")
        if run.get("status") != "completed" or not isinstance(conditions, dict):
            continue
        reasons = []
        if conditions.get("thermal_state_at_start") not in (None, "nominal"):
            reasons.append(f"started warm ({conditions['thermal_state_at_start']})")
        if conditions.get("thermal_state_at_end") in ("serious", "critical"):
            reasons.append(f"ended hot ({conditions['thermal_state_at_end']})")
        if conditions.get("power_source") == "battery":
            reasons.append("ran on battery")
        if conditions.get("low_power_mode") is True:
            reasons.append("ran in Low Power Mode")
        if reasons:
            notes.append(f"`{run['model'].get('id')}` {', '.join(reasons)}, so it may be slower than usual.")
    return notes


def validate_trial(trial, model_id: str) -> float:
    if not isinstance(trial, dict):
        raise Rejected("A trial is malformed.")
    workload = trial.get("workload")
    if workload not in WORKLOAD_MAX_NEW:
        raise Rejected("A trial has an unknown workload.")
    number(trial.get("trial"), "trial", 1, MAX_TRIALS, integer=True)
    prompt = number(trial.get("prompt_tokens"), "prompt_tokens", 1, 70_000, integer=True)
    cached = number(trial.get("cached_tokens"), "cached_tokens", 0, 70_000, integer=True)
    generated = number(trial.get("generated_tokens"), "generated_tokens", 0,
                       WORKLOAD_MAX_NEW[workload], integer=True)
    prefill = number(trial.get("prefill_seconds"), "prefill_seconds", 0, 36_000)
    decode = number(trial.get("decode_seconds"), "decode_seconds", 0, 36_000)
    if cached > prompt:
        raise Rejected("A trial reused more tokens than its prompt had.")
    if workload != "follow-up" and cached > 64:
        raise Rejected("A cold trial reused a conversation it should not have.")
    if generated > 1 and decode <= 0:
        raise Rejected("A trial generated tokens in no time.")
    if trial.get("time_to_first_token_seconds") is not None:
        ttft = number(trial["time_to_first_token_seconds"], "time_to_first_token_seconds", 0, 36_000)
        if ttft + 0.001 < prefill:
            raise Rejected("A first token arrives before its prompt finished.")
    del model_id
    return prefill + decode


def plausibility_flags(result: dict) -> list[str]:
    flags = []
    for run in result["runs"]:
        if run["status"] != "completed":
            continue
        model = CONFIG["models"][run["model"]["id"]]
        decode = median([decode_rate(t) for t in run["trials"] if t["workload"] == "short"])
        prefill = median([prefill_rate(t) for t in run["trials"] if t["workload"] == "long"])
        if decode and decode > model["max_decode_tokens_per_second"]:
            flags.append(f"{model['name']} decodes at {decode:.1f} tok/s, faster than any Mac should.")
        if prefill and prefill > MAX_PREFILL_TOKENS_PER_SECOND:
            flags.append(f"{model['name']} reads prompts at {prefill:.0f} tok/s, which is implausible.")
        check = run.get("check") or {}
        if check.get("passed") is False:
            flags.append(f"{model['name']} failed the answer check, so it may be broken on this Mac.")
    return flags


def fingerprint(result: dict) -> str:
    """Identical measurements, whatever the id says."""
    trials = [
        [run["model"]["id"]] + [
            [t["workload"], t["trial"], t["prompt_tokens"], t["generated_tokens"],
             round(t["prefill_seconds"], 6), round(t["decode_seconds"], 6)]
            for t in run["trials"]
        ]
        for run in result["runs"]
    ]
    return hashlib.sha256(json.dumps(trials, sort_keys=True).encode()).hexdigest()


def group_key(result: dict, run: dict) -> tuple:
    machine = result["machine"]
    return (run["model"]["id"], machine["chip"], machine["memory_bytes"] >> 30)


def review(post: Post, others: list[tuple[Post, dict]], now: datetime) -> Review:
    """Decides one post against the accepted ones before it."""
    try:
        result = extract_result(post.body)
        notes = validate_structure(result)
    except Rejected as reason:
        return Review("rejected", [str(reason)])

    for other_post, other in others:
        if other_post.number == post.number:
            continue
        if other.get("id") == result["id"]:
            return Review("rejected", [f"This run was already posted in #{other_post.number}."], result)
        if fingerprint(other) == fingerprint(result):
            return Review("rejected", [f"These measurements match #{other_post.number} exactly."], result)

    flags = plausibility_flags(result)
    if post.author_created_at and now - post.author_created_at < MIN_ACCOUNT_AGE:
        flags.append("The posting account is less than a week old.")
    recent = [p for p, _ in others
              if p.author == post.author and p.number != post.number
              and now - p.created_at < timedelta(days=1)]
    if len(recent) >= MAX_POSTS_PER_DAY:
        flags.append("This account has posted many benchmarks in the last day.")

    # Compared only with other people's accepted results on the same chip
    # and memory, so nobody can move their own baseline.
    for run in result["runs"]:
        if run["status"] != "completed":
            continue
        decode = median([decode_rate(t) for t in run["trials"] if t["workload"] == "short"])
        if not decode:
            continue
        peers = [
            median([decode_rate(t) for t in other_run["trials"] if t["workload"] == "short"])
            for other_post, other in others
            if other_post.author != post.author and accepted(other_post)
            for other_run in other["runs"]
            if other_run["status"] == "completed" and group_key(other, other_run) == group_key(result, run)
        ]
        peers = [p for p in peers if p]
        if len(peers) >= OUTLIER_MIN_SAMPLES:
            typical = statistics.median(peers)
            if decode > typical * OUTLIER_HIGH or decode < typical * OUTLIER_LOW:
                name = CONFIG["models"][run["model"]["id"]]["name"]
                flags.append(f"{name} is far from {len(peers)} other results on the same Mac "
                             f"({decode:.1f} vs a typical {typical:.1f} tok/s).")

    if flags:
        return Review("needs review", flags + notes, result)
    return Review("community", notes, result)


def accepted(post: Post) -> bool:
    return bool(post.labels & {LABEL_COMMUNITY, LABEL_VERIFIED}) and LABEL_REJECTED not in post.labels


def comment_body(review_: Review) -> str:
    lines = [BOT_MARKER]
    if review_.verdict == "community":
        lines.append("Thanks! This result passed the automatic checks and is on the "
                     "[leaderboard](https://tuff-app.github.io/benchmarks/) "
                     "(it can take a few minutes to appear).")
    elif review_.verdict == "needs review":
        lines.append("Thanks! This result is held for a maintainer to look at before it "
                     "appears on the leaderboard:")
    else:
        lines.append("This result could not be accepted:")
    lines.extend(f"- {reason}" for reason in review_.reasons)
    if review_.verdict == "rejected":
        lines.append("\nEditing the post runs the checks again.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Leaderboard data


def build_dataset(posts: list[tuple[Post, dict]], now: datetime) -> dict:
    entries = []
    for post, result in posts:
        verified = LABEL_VERIFIED in post.labels
        trust = "verified" if verified else ("community" if LABEL_COMMUNITY in post.labels else "needs-review")
        machine = result["machine"]
        for run in result["runs"]:
            if run["status"] != "completed":
                continue
            trials = run["trials"]
            follow = next((t for t in trials if t["workload"] == "follow-up"), None)
            long_ttft = median([t.get("time_to_first_token_seconds") for t in trials if t["workload"] == "long"])
            entries.append({
                "post": post.number,
                "url": post.url,
                "author": post.author,
                "posted": post.created_at.isoformat(),
                "trust": trust,
                "model": run["model"]["id"],
                "model_name": CONFIG["models"][run["model"]["id"]]["name"],
                "chip": machine["chip"],
                "memory_gb": machine["memory_bytes"] >> 30,
                "gpu_cores": machine.get("gpu_cores"),
                "mac": machine["model_identifier"],
                "macos": machine["mac_os_version"],
                "tuff": result["app"]["version"],
                "build": result["app"]["build"],
                "mode": result["suite"]["mode"],
                "suite": f"{result['suite']['name']}/{result['suite']['version']}",
                "context": (run.get("settings") or {}).get("context_tokens"),
                "decode": rounded(median([decode_rate(t) for t in trials if t["workload"] == "short"])),
                "decode_spread": spread([decode_rate(t) for t in trials if t["workload"] == "short"]),
                "prefill": rounded(median([prefill_rate(t) for t in trials if t["workload"] == "long"])),
                "ttft": rounded(long_ttft),
                "followup_ttft": rounded(follow.get("time_to_first_token_seconds") if follow else None),
                "followup_reused": follow["cached_tokens"] if follow else None,
                "load_seconds": rounded(run.get("load_seconds")),
                "memory_peak_gb": rounded((max([t.get("peak_memory_bytes") or 0 for t in trials]) or 0) / (1 << 30)),
                "check": (run.get("check") or {}).get("passed"),
                "trials": sum(1 for t in trials if t["workload"] == "short"),
            })
    return {
        "generated": now.isoformat(),
        "suite": sorted(CONFIG["suites"]),
        "models": {key: value["name"] for key, value in CONFIG["models"].items()},
        "entries": entries,
        "groups": aggregate(entries),
    }


def aggregate(entries: list[dict]) -> list[dict]:
    """One row per model, chip, memory and TUFF version. Each person counts
    once per group (their own median), so posting the same thing many times
    cannot move a ranking."""
    groups: dict[tuple, dict[str, list[dict]]] = {}
    for entry in entries:
        if entry["trust"] == "needs-review":
            continue
        key = (entry["model"], entry["chip"], entry["memory_gb"], entry["tuff"])
        groups.setdefault(key, {}).setdefault(entry["author"], []).append(entry)
    rows = []
    for (model, chip, memory, tuff), people in groups.items():
        def per_person(metric: str) -> list[float]:
            return [v for v in (median([e[metric] for e in mine if e[metric] is not None])
                                for mine in people.values()) if v is not None]
        decode = per_person("decode")
        rows.append({
            "model": model, "chip": chip, "memory_gb": memory, "tuff": tuff,
            "decode": rounded(median(decode)),
            "decode_low": rounded(min(decode)) if decode else None,
            "decode_high": rounded(max(decode)) if decode else None,
            "prefill": rounded(median(per_person("prefill"))),
            "ttft": rounded(median(per_person("ttft"))),
            "followup_ttft": rounded(median(per_person("followup_ttft"))),
            "contributors": len(people),
            "results": sum(len(mine) for mine in people.values()),
            "verified": any(e["trust"] == "verified" for mine in people.values() for e in mine),
        })
    rows.sort(key=lambda row: (row["model"], -(row["decode"] or 0)))
    return rows


def rounded(value):
    if value is None:
        return None
    return round(value, 3 if value < 10 else 1)


def spread(values: list) -> list | None:
    values = [v for v in values if v is not None]
    return [rounded(min(values)), rounded(max(values))] if len(values) > 1 else None


# ---------------------------------------------------------------------------
# GitHub (thin wrapper around `gh api graphql`)


def graphql(query: str, **variables) -> dict:
    command = ["gh", "api", "graphql", "-f", f"query={query}"]
    for name, value in variables.items():
        if isinstance(value, list):
            for item in value:
                command += ["-f", f"{name}[]={item}"]
        else:
            flag = "-F" if isinstance(value, int) else "-f"
            command += [flag, f"{name}={value}"]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError("gh api graphql failed: "
                           + (completed.stderr.strip() or completed.stdout.strip()))
    data = json.loads(completed.stdout)
    if data.get("errors"):
        raise RuntimeError(data["errors"])
    return data["data"]


def owner_and_name() -> tuple[str, str]:
    repository = os.environ.get("GITHUB_REPOSITORY", "tuff-app/TUFF")
    owner, name = repository.split("/", 1)
    return owner, name


POSTS_QUERY = """
query($owner: String!, $name: String!, $category: ID!, $after: String) {
  repository(owner: $owner, name: $name) {
    discussions(first: 50, after: $after, categoryId: $category,
                orderBy: {field: CREATED_AT, direction: ASC}) {
      pageInfo { hasNextPage endCursor }
      nodes {
        id number url title body createdAt
        author { login ... on User { createdAt } }
        labels(first: 20) { nodes { name } }
        comments(first: 50) { nodes { id body author { login } } }
      }
    }
  }
}"""


def category_id() -> str | None:
    owner, name = owner_and_name()
    data = graphql("""query($owner: String!, $name: String!) {
      repository(owner: $owner, name: $name) {
        discussionCategories(first: 50) { nodes { id slug } } } }""", owner=owner, name=name)
    for node in data["repository"]["discussionCategories"]["nodes"]:
        if node["slug"] == CONFIG["category"]:
            return node["id"]
    return None


def fetch_posts() -> list[Post]:
    owner, name = owner_and_name()
    category = category_id()
    if not category:
        return []
    posts, after = [], None
    while True:
        variables = {"owner": owner, "name": name, "category": category}
        if after:
            variables["after"] = after
        page = graphql(POSTS_QUERY, **variables)["repository"]["discussions"]
        for node in page["nodes"]:
            author = node.get("author") or {}
            bot = next((c for c in node["comments"]["nodes"]
                        if BOT_MARKER in (c.get("body") or "")
                        and (c.get("author") or {}).get("login") == "github-actions"), None)
            posts.append(Post(
                number=node["number"], url=node["url"], title=node["title"],
                body=node["body"] or "", author=author.get("login", "ghost"),
                author_created_at=parse_time(author["createdAt"]) if author.get("createdAt") else None,
                created_at=parse_time(node["createdAt"]),
                labels={label["name"] for label in node["labels"]["nodes"]},
                discussion_id=node["id"], bot_comment_id=bot["id"] if bot else None,
                bot_comment_body=bot["body"] if bot else ""))
        if not page["pageInfo"]["hasNextPage"]:
            return posts
        after = page["pageInfo"]["endCursor"]


def parsed(posts: list[Post]) -> list[tuple[Post, dict]]:
    pairs = []
    for post in posts:
        try:
            result = extract_result(post.body)
            validate_structure(result)
        except Rejected:
            continue
        pairs.append((post, result))
    return pairs


def ensure_labels() -> dict[str, str]:
    owner, name = owner_and_name()
    for label, (color, description) in LABELS.items():
        subprocess.run(["gh", "label", "create", label, "--repo", f"{owner}/{name}",
                        "--color", color, "--description", description],
                       capture_output=True, text=True)
    data = graphql("""query($owner: String!, $name: String!) {
      repository(owner: $owner, name: $name) { labels(first: 100, query: "benchmark") {
        nodes { id name } } } }""", owner=owner, name=name)
    return {node["name"]: node["id"] for node in data["repository"]["labels"]["nodes"]}


def apply_review(post: Post, review_: Review) -> None:
    label_ids = ensure_labels()
    stale = [label_ids[label] for label in post.labels & BOT_LABELS
             if label != review_.label and label in label_ids]
    if stale:
        graphql("""mutation($id: ID!, $labels: [ID!]!) {
          removeLabelsFromLabelable(input: {labelableId: $id, labelIds: $labels}) { clientMutationId } }""",
                id=post.discussion_id, labels=stale)
    if review_.label not in post.labels:
        graphql("""mutation($id: ID!, $labels: [ID!]!) {
          addLabelsToLabelable(input: {labelableId: $id, labelIds: $labels}) { clientMutationId } }""",
                id=post.discussion_id, labels=[label_ids[review_.label]])
    body = comment_body(review_)
    # Re-reviewing an edited post often reaches the same verdict. Editing the
    # comment to the text it already has failed the whole run, so leave it.
    if post.bot_comment_id and post.bot_comment_body.strip() == body.strip():
        return
    if post.bot_comment_id:
        try:
            graphql("""mutation($id: ID!, $body: String!) {
              updateDiscussionComment(input: {commentId: $id, body: $body}) { clientMutationId } }""",
                    id=post.bot_comment_id, body=body)
            return
        except RuntimeError as error:
            # The new verdict still has to reach the person.
            print(f"Could not update the review comment, adding one instead: {error}")
    graphql("""mutation($id: ID!, $body: String!) {
      addDiscussionComment(input: {discussionId: $id, body: $body}) { clientMutationId } }""",
            id=post.discussion_id, body=body)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    review_parser = commands.add_parser("review")
    review_parser.add_argument("--discussion", type=int, required=True)
    build_parser = commands.add_parser("build")
    build_parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    now = datetime.now(timezone.utc)
    posts = fetch_posts()
    if arguments.command == "review":
        post = next((p for p in posts if p.number == arguments.discussion), None)
        if post is None:
            print(f"#{arguments.discussion} is not in the benchmarks category; nothing to do.")
            return 0
        # Earlier posts decide duplicates (the first one wins); accepted
        # posts of any age are the baseline for outliers.
        others = [(p, r) for p, r in parsed(posts) if p.number != post.number
                  and (p.created_at <= post.created_at or accepted(p))]
        decision = review(post, others, now)
        apply_review(post, decision)
        print(f"#{post.number}: {decision.verdict}")
        for reason in decision.reasons:
            print(f"  - {reason}")
        return 0

    pairs = [(p, r) for p, r in parsed(posts)
             if LABEL_REJECTED not in p.labels and p.labels & {LABEL_COMMUNITY, LABEL_VERIFIED, LABEL_REVIEW}]
    dataset = build_dataset(pairs, now)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(dataset, separators=(",", ":")))
    print(f"{len(dataset['entries'])} results from {len(pairs)} posts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
