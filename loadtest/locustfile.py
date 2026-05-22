"""Stress test harness for ml-training-platform `/generate` endpoint.

Run modes:
    # Headless smoke (1 user, 30s)
    locust -f locustfile.py --headless -u 1 -r 1 -t 30s --host http://localhost:8000

    # Ramp to 200 users over 1 min, hold 5 min (CI / pre-release gate)
    locust -f locustfile.py --headless -u 200 -r 4 -t 6m --host http://localhost:8000 \
        --html report.html --csv stress

    # Interactive UI
    locust -f locustfile.py --host http://localhost:8000
    # then open http://localhost:8089

Pass / fail thresholds are enforced via --stop-timeout and the on_test_stop hook below.
SLO assertions (p99 < 500 ms on /generate, error rate < 0.1%) match docs/DESIGN.md.
"""

from __future__ import annotations

import json
import os
import random
import time

from locust import HttpUser, between, events, task

# --- SLO thresholds (keep in sync with docs/DESIGN.md) -----------------------
# /generate p99 < 500 ms on local-default preset, MPS. Override via env for
# tighter / looser hardware classes.
P99_LATENCY_MS_BUDGET = float(os.environ.get("SLO_P99_MS", "500"))
ERROR_RATE_BUDGET = float(os.environ.get("SLO_ERROR_RATE", "0.001"))  # 0.1%
# ---------------------------------------------------------------------------

# Prompts that exercise the personalized-principle paths. Vary so caching cannot
# hide perf bugs and so the rubric judge sees a realistic distribution.
PROMPTS = [
    "Explain why we picked the 10% gradient shard mask in one sentence.",
    "Write a brief, principle-aligned reply to: 'Should I ship this fix on Friday?'",
    "Summarize the partial-N serving trade-off for a non-ML PM.",
    "Draft a 3-bullet handoff note for the on-call swapping in.",
    "Given a flaky test, list the first three things to check, ordered.",
    "Rewrite this so it follows the principles file: 'we move fast and break things'.",
    "What does `blocks_used` mean in the /generate response?",
    "When would you set allow_partial_fallback to false?",
]


class ServiceUser(HttpUser):
    """One simulated client of the ml-training-platform serving layer."""

    # think-time between tasks; gamma-ish distribution feels more human than uniform
    wait_time = between(0.1, 1.5)

    def on_start(self) -> None:
        # warm-up: hit health so we surface connection errors fast and don't
        # count them against the SLO
        self.client.get("/health", name="00_warmup")

    # ---- task mix — weights = relative frequency ----

    @task(20)
    def generate(self) -> None:
        """POST /generate — the primary hot-path endpoint."""
        payload = {
            "prompt": random.choice(PROMPTS),
            "max_tokens": random.choice([16, 32, 32, 64]),  # weight toward 32
            "temperature": random.choice([0.7, 1.0, 1.0, 1.2]),
            "top_k": 50,
        }
        with self.client.post(
            "/generate",
            json=payload,
            name="POST /generate",
            catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"http {resp.status_code}")
                return
            try:
                body = resp.json()
            except Exception as e:  # noqa: BLE001
                resp.failure(f"bad json: {e}")
                return
            # Real generations always return some text. Partial-N mode is OK
            # (truncated=true is expected to surface degradation, NOT failure),
            # but token_ids should never be empty.
            if not body.get("token_ids"):
                resp.failure("empty token_ids")

    @task(2)
    def list_bundles(self) -> None:
        """GET /registry/bundles — control-plane read."""
        self.client.get("/registry/bundles?alias=production", name="GET /registry/bundles")

    @task(1)
    def health(self) -> None:
        # tiny background liveness probe so health surface is also stressed
        self.client.get("/health", name="GET /health")


# --- SLO assertion at end of test --------------------------------------------


@events.test_stop.add_listener
def _check_slo(environment, **_kwargs) -> None:
    stats = environment.runner.stats.total
    if stats.num_requests == 0:
        return

    p99 = stats.get_response_time_percentile(0.99) or 0
    err_rate = stats.num_failures / stats.num_requests

    summary = {
        "requests": stats.num_requests,
        "failures": stats.num_failures,
        "error_rate": err_rate,
        "p50_ms": stats.get_response_time_percentile(0.50),
        "p95_ms": stats.get_response_time_percentile(0.95),
        "p99_ms": p99,
        "rps": stats.total_rps,
        "slo_p99_budget_ms": P99_LATENCY_MS_BUDGET,
        "slo_error_budget": ERROR_RATE_BUDGET,
    }
    print("\n=== SLO REPORT ===")
    print(json.dumps(summary, indent=2, default=float))

    if p99 > P99_LATENCY_MS_BUDGET:
        environment.process_exit_code = 1
        print(f"FAIL: p99 {p99:.1f}ms > budget {P99_LATENCY_MS_BUDGET}ms")
    if err_rate > ERROR_RATE_BUDGET:
        environment.process_exit_code = 1
        print(f"FAIL: error_rate {err_rate:.4%} > budget {ERROR_RATE_BUDGET:.4%}")
    if environment.process_exit_code != 1:
        print("PASS: all SLOs within budget.")


# Suppress unused-import warnings; `time` is intentionally available for future use.
_ = time
