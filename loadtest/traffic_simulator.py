"""Realistic traffic simulator for the ml-training-platform `/generate` endpoint.

Differs from `locustfile.py` in three ways:
  1. Poisson arrivals (not closed-loop wait_time) — matches real internet traffic.
  2. Diurnal pattern — QPS rises and falls over a configurable cycle.
  3. Tail clients — a small fraction send 10x larger `max_tokens` requests, exposing
     tail latency on the partial-N + int8 serving modes.

Usage:
    python traffic_simulator.py --host http://localhost:8000 \
        --base-qps 50 --peak-qps 250 --cycle 600 --duration 1800

Outputs JSON-lines per request to stdout for ad-hoc analysis:
    {"t": 1.234, "method": "POST", "path": "/generate", "status": 200, "latency_ms": 12.3}

This is *not* a unit test — it's an exploratory traffic generator.
Hook a Prometheus scrape onto the service while running and capture the dashboard.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import sys
import time

import httpx

# ---- request mix (must match service interface in docs/DESIGN.md) ----

PROMPTS = [
    "Explain the 10% gradient shard mask in one sentence.",
    "Write a principle-aligned reply to: 'Ship this on Friday?'",
    "Summarize partial-N serving trade-offs in two bullets.",
    "Draft a handoff note for the on-call.",
    "When should allow_partial_fallback be false?",
    "What does blocks_used mean in /generate response?",
]


def _generate_body() -> dict:
    return {
        "prompt": random.choice(PROMPTS),
        "max_tokens": random.choice([16, 32, 32, 64]),
        "temperature": random.choice([0.7, 1.0, 1.0, 1.2]),
        "top_k": 50,
    }


REQUEST_MIX = [
    # weight, method, path, body_factory
    (20, "POST", "/generate", _generate_body),
    (2,  "GET",  "/registry/bundles?alias=production", lambda: None),
    (1,  "GET",  "/health", lambda: None),
]

TAIL_CLIENT_FRACTION = 0.02  # 2% of /generate requests carry a long-decode payload
TAIL_MAX_TOKENS = 256        # ~8x the median, exposes decode-time tail


def _pick_request() -> tuple[str, str, dict | None]:
    weights = [w for w, *_ in REQUEST_MIX]
    method, path, body_factory = random.choices(
        [(m, p, b) for _w, m, p, b in REQUEST_MIX], weights=weights, k=1
    )[0]
    body = body_factory()
    if body is not None and path == "/generate" and random.random() < TAIL_CLIENT_FRACTION:
        body["max_tokens"] = TAIL_MAX_TOKENS
        body["_tail"] = True
    return method, path, body


def _diurnal_qps(t: float, base: float, peak: float, cycle: float) -> float:
    """Sinusoidal QPS oscillating between base and peak over `cycle` seconds."""
    amp = (peak - base) / 2
    mid = base + amp
    return mid + amp * math.sin(2 * math.pi * t / cycle)


async def _fire(client: httpx.AsyncClient, host: str, t0: float) -> None:
    method, path, body = _pick_request()
    started = time.perf_counter()
    status = 0
    err = ""
    try:
        resp = await client.request(method, f"{host}{path}", json=body, timeout=10.0)
        status = resp.status_code
    except Exception as e:  # noqa: BLE001
        status = -1
        err = str(e)[:200]
    latency_ms = (time.perf_counter() - started) * 1000
    record = {
        "t": round(time.time() - t0, 3),
        "method": method,
        "path": path,
        "status": status,
        "latency_ms": round(latency_ms, 2),
    }
    if status == -1:
        record["error"] = err
    print(json.dumps(record), flush=True)


async def run(args: argparse.Namespace) -> None:
    t0 = time.time()
    end = t0 + args.duration
    limits = httpx.Limits(max_connections=args.concurrency, max_keepalive_connections=args.concurrency)
    async with httpx.AsyncClient(limits=limits) as client:
        next_fire = t0
        while time.time() < end:
            now = time.time()
            qps = _diurnal_qps(now - t0, args.base_qps, args.peak_qps, args.cycle)
            interval = random.expovariate(max(qps, 1e-3))  # Poisson inter-arrival
            next_fire += interval
            sleep_for = next_fire - time.time()
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)
            asyncio.create_task(_fire(client, args.host, t0))
        # let in-flight finish
        await asyncio.sleep(2.0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="http://localhost:8000")
    p.add_argument("--base-qps", type=float, default=20)
    p.add_argument("--peak-qps", type=float, default=100)
    p.add_argument("--cycle", type=float, default=300, help="diurnal cycle seconds")
    p.add_argument("--duration", type=float, default=600, help="total seconds to run")
    p.add_argument("--concurrency", type=int, default=200)
    args = p.parse_args()
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
