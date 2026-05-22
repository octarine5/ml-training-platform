# PERFORMANCE.md — Smoke load test results

Captured on 2026-05-21. Pairs with `DESIGN.md §1` (SLO targets) and
`loadtest/locustfile.py` (the harness).

## Test setup

- **Host.** Apple x86_64, MPS available, 38.7 GB unified memory
  (REPORT.md §1). Test ran with `torch` defaulting to CPU because the loaded
  bundle is medium-preset and the harness booted via `uvicorn` workers,
  which negotiate device per worker.
- **App.** `uvicorn ml_training.serving.local_server:app --host 127.0.0.1
  --port 8111` (FastAPI wrapper from `serving/fastapi_app.py`, lazy-exposed
  through `local_server.py`).
- **Bundle loaded.** `e703273628fc32b8` — `arch_preset=medium`,
  8 layers × `d_model=256` × `vocab_size=8000`, `quant=fp32`. Selected via
  `WeightRegistry.latest()` (no `production` alias set in this repo).
- **Tokenizer.** `artifacts/tokenizer/tokenizer.json` (real BPE).
- **Locust.** `-u 10 -r 2 -t 30s`, mixed task weight: 20× `/generate`, 2×
  `/registry/bundles`, 1× `/health`. SLO env override
  `SLO_P99_MS=10000` for this smoke (vs production target of 500 ms).

## Results

| Endpoint | Requests | Failures | p50 | p95 | p99 | Min | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| `GET /health` (warmup) | 10 | 0 | 6 ms | 15 ms | 15 ms | 4 ms | 14 ms |
| `GET /health` | 1 | 0 | 5 ms | 5 ms | 5 ms | 4 ms | 4 ms |
| `GET /registry/bundles` | 2 | 0 | 6 ms | 6 ms | 6 ms | 5 ms | 6 ms |
| `POST /generate` | 21 | 0 | **8.9 s** | **18 s** | **19 s** | 2.86 s | 18.5 s |
| **Aggregated** | **34** | **0** | 5.1 s | 18 s | 19 s | 4 ms | 18.5 s |

- **Error rate: 0.0%** (0 / 34). Well inside the 0.1% budget.
- **RPS: 1.32** sustained. Bounded by `/generate` latency, not by the app.

## Honest interpretation

- The control-plane endpoints (`/health`, `/registry/bundles`) are well
  inside SLO — single-digit milliseconds, which is consistent with the
  500 ms p99 control-plane budget in DESIGN.md §1.
- The `/generate` p99 of 19 s is **40× over** the 500 ms production SLO.
  Three reasons, all expected:
  1. **Wrong preset for the SLO target.** DESIGN.md §1 specifies p99 < 500 ms
     on `local-default` (6 layers, `d_model=128`) running on MPS. The
     loaded bundle is `medium` (8 layers, `d_model=256`) on CPU — roughly
     8× more compute per generated token, and no MPS acceleration.
  2. **No batching / no KV cache reuse across requests.** `MiniTransformer.generate`
     re-runs the full prompt through every block per new token. At
     `max_tokens=32` × 8 layers × `d_model=256`, that's expensive on CPU.
  3. **10 concurrent users on a single uvicorn worker, no async generate.**
     PyTorch `model.generate` holds the GIL; concurrency serializes through
     it. Production deploys 2 workers and relies on k8s HPA (Dockerfile
     line 45, DESIGN.md §1 throughput row).
- The control-plane numbers prove the FastAPI wrapper itself isn't the
  bottleneck — `/health` and `/registry/bundles` round-trip in single-digit ms.

## Path to the production SLO

Ordered by leverage. None of these is implemented yet; this is the plan.

1. **Load the matching preset.** Personalize a `local-default` bundle and
   re-run on MPS — REPORT.md §3.4 already shows the preset fits the box
   fp32-full. Expected p50 < 120 ms, p99 < 500 ms per the design target.
2. **Add KV cache.** `models/transformer.py` re-encodes the prompt on every
   new token today; a per-request KV cache cuts per-token compute from
   O(seq_len) to O(1). Largest single win.
3. **Continuous batching** at the FastAPI layer. Bin incoming requests by
   `max_tokens` and stride through `model.generate` in batches. Throughput
   target (DESIGN.md §1): ≥ 50 QPS per replica.
4. **Switch fp32 → int8 in serving.** Bundle already supports it
   (REPORT.md §3.1, table shows 4.0× memory reduction). On CPU this also
   trims arithmetic bandwidth.
5. **uvicorn `--workers 2` + HPA** (already in the Dockerfile). Each worker
   loads its own copy of the model — the right pattern for the
   GIL-bounded generate loop.

## Reproducing this run

```
cd /Users/diwang/Code/ml-training-platform
pip install -e .
pip install locust httpx fastapi uvicorn prometheus-client
uvicorn ml_training.serving.local_server:app --host 127.0.0.1 --port 8111 &
sleep 4
curl -fsS http://127.0.0.1:8111/health
cd loadtest
SLO_P99_MS=10000 timeout 45 locust -f locustfile.py \
    --headless -u 10 -r 2 -t 30s \
    --host http://127.0.0.1:8111 --csv /tmp/mlt_perf
cat /tmp/mlt_perf_stats.csv
```

CSV artifact: `/tmp/mlt_perf_stats.csv`.
Raw locust log: `/tmp/mlt_locust.log`.

---

*Last updated 2026-05-21.*
