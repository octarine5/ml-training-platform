# PRODUCT.md — Personalized ML Training Platform

Companion to `PROPOSAL.md` (positioning), `DESIGN.md` (architecture), and
`REPORT.md` (delivery & sizing). This doc names the **who** and the **what**:
who uses the platform, what jobs they hire it for, and which features serve
which job.

---

## 1. Personas

### P1 — ML Platform Engineer ("Platform")
Owns the factory itself. Cares about training-job throughput, hardware-pool
utilization, drift detection coverage, rollback time. Lives in
`control_plane/` and `monitoring/`.

- **Jobs to be done**
  - Keep training jobs from queueing more than 5 min on the shared pool.
  - Ship a new base preset (`local-default`, `medium`, `256L-base`) without
    breaking existing bundles. Bundle manifest carries `schema_version`
    (DESIGN.md §2.4).
  - Detect quality regression on production aliases nightly; auto-block
    promotion when judge recall drops below 0.85 (DESIGN.md §1, ADR-004).
  - Rollback an alias in < 30 s when a regression escapes
    (`WeightRegistry.promote`, DESIGN.md §1).
- **Pain today** — no single command takes "raw text + principles" all the way
  to a deployed bundle without bespoke glue. The `ml-train personalize`
  command and the registry alias model close that gap (REPORT.md §1).
- **Success metric** — *time-to-first-bundle* (cold repo → `/generate`
  returns 200) under 30 min; rollback under 30 s.

### P2 — Data Scientist / Researcher ("DS")
Trains or personalizes a model for a specific user, principle set, or
domain. Lives in `cli.py`, `training/`, `personalization/`, `evaluation_judge.py`.

- **Jobs to be done**
  - Personalize a base model against `principles.yaml` without retraining
    end-to-end. LoRA on Q,V (~25 k trainable params on `local-default`) plus
    the 10% gradient shard mask (REPORT.md §3.3, ADR-002/003).
  - Iterate on phased training plans without OOM on a single host
    (`PhasedTrainer.plan_phases`, ADR-001).
  - Evaluate candidate bundles via rubric judge + perplexity + drift
    (`evaluation_judge.ModelJudge`, REPORT.md §1 step 10).
  - Reproduce a teammate's run from bundle id (manifest pins arch_spec,
    base_hash, principles_hash, source URIs — REPORT.md §1 step 8).
- **Pain today** — full fine-tune is too slow and per-user artifacts too big.
  LoRA + per-tensor int8 cuts memory 4.0× on `256L-base` (3.95 GB → 0.99 GB)
  and disk to ~1 MB on `local-default` (REPORT.md §3.1, §3.2).
- **Success metric** — judge score on personalization corpus ≥ baseline; LoRA
  fine-tune wall-clock < 10 min on `local-default`.

### P3 — MLOps / SRE ("MLOps")
Runs the deployed serving fleet. Lives in `serving/`, `deploy/`, `monitoring/`.

- **Jobs to be done**
  - Hit `/generate` p99 < 500 ms on `local-default` / MPS and p50 < 120 ms
    (DESIGN.md §1 SLOs).
  - Keep `/generate` availability ≥ 99.9%. Cold-start guard via `/ready`
    readiness probe (DESIGN.md §5 row 6).
  - Survive bundle-doesn't-fit on a target box without refusing traffic —
    partial-N serving keeps tokens flowing with `truncated=true` flagged
    (ADR-004, REPORT.md §2).
  - Run canary on a subset of users before full alias swap (see SDLC.md §3).
- **Pain today** — without partial-N, you either upsize the box, ship an
  aggressively-quantized model with a quality cliff, or refuse the deploy.
  Partial-N gives an explicit, judge-visible degraded mode with a kill switch
  (`hardware.allow_partial_fallback: false`).
- **Success metric** — p99 ≤ 500 ms, error rate < 0.1%, rollback MTTR < 60 s.

---

## 2. Feature → persona map

| Feature | Module / artifact | P1 Platform | P2 DS | P3 MLOps |
|---|---|---|---|---|
| Phased training (block subsets, freezes others) | `training/phased.py` (ADR-001) | secondary | **primary** | — |
| 10% row-shard gradient mask | `build_layer_shard_mask` + `register_hook` (ADR-002) | — | **primary** | — |
| LoRA adapters on Q,V | `personalization/lora.py` (ADR-003) | — | **primary** | secondary |
| int8 / fp16 / fp32 quantization | `packaging/weights.py` + `HardwarePool.choose` | secondary | **primary** | **primary** |
| Per-element outlier-trimmed checkpoint averaging | `training/checkpoint_avg.py` | — | **primary** | — |
| Drift detection (rubric judge + recall) | `evaluation_judge.py` | **primary** | **primary** | secondary |
| Partial-N serving (truncated stack, `blocks_used`) | `serving/local_server.py` (ADR-004) | secondary | — | **primary** |
| Bundle registry + alias promote/rollback | `control_plane/registry.py` | **primary** | secondary | **primary** |
| Hardware pool detection + memory budget | `control_plane/hardware.py` | **primary** | secondary | secondary |
| `/generate` FastAPI surface + Prometheus | `serving/fastapi_app.py` + `monitoring/observability.py` | secondary | — | **primary** |
| `ml-train personalize` one-command pipeline | `cli.py` | secondary | **primary** | — |

---

## 3. User journeys

### Journey A — DS personalizes a model end-to-end

```
DS edits principles.yaml
   │
   ▼
ml-train personalize --base-preset local-default --principles principles.yaml --quant int8
   │
   │  (1) HardwarePool picks device + quant
   │  (2) FineWebLoader streams motionlabs/fineweb-ultra-mini
   │  (3) PhasedTrainer runs N phases, 10% row mask, writes phase_*.pt
   │  (4) average_state_dicts merges with |z|>2 outlier trim
   │  (5) attach_lora on Q,V (rank 8, alpha 16)
   │  (6) WeightPackager writes weights.safetensors.zst + manifest
   │  (7) WeightRegistry.register(bundle_id)
   ▼
ModelJudge scores rubric + perplexity → judge_score logged on the bundle
   │
   ▼
DS promotes to "staging" alias; Platform reviews drift report → "production"
```

Touch-points: `principles.yaml`, `platform.yaml`, `cli.py personalize`,
`artifacts/registry/<bundle_id>/`, `tests/data/judge_corpus.jsonl`.

### Journey B — MLOps cuts a new bundle to production

```
candidate bundle_id  ──►  POST /registry/promote {bundle_id, alias:"canary"}
   │
   │  k8s subset deployment shifts 5% of traffic to canary pods
   ▼
Prometheus watches /generate p99, error rate, judge_score for 1 h
   │
   ├── all green  ──► promote(bundle_id, "production")
   │
   └── any red   ──► promote(previous_bundle_id, "canary")  (< 30 s, alias swap)
```

### Journey C — Tight-budget box, model doesn't fit

```
HardwarePool detects 0.5 GB budget on target
   │
   ▼
hardware.allow_partial_fallback = true (default)
   │
   ▼
LocalServer(mode=partial, partial_blocks=16) loads first 16 of 256 blocks,
final LayerNorm + LM head still run on the truncated stack.
   │
   ▼
/generate responses carry truncated=true, blocks_used=16. Judge sees the
degradation; MLOps decides whether to grow the box or accept the drop.
```

---

## 4. Non-goals

- **Multi-tenant control plane with per-team budgets** — flagged as an open
  question in DESIGN.md §6. v1 is single-tenant.
- **Streaming `/generate` (SSE / chunked)** — v2 candidate. v1 returns the
  full completion in one response.
- **External LLM judge** — `ModelJudge.external_scorer` is a callable hook
  (REPORT.md §6 item 3). The default judge is the local rubric. v1 ships
  the hook; v2 wires a stronger scorer.
- **Distributed training across hosts** — `PipelineParallelism.partition`
  produces the split points the phased trainer would consume, but multi-host
  orchestration is out of scope for v1 (REPORT.md §6 item 4).

---

## 5. Pricing / cost story (v1)

From REPORT.md §3 and DESIGN.md §1:

| Lever | Saving |
|---|---|
| int8 weights vs fp32 | 4.0× memory on `256L-base` |
| safetensors + zstd | ~1 MB shipping size for personalized `local-default` |
| LoRA on Q,V vs full fine-tune | 88× fewer trainable params on `local-default` |
| Partial-N serving | Avoids upsizing hardware; degraded-but-live serving |
| 10% row mask | ~10× fewer effective updates / step at same wall-time |

Cost SLO: **$ / 1M /generate requests < $0.80** on `local-default` int8
(DESIGN.md §1).

---

*Last updated 2026-05-21. Owner: ml-platform@.*
