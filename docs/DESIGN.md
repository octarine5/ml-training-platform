# ML Training Platform — Design Doc

Companion to `PROPOSAL.md`. Covers the four engineering anchors:
**(1) Service KPIs** · **(2) Interface** · **(3) Modules** · **(4) Key design decisions**.

## 1. Service KPIs (SLOs / SLIs)

| Tier | Metric | Target | Measurement | Error budget | Alert at |
|---|---|---|---|---|---|
| Latency (control) | training job submission p99 | < 500 ms | `http_request_duration_seconds{handler="/jobs",quantile="0.99"}` | 0.1% requests | p99 > 500 ms for 10 min |
| Latency (serving) | `/generate` p99 (`local-default`, MPS) | < 500 ms | `http_request_duration_seconds{handler="/generate",quantile="0.99"}` | 0.1% requests | p99 > 500 ms for 10 min |
| Latency (serving) | `/generate` p50 | < 120 ms | `http_request_duration_seconds{handler="/generate",quantile="0.5"}` | — | p50 > 2× target for 5 min |
| Throughput (serving) | sustained QPS per replica | ≥ 50 | `rate(http_requests_total{handler="/generate"}[1m])` | — | < 80% of target for 10 min |
| Training | end-to-end time on GPT-2-small-equivalent (124M params) | ≤ 4 h on 8×A100 | `phased_trainer_duration_seconds` summed across phases | — | > 5 h on identical config |
| Quality | drift detector recall on regression corpus | ≥ 0.9 | `drift_detector_recall` computed nightly via `evaluation_judge.ModelDriftDetector` over a labeled corpus | — | < 0.85 for 2 nights |
| Reliability | model rollback (alias swap) | < 30 s | `registry_promote_duration_seconds` | — | > 60 s |
| Availability | success rate on `/generate` | ≥ 99.9% | `1 - rate(http_requests_total{handler="/generate",status=~"5.."}[5m]) / rate(http_requests_total{handler="/generate"}[5m])` | 43 min/month | < 99.5% for 5 min |
| Cost | $ / 1M `/generate` requests (`local-default`, int8) | < $0.80 | (compute$ + storage$) / total requests | — | > 1.5× target for 1 day |
| Saturation | GPU utilization (serving) | < 80% sustained | `nvidia_gpu_utilization` (or process_cpu on MPS) | — | > 90% for 10 min |
| Freshness | dataset stream lag (fineweb-ultra-mini) | < 60 s | `data_freshness_seconds` from `data_sources.fineweb.FineWebLoader` | — | > 2× target |

**SLO derivation.** The `/generate` p99 < 500 ms target is the user-perceptible chat-turn threshold on the `local-default` preset (6 layers, d_model 128) running on MPS — measured per `REPORT.md §1` smoke runs, headroom set against the partial-N degradation envelope. The training-submission < 500 ms target is the standard "interactive control-plane" threshold (sync submit, async dispatch). The 4 h on 8×A100 for GPT-2-small is the standard public benchmark (124M params, OpenWebText scale equivalent) and matches the throughput class the phased trainer plus 10% shard mask hits in the design. Drift recall ≥ 0.9 is the threshold below which the judge starts admitting visibly regressed bundles — calibrated against the rubric in `evaluation_judge.ModelJudge`. Rollback < 30 s is constant-time: an in-memory alias swap in `control_plane.registry.WeightRegistry.promote`.

## 2. Interface

### 2.1 REST API (OpenAPI summary)

| Method | Path | Purpose | Request | Response | Errors |
|---|---|---|---|---|---|
| GET  | `/health`  | Liveness | — | `{"status":"ok"}` | 503 |
| GET  | `/ready`   | Readiness (model loaded, tokenizer attached) | — | `{"ready":true,"deps":{"model":true,"tokenizer":true}}` | 503 |
| GET  | `/metrics` | Prometheus exposition | — | text/plain | — |
| POST | `/generate` | Generate completion from prompt | `GenerationRequest = {prompt: str, max_tokens: int=32, temperature: float=1.0, top_k: int=50}` | `GenerationResponse = {text: str, token_ids: int[], truncated: bool, blocks_used: int, quant: str, extra: {prompt_token_count, generated_token_count, bundle_id}}` | 400 invalid JSON, 422 validation, 500 model error |
| POST | `/jobs/train` | Submit training job | `{config_path: str, base_preset: str, num_phases: int}` | `{job_id: str, status: "queued"}` | 422 invalid config, 503 hardware pool full |
| POST | `/jobs/personalize` | Submit personalization job | `{base_preset: str, principles_file: str, quant: "fp32"|"fp16"|"int8", max_records: int}` | `{job_id: str, status: "queued"}` | 422 validation, 503 hardware pool full |
| GET  | `/registry/bundles` | List bundles | `?alias=production` | `[{bundle_id, arch, quant, base_hash, created_at}]` | 5xx |
| POST | `/registry/promote` | Promote a bundle to an alias (rollback / cutover) | `{bundle_id: str, alias: str}` | `{ok: true, previous: str|null}` | 404 unknown bundle |

The primary on-the-hot-path endpoint is `POST /generate`, implemented in `src/ml_training/serving/local_server.py` (`LocalServer.serve_http` and the dataclass-typed `GenerationRequest` / `GenerationResponse`). Today this uses `http.server` directly; the FastAPI app factory in `app.py` wraps it for the deploy target so `/metrics`, `/health`, `/ready` come for free.

Full schema: served by FastAPI at `GET /openapi.json` and `GET /docs` (Swagger UI).

### 2.2 CLI interface

| Command | Purpose | Key flags |
|---|---|---|
| `ml-train train` | Run phased training from a config | `--config platform.yaml`, `--base-preset`, `--num-phases`, `--steps-per-phase` |
| `ml-train personalize` | Full pipeline: train → average → LoRA → package → register | `--config platform.yaml`, `--base-preset local-default`, `--principles principles.yaml`, `--quant int8`, `--max-records 200`, `--mock-dataset` |
| `ml-train evaluate` | Run `evaluation_judge.ModelJudge` over a corpus | `--checkpoint model.ckpt`, `--corpus tests/data/judge_corpus.jsonl` |
| `ml-train profile-transformer` | Profile a preset without training | `--preset 256L-base` |
| `ml-train serve` | Boot `LocalServer` on `/generate` | `--bundle-id`, `--mode full|partial|int8`, `--partial-blocks N`, `--host`, `--port` |

### 2.3 Event interface

| Topic / Stream | Direction | Schema | Partition key | Retention |
|---|---|---|---|---|
| `ml-platform.jobs.training` | out | `{job_id, phase, step, loss, t_seconds}` | `job_id` | 7 days |
| `ml-platform.bundles.promoted` | out | `{bundle_id, alias, previous_bundle_id, ts}` | `alias` | 30 days |
| `ml-platform.drift.alerts` | out | `{alias, judge_score, baseline, recall, ts}` | `alias` | 30 days |

### 2.4 Backwards-compatibility policy

- Additive only within a major version (new optional fields on `GenerationResponse.extra` OK, no field removals).
- Deprecation window: 1 release / 90 days.
- Breaking `/generate` changes go via `/v2/generate`.
- Bundle manifest schema versioned via `manifest.schema_version`; old bundles continue to load read-only.

## 3. Module map

```
src/ml_training/
├── app.py                              # FastAPI app factory wrapping LocalServer + control plane
├── cli.py                              # ml-train CLI (train, personalize, evaluate, profile, serve)
├── architecture.py                     # TransformerSpec + presets (local-default, 256L-base)
├── tokenization.py                     # Byte-level BPE
├── data_pipeline.py                    # batching, masking
├── data_sources/
│   └── fineweb.py                      # HF datasets streaming, mock mode
├── models/
│   └── transformer.py                  # MiniTransformer (real Q/K/V attention)
├── training/
│   ├── phased.py                       # PhasedTrainer + 10% row-shard gradient mask
│   └── checkpoint_avg.py               # outlier-trimmed averaging (|z|>2)
├── packaging/
│   └── weights.py                      # safetensors + zstd, fp32 / fp16 / int8
├── personalization/
│   ├── principles_to_dataset.py        # principles.yaml → contrastive pairs
│   └── lora.py                         # LoRA adapters on Q,V
├── serving/
│   └── local_server.py                 # LocalServer + POST /generate (http.server)
├── control_plane/
│   ├── config.py                       # PlatformConfig loader
│   ├── registry.py                     # WeightRegistry (bundles + alias-based promote)
│   ├── sources.py                      # SourceLinks (dataset / tokenizer / base weights)
│   └── hardware.py                     # HardwarePool (CUDA / MPS / CPU detection + budget)
├── evaluation_judge.py                 # ModelJudge (rubric + perplexity) + drift detector
└── evaluation.py / fine_tuning.py / orchestrator.py / features.py / distribution.py  # legacy reused
```

| Module | Responsibility | Stable contract | Notes |
|---|---|---|---|
| `serving/local_server.py` | HTTP serialization of `/generate`, model load, partial-N + int8 modes | `GenerationRequest` / `GenerationResponse` dataclass shape | Thin handler; calls `MiniTransformer.generate` directly |
| `training/phased.py` | Phased trainer with deterministic 10% row mask via `register_hook` | `PhasedTrainer.plan_phases() / train()` signatures | The mask is the load-bearing efficiency move |
| `training/checkpoint_avg.py` | Per-element outlier-trimmed averaging across phase checkpoints | `average_state_dicts(paths, z_threshold=2.0)` | Outputs averaged `state_dict` ready for packaging |
| `packaging/weights.py` | safetensors + zstd write/read, real per-tensor int8 symmetric quant | `WeightPackager.pack(state_dict, quant)` → bundle dir | Manifest carries `quant_scales` |
| `personalization/lora.py` | LoRA wrapping of Q,V on every block | `attach_lora(model, rank, alpha)` | Only A/B + final LN + token embedding trainable |
| `control_plane/registry.py` | Bundle registry + alias promote (rollback path) | `register(bundle_id) / promote(bundle_id, alias) / resolve(alias)` | Promote is constant-time, < 30 s SLO |
| `control_plane/hardware.py` | Device detection + memory budget + quant suggestion | `HardwarePool.choose()` | Drives the "auto" quant pick at packaging time |
| `evaluation_judge.py` | Rubric judge + perplexity + drift detection | `ModelJudge.score(prompts) / ModelDriftDetector.check(metrics)` | Drift recall is the quality SLO source |
| `cli.py` | One-command entry (`personalize`) and per-stage entries | `ml-train <subcommand>` | Wraps every module above |
| `observability.py` | Prometheus + OTel instrumentation | metric names + labels | Imported by `app.py` |

Dependency direction: `cli / app → control_plane + serving + training → packaging + models + tokenization → data_sources`. No back-edges.

## 4. Key design decisions (ADR-lite)

### ADR-001 — Phased training (architecture → models → train → serve)

- **Context.** A 256-layer transformer cannot be trained dense on a single-host budget, and a 6-layer local model still wants the same code path. We need a training procedure that scales between them without two trainers. Per `REPORT.md §1` and §5.
- **Options considered.**
  1. Single dense trainer over all layers — doesn't scale, OOMs on 256L-base.
  2. Pipeline-parallelism only — solves memory but needs multi-host, can't run locally.
  3. Phased trainer that freezes all-but-current-phase blocks and rotates through phases on one host, with `PipelineParallelism.partition` choosing the phase boundaries.
- **Decision.** Phased trainer (option 3). `architecture` defines blocks; `training/phased.py` walks contiguous block subsets, freezes the rest, trains, checkpoints; `training/checkpoint_avg.py` merges per-phase checkpoints with `|z|>2` outlier trimming; `packaging/weights.py` ships the merged state.
- **Trade-off accepted.** Phases don't see global gradients, so convergence is slower per-step than a true dense run; we pay extra steps for the ability to fit on one host.
- **Reversal cost.** Low — drop in a dense trainer alongside; phases are a policy on top of the same `MiniTransformer`. Reversed only if multi-host becomes the only target.

### ADR-002 — 10% shard mask during pretraining

- **Context.** Even within a phase, updating every parameter every step is wasteful for the personalization use case (small effective rank of change). Per `REPORT.md §3.3`.
- **Options considered.**
  1. Full dense gradients per step.
  2. Structured sparsity (e.g. block-sparse) — needs custom kernels.
  3. Deterministic 10% row mask: `build_layer_shard_mask(...)` + per-parameter `register_hook` that zeros gradients outside a fixed 10% row set per step.
- **Decision.** Option 3 — the 10% row mask. Forward + backward still run in full (cheap on small models), only the gradient update is sparse. Optimizer state stays full so resumability is preserved.
- **Trade-off accepted.** ~10× fewer effective updates per step at the same wall-time; recovered by running more steps or by LoRA on top.
- **Reversal cost.** Very low — `shard_fraction: 1.0` disables the mask.

### ADR-003 — LoRA + quantization for personalization

- **Context.** Personalization should not retrain the whole model. Per `REPORT.md §1` step 7 and §3.1.
- **Options considered.**
  1. Full fine-tune — slow, large per-user artifacts.
  2. Prompt-tuning — too weak for principle-driven behavior changes.
  3. LoRA adapters on Q,V (rank 8, alpha 16) + symmetric per-tensor int8 quantization at packaging.
- **Decision.** Option 3. Trainable params drop from 2.2M base to ~25k on `local-default` (88× fewer); int8 cuts memory 4.0× on `256L-base` (3.95 GB → 0.99 GB).
- **Trade-off accepted.** int8 induces small quality drop; judge-gated promotion in the registry blocks regressions.
- **Reversal cost.** Low — `merge_at_packaging: false` in `platform.yaml` keeps adapters separate; `quant: fp32` skips quantization.

### ADR-004 — Partial-N serving (only N model variants / blocks live)

- **Context.** A model bundle may not fit on the target box, or we want a degraded-but-live mode for cheap replicas. Per `REPORT.md §2` and §3.4.
- **Options considered.**
  1. Refuse the deploy when the full model doesn't fit (status quo).
  2. Always quantize harder (int4) — quality cliff worse than partial-N at equivalent footprint.
  3. Partial-N: serve only the first N transformer blocks; final LayerNorm + LM head still run on the truncated stack; response carries `truncated=true` and `blocks_used=N` for the judge to see.
- **Decision.** Partial-N (option 3). Lever: `hardware.allow_partial_fallback: true | false` in `platform.yaml`.
- **Trade-off accepted.** Quality degrades measurably (judge rubric score drops); never refuses to return tokens. Caller opt-out exists.
- **Reversal cost.** Very low — set `mode=full` and drop the partial code path; it's an `Enum`-switched branch in `LocalServer`.

## 5. Failure modes & mitigations

| Failure | Detection | Mitigation | Owner |
|---|---|---|---|
| HuggingFace `datasets` streaming outage (fineweb-ultra-mini) | `data_freshness_seconds` alert | `--mock-dataset` flag + tokenizer cache at `artifacts/tokenizer/` | data |
| Phase OOM on `256L-base` | `phased_trainer_duration_seconds` + restart-count | Reduce phase block-span; spill to CPU offload; fall back to LoRA-only on the smallest preset | platform |
| Partial-N quality cliff (judge score < threshold) | `drift_detector` alert | Block alias promote in `WeightRegistry.promote`; require manual override | ML |
| int8 quant numerical instability | Drift recall < 0.85 for 2 nights | Auto-fallback to fp16 at packaging via `HardwarePool` chooser | ML |
| `/generate` p99 regression | Prometheus `histogram_quantile(0.99, …)` alert | Roll back alias via `registry.promote(previous, alias)` (< 30 s) | service |
| Server cold-start (model load) blocking `/ready` | `/ready` returns false until model + tokenizer attached | k8s `readinessProbe` with `initialDelaySeconds: 30` lets HPA route around cold pods | service |

## 6. Open questions

- Should bundle promotion be sync (block on judge eval) or async (eval after promote, auto-rollback)? Trade-off is rollout speed vs. exposure time.
- Multi-tenant control plane: per-team budgets + GPU pool quotas — needed for v1 or v2?
- Streaming `/generate` (SSE / chunked) — punt to v2 or fold into v1 alongside continuous batching?

---
*Last updated: 2026-05-20. Owner: ml-platform@.*
