# ML Training Platform — Proposal

> One-pager that answers: who needs this, how big is the prize, what infra leverage does the design unlock.

## 1. Business use case

**Problem.** Teams that want to train and serve their own personalized language models today have to glue together a dataset loader, a tokenizer, a distributed trainer, a quantizer, a packaging format, a serving runtime, an evaluation harness, and a drift detector. Each of those is a project in itself; together they take 6–12 months and a dedicated ML platform team before a single personalized model reaches production. The ML Training Platform collapses that stack into one control-plane + data-plane system with a phased trainer, LoRA personalization, partial-N serving, and a built-in model judge — turning "I want a small model that follows my principles" from a quarter-long engagement into a single `ml-train personalize` command.

**Primary users / consumers.** Internal ML and product teams that need on-prem or single-GPU personalized models (privacy-sensitive workloads, edge deployment), external enterprise customers running regulated workloads who cannot ship data to Bedrock / Vertex, and platform teams that want a Vertex AI / SageMaker substitute they can self-host.

**Existing alternatives.** Google Vertex AI and AWS SageMaker (fully managed but lock you into a single cloud, no partial-N serving, no built-in shard-mask trainer); MosaicML / Databricks training stack (training only, no serving + personalization loop); HuggingFace TRL + vLLM + Optimum (OSS but unintegrated — every team rebuilds the glue). None of them give you the "train → average → package → serve with quality fallback" loop in one command, and none of them let you serve a model that doesn't fully fit on the box via partial-N degradation rather than refusing the deploy.

**Success criteria (12-month).** (1) 80% of internal personalized-model traffic served via this platform. (2) Median time from "I have principles.yaml" to "personalized model serving traffic" cut from 14 days to 2 days. (3) Training cost per converged personalized model cut 65% vs. the full-dense baseline via the 10% shard mask + LoRA. (4) Serving cost per million `/generate` requests cut 45% vs. the full-precision baseline via partial-N + int8.

## 2. Opportunity sizing

| Layer | Definition | Value | Source / Assumption |
|---|---|---|---|
| **TAM** | Total addressable market for ML platforms (training + serving) | $30B by 2028 | IDC / Gartner ML platforms category (managed + self-hosted), 24% CAGR off 2024 ~$12B base |
| **SAM** | Enterprise ML platform segment we can serve (self-hosted + hybrid, English-language LLM workloads, <100B params) | $10B | ~33% of TAM — excludes hyperscaler-exclusive workloads, excludes pure inference-only and pure annotation segments |
| **SOM** | Realistically capturable in 3 years | $100M | 1% of SAM — conservative against Vertex / SageMaker incumbents; based on 30-eng platform org with mid-market GTM |
| **Internal value** | Eng-hours saved + cloud-cost avoided vs. building this stack in-house per team | $8M/yr | Saves ~16 platform FTE @ $250k each across 4 internal product orgs that would otherwise each build a partial copy |

**Demand signal.** Internal personalization backlog has 12 product teams waiting for a single-GPU LoRA + serving path that today routes through ad-hoc notebooks. External: every customer conversation around "can we run this on-prem" today ends because we point at SageMaker; closing that gap unlocks the regulated-workload pipeline (~$30M ARR in qualified opportunities).

## 3. Infra performance boost (quantified)

Baseline = a "naïve" pipeline: dense full-parameter training, fp32 full-model serving, no packaging compression, no degradation fallback. Target = the design in `REPORT.md`.

| Metric | Baseline | Target (this service) | Δ | How |
|---|---|---|---|---|
| Training cost per converged personalized model | $1.00 (index) | $0.35 | -65% | 10% gradient row mask during pretraining (only ~10% of weight rows update per step) + LoRA personalization (~25k trainable params vs. 2.2M base on `local-default`, 88× fewer) |
| Serving cost per 1M `/generate` requests | $1.00 (index) | $0.55 | -45% | Partial-N serving (N of M blocks live) + int8 quantization (4.0× memory reduction on `256L-base`, 3.95 GB → 0.99 GB) + safetensors+zstd packaging (6.3× shipping-size cut at int8) |
| Model iteration cycle (principles edit → serving) | 2 weeks | 2 days | -86% (7×) | One `ml-train personalize` command runs hardware-detect → stream dataset → BPE → phased train → average → LoRA → package → serve, replacing the 9-step manual stitch |
| `/generate` p99 latency (`local-default`, MPS) | ~1.2 s | < 500 ms | -58% | Partial-N short-circuits unused blocks; int8 cuts memory pressure on KV cache; resolved device (CUDA / MPS / CPU) picked at load time |
| Training job submission p99 | n/a (manual) | < 500 ms | new SLO | Control plane (`PlatformConfig` + `WeightRegistry` + `HardwarePool`) accepts submissions sync and dispatches async |
| Storage per model bundle (`local-default`, int8, real-trained entropy) | ~12.5 MB (fp32 raw) | ~1 MB | -92% | int8 weights + zstd level 10 + manifest-only metadata |
| Model rollback time | minutes (redeploy) | < 30 s | new SLO | `registry.promote(bundle_id, alias)` alias swap — server holds bundles in memory, alias flip is constant-time |

**Why these numbers are achievable.**
- The -65% training-cost number is the product of the 10% row mask (~10× fewer effective updates per step at the same wall-time, per `REPORT.md §3.3`) and LoRA on the personalization stage (88× fewer trainable params on `local-default`).
- The -45% serving-cost number is the product of int8 (4.0× memory reduction on `256L-base`, measured in `REPORT.md §3.1`) and partial-N (drops blocks N+1..M from compute) under a mixed-mode serving fleet.
- The 2wk → 2d cycle is what the one-command flow already does on a smoke run (`REPORT.md §1`); the gap is hardening + multi-tenant control plane, not new ML.

## 4. Risk & non-goals

- **Non-goals.** No multi-region failover in v1 (single-cluster control plane). No streaming `/generate` (full response only — streaming is a v2 follow-up that needs server-side batching). No models > 256 layers in v1 (`256L-base` is the largest profilable preset; bigger needs real multi-host pipeline parallelism, currently profile-only). No RLHF — personalization is principle-driven LoRA, not preference modeling.
- **Top risks.** (1) Partial-N quality cliff: dropping below N=2 on a 6-block model degrades judge score sharply; mitigated by drift detector + `allow_partial_fallback: false` lever. (2) HuggingFace `datasets` streaming dependency (fineweb-ultra-mini) — outage breaks fresh training; mitigated by `--mock-dataset` flag and tokenizer cache. (3) int8 quantization recall drop on principles compliance; mitigated by judge-gated promotion in the registry.
- **Migration cost.** Internal teams currently on notebooks: ~1 week per team to port to the personalize CLI. External: SageMaker / Vertex parity for the top 10 features takes ~2 quarters.

## 5. Decision asks

- [ ] Headcount: 6 eng for 9 months (3 platform, 2 ML, 1 SRE).
- [ ] Budget: $2.4M for infra (GPU pool for the control-plane training fleet + 3× clusters for staging / prod / DR).
- [ ] Cross-team dependency from: Data Platform (fineweb-style dataset mirroring), Security (on-prem signing for `weights.safetensors.zst` bundles), Billing (per-bundle and per-1M-request meters).

---
*Last updated: 2026-05-20. Owner: ml-platform@.*
