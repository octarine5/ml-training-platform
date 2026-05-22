# SDLC.md — Software Delivery Lifecycle

How code, models, and bundles move from a developer's laptop to production
serving on the ML Training Platform. Pairs with `PRODUCT.md` (who),
`DESIGN.md` (architecture), `REPORT.md` (delivery decisions).

---

## 1. Branching model

Trunk-based, short-lived feature branches.

```
main                  ●────●────●────●────●────●────►  always deployable
                       │         │         │
feat/lora-rank          ────●────             ──►  squash-merge after CI green
feat/partial-streaming        ────●────●────●────►  squash-merge
hotfix/p99-regression                          ●─►   fast-path; canary skipped only with explicit override
```

Rules:
- Direct push to `main` blocked. PRs only.
- Feature branches: `feat/<scope>`, `fix/<scope>`, `hotfix/<scope>`,
  `chore/<scope>`. Squash merge keeps `main` linear.
- Branches live ≤ 3 days. Anything older rebases on `main` before merge.
- Bundle promotions are *not* branches — they happen through
  `control_plane.registry.WeightRegistry.promote` (alias swap, < 30 s).

Tags + releases:
- Code: SemVer on `main`, e.g. `v0.2.0` (matches `pyproject.toml`).
- Bundles: opaque content-hash `bundle_id` (sha256 of compressed weights,
  truncated) — never versioned alongside the code.

---

## 2. CI/CD

### 2.1 Pipeline stages

```
[1] lint        ── ruff, black --check, mypy (--strict on src/ml_training/control_plane)
[2] unit        ── pytest tests/ (168 tests, ~14 s; see REPORT.md §4)
[3] arch tests  ── tests/test_architecture.py, test_distribution.py, test_phased_training.py
[4] build       ── pyproject.toml setuptools build → wheel; deploy/Dockerfile build
[5] smoke       ── ml-train personalize --mock-dataset --max-records 50 (5 min budget)
[6] offline eval gate  ── ModelJudge on tests/data/judge_corpus.jsonl
                          + ModelDriftDetector against last good bundle
[7] load        ── loadtest/locustfile.py headless, -u 50 -t 2m, fail on p99 > 500 ms or err > 0.1%
[8] publish     ── push image to registry; ml-train register-bundle
[9] canary      ── promote(bundle_id, "canary"); 5% of traffic for 60 min
[10] promote    ── promote(bundle_id, "production") iff canary green
```

Stages [1]-[4] run on every PR. Stages [5]-[7] run on PRs that touch
`src/ml_training/`, `deploy/`, or `monitoring/`. Stages [8]-[10] run on `main`
merge.

### 2.2 The offline eval gate (stage [6])

Hard gate before any bundle reaches a real alias. Lives in
`evaluation_judge.py`.

Pass criteria:
- **Judge score ≥ 0.85 × baseline** (baseline = `production` alias judge
  score from last green run).
- **Perplexity within +5%** of baseline on the held-out batch stream.
- **Drift detector recall ≥ 0.9** on the regression corpus (DESIGN.md §1).
- **No new safetensors tensor names** unless `manifest.schema_version` bumps
  (back-compat policy, DESIGN.md §2.4).

A failed eval gate aborts the pipeline; the bundle stays registered
(immutable) but is not promoted to any alias. DS gets the rubric report in
the PR comment.

### 2.3 Canary on subset of users (stage [9])

Two implementations supported:

**(a) Alias-based traffic split (default).** k8s `Service` selector matches
both `production` and `canary` pods; HPA ratio is 95:5. Bundle resolution at
pod startup reads its alias from env. To shift: `kubectl scale` the canary
deployment. Roll forward = `promote(bundle_id, "production")` (alias swap,
constant time). Roll back = `promote(previous_bundle_id, "canary")`.

**(b) Header-pinned cohort.** Client sends `X-Cohort: canary` for 5% of
users (chosen by hash of user_id). The FastAPI app reads the header and
swaps to the canary `LocalServer` instance held in-process. Used when we
want the cohort to be deterministic across replicas, not random.

Canary gates (auto-rollback triggers):
- `/generate` p99 > 1.5× baseline for 10 min.
- `/generate` 5xx rate > 0.5% over 5 min.
- `judge_score` on shadow eval > 5% regression.
- Any pod restart loop (> 3 restarts in 10 min).

---

## 3. Environments

| Env | Purpose | Aliases live | Hardware budget | Data |
|---|---|---|---|---|
| `dev` | Per-engineer laptop | none (direct bundle dir) | MPS / CPU | `--mock-dataset` |
| `ci` | Pipeline-only | none | CPU runner | `--mock-dataset --max-records 200` |
| `staging` | Shadow of prod | `staging` | 1× small GPU | fineweb-ultra-mini streaming |
| `prod` | Live traffic | `canary`, `production`, `previous` | full pool | fineweb-ultra-mini streaming |

The `previous` alias is always set to whatever `production` was before the
last promote — this is the constant-time rollback target.

---

## 4. Runbook

### 4.1 Drift detection alert

**Symptom.** Pager fires on `drift_detector_recall < 0.85 for 2 nights`
(DESIGN.md §1, §5).

1. Pull the latest drift report: `ml-train evaluate --checkpoint
   $(ml-train registry resolve production) --corpus
   tests/data/judge_corpus.jsonl`.
2. Compare against the `previous` alias's last-known-good report. If the
   regression localizes to a specific rubric dimension (e.g. "principle
   alignment"), it's a personalization regression — proceed to step 3. If
   it's broad-spectrum, it's likely a base-weight or tokenizer regression —
   skip to step 5.
3. **Rollback first, investigate after**: `WeightRegistry.promote(
   previous_bundle_id, "production")`. The alias swap is < 30 s. Confirm
   `/generate` p99 normalizes within 5 min.
4. Open an incident, attach the drift report. Re-personalize against the
   current `principles.yaml`; the offending bundle stays registered but
   alias-less.
5. If base-weight or tokenizer regression: lock the `tokenizer.json` hash
   in `principles.yaml` source URIs (REPORT.md §1 step 8 — manifest pins
   source hashes) and pin to the last green base preset.

### 4.2 Rollback to last good model

```
# one-liner — alias swap, constant time
ml-train registry promote --bundle-id $(cat artifacts/registry/aliases.json | jq -r .previous) --alias production
```

Or via API:
```
POST /registry/promote {"bundle_id": "<prev>", "alias": "production"}
```

Bundles are immutable; rollback never re-trains, it only swaps the alias
pointer.

After rollback:
- Set `canary` alias to the same bundle as `production` (prevents stale
  canary from auto-promoting).
- Capture the failing bundle id in the incident ticket — do **not** delete.
- Run `ml-train evaluate` against the failing bundle on the broader
  corpus to characterize the regression before closing the incident.

### 4.3 Training job stuck

**Symptom.** `phased_trainer_duration_seconds` > 2× expected, or a phase has
made no checkpoint progress in 15 min.

1. Check the phase that's stuck: `ls -lt artifacts/phase_checkpoints/` — the
   newest file's `mtime` tells you which phase is alive.
2. Inspect device + memory: `ml-train profile-transformer --preset
   <preset>` shows expected footprint vs detected budget
   (`HardwarePool.choose`).
3. If OOM: edit `platform.yaml` to reduce phase block-span (fewer blocks
   trainable per phase) or enable CPU offload. The phased trainer accepts a
   shorter `plan_phases` output without code change.
4. If hang (no OOM, no progress): SIGTERM the trainer. Phase checkpoints
   written so far survive — `average_state_dicts` will merge whatever's on
   disk, just with fewer phases contributing. Document the partial run in
   the bundle manifest's `created_at`.
5. If the dataset stream is the culprit (`data_freshness_seconds` > 60 s
   alert, DESIGN.md §5): swap to `--mock-dataset` to unblock, then come
   back when fineweb-ultra-mini is healthy.

### 4.4 `/generate` p99 regression

1. `kubectl top pods` — is a single replica hot? If yes, restart it.
2. Check Prometheus for `truncated=true` rate. A spike means partial-N is
   active where it shouldn't be — verify `hardware.allow_partial_fallback`
   and bundle vs target-box fit (REPORT.md §3.4 table).
3. If broad regression: rollback per §4.2.

### 4.5 Cold-start blocking `/ready`

Expected for 20-30 s after pod start (model load is heavy). Confirm:
- `readinessProbe.initialDelaySeconds ≥ 30` (DESIGN.md §5).
- `/health` returns 200 within 2 s (uvicorn up, app object built).
- `/ready` returns 503 with `{"deps": {"model": false}}` during load.

If `/ready` stays 503 past 60 s, check the bundle's `compressed_size_bytes`
— very large bundles need a higher `initialDelaySeconds`. Edit the k8s
manifest; do not lower `compressed_size_bytes` by re-quantizing without
re-running the offline eval gate.

---

## 5. Change management

- **Pre-commit hooks** (suggested): ruff, black, mypy on touched files,
  `pytest tests/test_architecture.py tests/test_packaging.py` (the two
  fast-feedback tests).
- **PR checklist** (template-enforced):
  - [ ] Touched a `serving/`, `training/`, or `packaging/` module? Re-ran
        `ml-train personalize --mock-dataset --max-records 50` locally.
  - [ ] Touched a manifest field? Bumped `manifest.schema_version`?
  - [ ] Touched an SLO target? Updated `DESIGN.md §1` and the locust
        `P99_LATENCY_MS_BUDGET`/`ERROR_RATE_BUDGET` env defaults.
  - [ ] Added a new dependency? Pinned in `pyproject.toml`, justified in PR
        description.
- **Code owners** — `serving/` + `deploy/` → MLOps; `training/` +
  `personalization/` + `evaluation_judge.py` → DS leads; `control_plane/`
  + `monitoring/` → Platform.

---

*Last updated 2026-05-21. Owner: ml-platform@.*
