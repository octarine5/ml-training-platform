SHELL := /bin/bash
.DEFAULT_GOAL := help

SERVICE ?= ml-training-platform
IMAGE   ?= ml-training-platform:local
HOST    ?= http://localhost:8000
MODULE  ?= ml_training.serving.local_server:app

.PHONY: help install dev test up down logs build smoke load sim k8s-up k8s-down dash personalize serve clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## install editable package + dev deps
	pip install -e ".[dev]" || pip install -e .
	pip install locust httpx pytest prometheus-client opentelemetry-api opentelemetry-sdk uvicorn fastapi

dev: install ## run with hot reload
	uvicorn $(MODULE) --reload --host 0.0.0.0 --port 8000

test: ## unit tests
	pytest -q

build: ## docker build
	docker build -t $(IMAGE) -f deploy/Dockerfile .

up: ## docker compose up (service + prometheus + grafana)
	docker compose -f deploy/docker-compose.yml up -d --build

down: ## docker compose down
	docker compose -f deploy/docker-compose.yml down -v

logs: ## tail service logs
	docker compose -f deploy/docker-compose.yml logs -f service

smoke: ## hit /health to confirm the stack is alive
	@curl -fsS $(HOST)/health && echo " — healthy"

personalize: ## end-to-end personalize flow (mock dataset, smoke profile)
	ml-train personalize \
	  --config platform.yaml \
	  --base-preset local-default \
	  --principles principles.yaml \
	  --quant int8 \
	  --max-records 200 \
	  --mock-dataset

serve: ## boot LocalServer on the latest production-aliased bundle
	ml-train serve --bundle-alias production --mode full --host 0.0.0.0 --port 8000

load: ## quick stress test (200 users, 5 minutes) against /generate
	cd loadtest && locust -f locustfile.py --headless -u 200 -r 4 -t 5m \
	  --host $(HOST) --html ../stress_report.html --csv ../stress

sim: ## realistic traffic simulator (10 min, diurnal)
	python loadtest/traffic_simulator.py --host $(HOST) \
	  --base-qps 20 --peak-qps 120 --cycle 300 --duration 600 \
	  > traffic.jsonl

k8s-up: ## apply to current kubectl context
	kubectl apply -f deploy/k8s/deployment.yaml
	kubectl -n $(SERVICE) rollout status deployment/$(SERVICE)

k8s-down: ## delete from current kubectl context
	kubectl delete -f deploy/k8s/deployment.yaml --ignore-not-found

dash: ## open Grafana and Prometheus
	open http://localhost:3000 http://localhost:9090

clean: ## kill local artifacts
	rm -f stress_report.html stress_stats.csv stress_failures.csv stress_exceptions.csv traffic.jsonl
	docker compose -f deploy/docker-compose.yml down -v
