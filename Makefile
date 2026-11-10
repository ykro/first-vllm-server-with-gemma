PYTHON ?= $(shell if [ -x ../../.venv-shared/bin/python ]; then echo ../../.venv-shared/bin/python; else command -v python3; fi)
PYTEST ?= $(shell if [ -x ../../.venv-shared/bin/pytest ]; then echo ../../.venv-shared/bin/pytest; else command -v pytest; fi)
RUFF ?= $(shell if [ -x ../../.venv-shared/bin/ruff ]; then echo ../../.venv-shared/bin/ruff; else command -v ruff; fi)
REGION ?= us-east4

.PHONY: all verify verify-live report test lint stage deploy warm cool tunnel teardown mock clean

# No GPU needed: lint, tests and the judge against the built-in mock server
all: lint test verify

verify:
	$(PYTHON) verify_first_server.py --mock

# Real numbers: needs the tunnel open (make tunnel) in another terminal
verify-live:
	$(PYTHON) verify_first_server.py

report:
	$(PYTHON) verify_first_server.py --verbose

test:
	$(PYTEST) -q

lint:
	$(RUFF) check .

stage:
	bash gcp/stage_weights.sh

deploy:
	bash gcp/deploy_vllm.sh

# Keep one GPU instance warm while you work (no cold starts), and let it scale to zero after
warm:
	gcloud run services update vllm-gemma --region $(REGION) --project $(GCP_PROJECT_ID) --min-instances 1

cool:
	gcloud run services update vllm-gemma --region $(REGION) --project $(GCP_PROJECT_ID) --min-instances 0

tunnel:
	gcloud run services proxy vllm-gemma --region $(REGION) --port 8000

teardown:
	bash gcp/teardown.sh

mock:
	$(PYTHON) -m vllm_lab.mock_server

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__
