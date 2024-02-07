.PHONY: *

VENV=venv
PYTHON=$(VENV)/bin/python3


# ================== LOCAL WORKSPACE SETUP ==================
venv:
	python3 -m venv $(VENV)
	@echo 'Path to Python executable $(shell pwd)/$(PYTHON)'

install_all: venv
	@echo "=== Installing common dependencies ==="
	$(PYTHON) -m pip install -r requirements.txt

	make pre_commit_install

# ========================= TRAINING ========================
run_training:
	$(PYTHON) -m  src.train ./configs/dinov2.yaml


# ========================= SUBMISSON ========================
make_submission:
	$(PYTHON) -m  src.make_submission ./configs/dinov2.yaml

