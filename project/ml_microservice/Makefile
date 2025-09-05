PY=python
VENV=.venv
APP=app.py

init:
	python3 -m venv $(VENV) && . $(VENV)/bin/activate && python -m pip install -U pip && pip install -r requirements.txt

run:
	uvicorn app:app --host 0.0.0.0 --port 8000 --reload

train:
	MLFLOW_EXPERIMENT_NAME=churn-exp $(PY) -m src.train --config configs/config.yaml

evaluate:
	MLFLOW_EXPERIMENT_NAME=churn-exp $(PY) -m src.evaluate --config configs/config.yaml

test:
	pytest -q

lint:
	flake8 .

format:
	ruff check --fix . || true

build:
	docker build -t mlops-microservice:latest .
