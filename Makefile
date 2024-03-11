.PHONY: init data_extraction baseline_model model_training

init:

	curl -sSL https://install.python-poetry.org | python3 -poetry install


install: pyproject.toml

	poetry install 


clean:
	rm -rf 'find . -type d -name __pycache__'


check:
	poetry run ruff src/


data_extraction:

	poetry run python3 src/data_extraction.py


baseline_model:

	poetry run python3 src/training_pipeline/baseline.py


train:

	poetry run python3 src/training_pipeline/model_training.py
