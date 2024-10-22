.PHONY: init data_extraction baseline_model model_training

data: 
	poetry run python3 src/data_extraction.py

train:

	poetry run python3 src/training_pipeline/model_training.py