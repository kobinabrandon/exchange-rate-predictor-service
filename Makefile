.PHONY: init data_extraction baseline_model model_training deploy prepare_deployment test-endpoint

DEPLOYMENT_DIR = deployment_dir

init:

	curl -sSL https://install.python-poetry.org | python3 -poetry install

data_extraction:

	poetry run python3 src/data_extraction.py

baseline_model:

	poetry run python3 src/baseline.py

train:

	poetry run python3 src/train.py

prepare_deployment:

	# rm -rf ${DEPLOYMENT_DIR} && mkdir ${DEPLOYMENT_DIR}

	poetry export -f requirements.txt --output ${DEPLOYMENT_DIR}/requirements.txt --without-hashes

	cp -r src/predict.py ${DEPLOYMENT_DIR}/main.py 

	# cp -r src ${DEPLOYMENT_DIR}/src/

	# pip install cerebrium --upgrade 

deploy: prepare_deployment

	cd ${DEPLOYMENT_DIR} && poetry run cerebrium deploy --api-key $(CEREBRIUM_API_KEY) --hardware-CPU 