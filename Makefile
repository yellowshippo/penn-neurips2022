
# Global settings
CUDA = cu111  # CUDA version 11.1
PYTHON ?= poetry run python3
RUN = poetry run
PIP = pip3
RM = rm -rf


# User settings
DEVICE_ID = 0

## Gradient dataset
GRAD_MODEL ?= neumannisogcn
N_SAMPLE ?= 100
PRETRAINED_GRAD_MODEL ?= pretrained/grad/neumannisogcn

## Advection diffusion dataset
AD_MODEL ?= penn
PRETRAINED_AD_MODEL ?= pretrained/ad/penn

## Incompressible flow dataset
FLUID_MODEL ?= penn_n16_rep8
TW ?= 20
PRETRAINED_PENN_MODEL ?= pretrained/fluid/penn_n16_rep8
PRETRAINED_MPPDE_MODEL ?= pretrained/fluid/mp-pde_tw20_n128
MPPDE_N_HIDDEN=128


# Installation
## Install local libraries
install: poetry
	$(PYTHON) -m pip install lib/femio/dist/femio-*.whl lib/siml/dist/pysiml-*.whl
	make -C mp-neural-pde-solvers install  # Install baseline

install_cpu: poetry
	$(PYTHON) -m pip install lib/femio/dist/femio-*.whl lib/siml/dist/pysiml-*.whl
	make -C mp-neural-pde-solvers install_cpu  # Install baseline

poetry:
	$(PIP) install pip --upgrade
	cd lib/femio && poetry build --format=wheel && cd ../..
	cd lib/siml && poetry build --format=wheel && cd ../..
	poetry config virtualenvs.create true
	poetry config virtualenvs.in-project true
	-$(PYTHON) -m pip uninstall -y femio pysiml

pretrained_models:
	wget "https://savanna.ritc.jp/~horiem/penn_neurips2022/pretrained/pretrained_20220802.tar.gz" -O pretrained/pretrained.tar.gz
	# Alternative way in case of trouble
	# wget "https://drive.google.com/uc?export=download&id=1gJegDubmIC2SrlxzQDouCvFBHcBtoU47&confirm=t" -O pretrained/pretrained.tar.gz
	cd pretrained && tar xvf pretrained.tar.gz && mv pretrained_20220802/* . && rmdir pretrained_20220802


# Gradient dataset
grad_data:
	wget "https://savanna.ritc.jp/~horiem/penn_neurips2022/data/grad/grad_data.tar.gz" -O data/grad/grad_data.tar.gz
	wget "https://savanna.ritc.jp/~horiem/penn_neurips2022/data/grad/grad_interim.tar.gz" -O data/grad/grad_interim.tar.gz
	# Alternative way in case of trouble
	# wget "https://drive.google.com/uc?export=download&id=113nzfpwRZSsDjZeNU9kg0j9X7QBSC49Y&confirm=t" -O data/grad/grad_data.tar.gz
	# wget "https://drive.google.com/uc?export=download&id=1e94-463XeRtddynBrqKm041HlFci7yXn&confirm=t" -O data/grad/grad_interim.tar.gz
	cd data/grad && tar xvf grad_data.tar.gz && tar xvf grad_interim.tar.gz

generate_grad:
	$(RM) data/grad/interim data/grad/preprocessed
	$(PYTHON) src/generate_grad.py data/grad/interim/train -n $(N_SAMPLE) -d 10
	$(PYTHON) src/generate_grad.py data/grad/interim/validation -n $(N_SAMPLE) -d 10
	$(PYTHON) src/generate_grad.py data/grad/interim/test -n $(N_SAMPLE) -d 10
	$(PYTHON) src/preprocess_interim_data.py data/grad/data.yml -r true

grad_train:
	$(PYTHON) src/train.py data/grad/$(strip $(GRAD_MODEL)).yml \
		--gpu-id $(DEVICE_ID) \
		--out-dir models/grad/$(strip $(GRAD_MODEL))

grad_eval:
	$(PYTHON) src/eval.py $(PRETRAINED_GRAD_MODEL) \
		data/grad/preprocessed/test \
		--preprocessors-pkl data/grad/preprocessed/preprocessors.pkl \
		--write-simulation-base data/grad/interim \
		--analyse-error-mode grad \
		--output-base results/grad/$(strip $(GRAD_MODEL))


# Advection diffusion dataset
ad_data: data/ad/preprocessed

data/ad/preprocessed:
	wget "https://savanna.ritc.jp/~horiem/penn_neurips2022/data/ad/ad_preprocessed.tar.gz" -O data/ad/ad_preprocessed.tar.gz
	wget "https://savanna.ritc.jp/~horiem/penn_neurips2022/data/ad/ad_interim.tar.gz" -O data/ad/ad_interim.tar.gz
	# Alternative way in case of trouble
	# wget "https://drive.google.com/uc?export=download&id=1bQ6RKcSbHsg2D0gikYRelCBu6bN88Lk2&confirm=t" -O data/ad/ad_preprocessed.tar.gz
	# wget "https://drive.google.com/uc?export=download&id=1ek9jy0quseeMvTqb-0DkQl-B85n7v0c9&confirm=t" -O data/ad/ad_interim.tar.gz
	cd data/ad && tar xvf ad_preprocessed.tar.gz && tar xvf ad_interim.tar.gz

ad_eval:
	$(PYTHON) src/eval.py \
		$(PRETRAINED_AD_MODEL) \
		data/ad/preprocessed/test \
		--preprocessors-pkl data/ad/preprocessed/preprocessors.pkl \
		--write-simulation-base data/ad/interim \
		--analyse-error-mode ad \
		--output-base results/ad/$(notdir $(PRETRAINED_AD_MODEL))

ad_train:
	$(PYTHON) src/train.py data/ad/$(strip $(AD_MODEL)).yml \
		--gpu-id $(DEVICE_ID) \
		--out-dir models/ad/$(strip $(AD_MODEL))

ad_bnd_data:
	$(PYTHON) src/advdiff_convert_raw_data.py data/ad_bnd/data.yml \
		--upper-boundary true
	$(PYTHON) src/preprocess_interim_data.py data/ad_bnd/data.yml \
		--recursive false


# Incompressible flow dataset
fluid: fluid_data fluid_train

fluid_data: data/fluid/preprocessed

data/fluid/preprocessed:
	wget "https://savanna.ritc.jp/~horiem/penn_neurips2022/data/fluid/fluid_data.tar.gz.partaa" -O data/fluid/fluid_data.tar.gz.partaa
	wget "https://savanna.ritc.jp/~horiem/penn_neurips2022/data/fluid/fluid_data.tar.gz.partab" -O data/fluid/fluid_data.tar.gz.partab
	wget "https://savanna.ritc.jp/~horiem/penn_neurips2022/data/fluid/fluid_data.tar.gz.partac" -O data/fluid/fluid_data.tar.gz.partac
	wget "https://savanna.ritc.jp/~horiem/penn_neurips2022/data/fluid/fluid_data.tar.gz.partad" -O data/fluid/fluid_data.tar.gz.partad
	wget "https://savanna.ritc.jp/~horiem/penn_neurips2022/data/fluid/fluid_data.tar.gz.partae" -O data/fluid/fluid_data.tar.gz.partae
	wget "https://savanna.ritc.jp/~horiem/penn_neurips2022/data/fluid/fluid_data_interim.tar.gz" -O data/fluid/fluid_data_interim.tar.gz
	# Alternative way in case of trouble
	# wget "https://drive.google.com/uc?export=download&id=1x10-rsRp1XckS-OiXpFCM9l6f2xM0v0I&confirm=t" -O data/fluid/fluid_data.tar.gz.partaa
	# wget "https://drive.google.com/uc?export=download&id=1Q8cC5Lh3LYsesdt6LveFPzBLoLQVuliT&confirm=t" -O data/fluid/fluid_data.tar.gz.partab
	# wget "https://drive.google.com/uc?export=download&id=15eH9jvUWHvGrWldB0BcwuqRtqR8UnzlS&confirm=t" -O data/fluid/fluid_data.tar.gz.partac
	# wget "https://drive.google.com/uc?export=download&id=1GdntVtfG6wgU_8FQMXsCmpylhylWCo8B&confirm=t" -O data/fluid/fluid_data.tar.gz.partad
	# wget "https://drive.google.com/uc?export=download&id=1aIykQ2Yyp-591GMhIX5dHkt0d6hevtYB&confirm=t" -O data/fluid/fluid_data.tar.gz.partae
	# wget "https://drive.google.com/uc?export=download&id=1a-W6oB9DZL6T9ewbSGvQI8a1FI71OxzW&confirm=t" -O data/fluid/fluid_data_interim.tar.gz
	cat data/fluid/fluid_data.tar.gz.part* > data/fluid/fluid_data.tar.gz
	cd data/fluid && tar xvf fluid_data.tar.gz && tar xvf fluid_data_interim.tar.gz

fluid_train:
	$(PYTHON) src/train.py data/fluid/$(strip $(FLUID_MODEL)).yml \
		--gpu-id $(DEVICE_ID) \
		--continue-training true \
		--lr 5.0e-4

fluid_eval:
	OMP_NUM_THREADS=1 $(PYTHON) src/eval.py \
		$(PRETRAINED_PENN_MODEL) \
		data/fluid/preprocessed/test \
		--preprocessors-pkl data/fluid/preprocessed/preprocessors.pkl \
		--write-simulation-base data/fluid/interim \
		--analyse-error-mode fluid \
		--output-base results/fluid/$(notdir $(PRETRAINED_PENN_MODEL))

transformed_fluid_eval:
	$(PYTHON) src/eval.py \
		$(PRETRAINED_PENN_MODEL) \
		data/fluid/transformed/preprocessed \
		--preprocessors-pkl data/fluid/preprocessed/preprocessors.pkl \
		--write-simulation-base data/fluid/transformed/interim \
		--analyse-error-mode fluid \
		--output-base results/transformed_fluid/$(notdir $(PRETRAINED_PENN_MODEL))

fluid_train_mppde:
	make -C mp-neural-pde-solvers train_tw$(TW)

fluid_eval_mppde:
	make -C mp-neural-pde-solvers eval_tw$(TW) \
		SAVE_DIRECTORY=../results/fluid/$(notdir $(PRETRAINED_MPPDE_MODEL)) \
		MODEL_PATH=../$(PRETRAINED_MPPDE_MODEL)/model.pt \
		HIDDEN_FEATURES=$(MPPDE_N_HIDDEN)
	$(PYTHON) src/generate_vtu.py \
		results/fluid/$(notdir $(PRETRAINED_MPPDE_MODEL)) \
		data/fluid/interim/test \
		-p data/fluid/preprocessed/preprocessors.pkl

transformed_fluid_eval_mppde:
	make -C mp-neural-pde-solvers transformed_eval_tw$(TW) \
		SAVE_DIRECTORY=../results/transformed_fluid/$(notdir $(PRETRAINED_MPPDE_MODEL)) \
		MODEL_PATH=../$(PRETRAINED_MPPDE_MODEL)/model.pt \
		HIDDEN_FEATURES=$(MPPDE_N_HIDDEN)
	$(PYTHON) src/generate_vtu.py \
		results/transformed_fluid/$(notdir $(PRETRAINED_MPPDE_MODEL)) \
		data/fluid/transformed/interim/test \
		-p data/fluid/preprocessed/preprocessors.pkl


# Test
test_grad: generate_grad grad_train
	make grad_eval PRETRAINED_GRAD_MODEL=models/grad/$(strip $(GRAD_MODEL))

test_ad:
	$(RM) tests/data/advection_diffusion/penn
	$(PYTHON) src/train.py tests/data/advection_diffusion/penn.yml \
		--gpu-id $(DEVICE_ID) \
		--out-dir tests/data/advection_diffusion/penn
	$(PYTHON) src/eval.py tests/data/advection_diffusion/penn \
		tests/data/advection_diffusion/preprocessed \
		--preprocessors-pkl tests/data/advection_diffusion/preprocessed/preprocessors.pkl \
		--write-simulation-base tests/data/advection_diffusion/interim \
		--analyse-error-mode ad

test_ad_eval:
	$(PYTHON) src/eval.py \
		tests/data/pretrained/advection_diffusion/penn \
		tests/data/advection_diffusion/preprocessed \
		--preprocessors-pkl tests/data/advection_diffusion/preprocessed/preprocessors.pkl \
		--write-simulation-base tests/data/advection_diffusion/interim \
		--analyse-error-mode ad

test_fluid:
	$(RM) tests/data/fluid/penn
	$(PYTHON) src/train.py tests/data/fluid/penn.yml \
		--gpu-id $(DEVICE_ID) \
		--out-dir tests/data/fluid/penn \
		--continue-training true \
		--lr 5.0e-4
	$(PYTHON) src/eval.py tests/data/fluid/penn \
		tests/data/fluid/preprocessed \
		--preprocessors-pkl tests/data/fluid/preprocessed/preprocessors.pkl \
		--write-simulation-base tests/data/fluid/interim \
		--analyse-error-mode fluid

test_fluid_eval:
	$(PYTHON) src/eval.py \
		tests/data/pretrained/fluid/penn_n4_rep4 \
		tests/data/fluid/preprocessed \
		--preprocessors-pkl tests/data/fluid/preprocessed/preprocessors.pkl \
		--write-simulation-base tests/data/fluid/interim \
		--analyse-error-mode fluid


# Other
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	$(RM) models/* data/grad/interim data/grad/preprocessed

## Delete all data
delete_all_data: clean
	$(RM) data/interim/*
	$(RM) data/preprocessed/*

deploy_test_models:
	$(RM) tests/data/pretrained
	$(PYTHON) src/deploy_models.py -k fluid
	$(PYTHON) src/deploy_models.py -k ad
