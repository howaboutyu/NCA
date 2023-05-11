# Variables
INSTALL_SCRIPT = ./scripts/install_jax_gpu.sh
REQUIREMENTS_FILE = requirements.txt
DOCKER_IMAGE = nca
DOCKERFILE = Dockerfile


# Targets
.PHONY: setup clean

setup:
	@echo "Installing JAX and other dependencies..."
	chmod +x $(INSTALL_SCRIPT)
	bash $(INSTALL_SCRIPT)
	@echo "Installing Python dependencies..."
	pip install -r $(REQUIREMENTS_FILE)
	@echo "Done ✌️"

clean:
	@read -p "Are you sure you want to delete ckpts and logs? [y/N] " confirmation && \
    	if [ "$$confirmation" = "y" ] || [ "$$confirmation" = "Y" ] ; then \
    	    rm -rv ckpts ; \
    	    rm -rv logs ; \
    	else \
    	    echo "Aborted." ; \
    	fi


docker-build:
	@echo "Building Docker image $(DOCKER_IMAGE)..."
	docker build -t $(DOCKER_IMAGE) -f $(DOCKERFILE) .
	@echo "Done building docker image ✌️"


start-devel:
	@echo "Starting development docker"
	docker run -it \
	    --gpus=all \
	    --ipc=host \
	    --ulimit memlock=-1 \
	    --ulimit stack=67108864 \
	    -v`pwd`:/nca \
	    nca:latest \
	    bash

download-pokemon-data:
	@if [ -d "slack-emoji-pokemon" ]; then \
		echo "Pokemon emoji data already downloaded. Skipping..."; \
	else \
		echo "Downloading Pokemon emoji data..."; \
		git clone https://github.com/Templarian/slack-emoji-pokemon; \
		echo "Done! The emojis are located in slack-emoji-pokemon/emojis"; \
	fi
