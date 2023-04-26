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
