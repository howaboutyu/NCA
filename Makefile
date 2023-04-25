# Variables
INSTALL_SCRIPT = ./scripts/install_jax_gpu.sh
REQUIREMENTS_FILE = requirements.txt

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
