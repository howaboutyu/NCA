setup:
	echo Installing jax and other dependencies
	chmod +x ./scripts/install_jax_gpu.sh
	bash ./scripts/install_jax_gpu.sh
	pip install -r requirements.txt
	echo Done ✌️

clean:
	@read -p "Are you sure you want to delete ckpts and logs? [y/N] " confirmation && \
    	if [ "$$confirmation" = "y" ] || [ "$$confirmation" = "Y" ] ; then \
    	    rm -rv ckpts ; \
    	    rm -rv logs ; \
    	else \
    	    echo "Aborted." ; \
    	fi
