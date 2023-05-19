<!-- <h3 align="center">Project Title</h3> -->



<div align="center">

[![pytest](https://github.com/howaboutyu/NCA/actions/workflows/pytest.yml/badge.svg)](https://github.com/howaboutyu/NCA/actions/workflows/pytest.yml)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)

</div>


---

<p align="center"> A Jax implementation of Neural Cellular Automata 
    <br> 
    <img src="./docs/gifs/output_half.gif" alt="NCA">

</p>


## 📝 Table of Contents

- [About](#about)
- [Getting Started](#getting_started)
- [Usage](#usage)
- [Built Using](#built_using)
- [TODO](../TODO.md)
- [Acknowledgments](#acknowledgement)

## 🧐 About <a name = "about"></a>

This project is a Jax implementation of Neural Cellular Automata (NCA) as described in [Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca/). The original implementation is in [Tensorflow](https://colab.research.google.com/github/google-research/self-organising-systems/blob/master/notebooks/growing_ca.ipynb#scrollTo=4O4tzfe-GRJ7), and this project is a re-implementation in Jax. 

## 🏁 Getting Started <a name = "getting_started"></a>

These instructions will get you setup to train the model on your local machine, preferably with a GPU. 



### Installing

#### Without docker 

Install the dependencies with pip

```
pip install -r requirements.txt
```

To install Jax refer to the [Jax documentation](https://github.com/google/jax#installation)

#### With GPU docker 

This docker is based on the [nvcr.io/nvidia/tensorflow:22.09-tf2-py](https://docs.nvidia.com/deeplearning/frameworks/tensorflow-release-notes/rel-22-09.html#rel-22-09) image. This image has CUDA 11.8 and cuDNN 8.6, ensure that you satisfy the [driver requirements](https://docs.nvidia.com/deeplearning/frameworks/tensorflow-release-notes/rel-22-09.html#rel-22-09).


```bash
sudo make docker-build

# attach to the docker container
sudo make start-devel
```


#### 🔧 Running the tests <a name = "tests"></a>

Check everything is installed correctly by running some tests 🔧 with pytest

```bash
pytest
```

## 🎈 Usage <a name="usage"></a>

### Training

To train the model run the following command

```bash
python main.py --config configs/growing_nca.yaml
``` 

#### Configs

The configs can be defined using [YAML](https://yaml.org/) files. The default config to replicate the results in the paper can be found in `configs/growing_nca_with_damage.yaml`. All the default configurations can be found in `nca/configs.py`.


#### Tensorboard logging

In this project we use [TensorboardX](https://github.com/lanpa/tensorboardX) was used. It logs the train/val losses, the training NCA state propagation as a gif and also the NCA propagation from the seed state. To view the logs run the following command

```bash 
tensorboard --logdir ./logs 
```

### Inference

To run inference on a trained model run the following command; we will demonstrate this on the pretrained model, you can download it using the command below

```bash
make download-pretrained
```

Alternatively, you can download the checkpoint from the GitHub releases page: [link](https://github.com/howaboutyu/NCA/releases/download/v1.0.0-squinting-face-with-tongue/checkpoint_squinting_face_with_tongue). Once the download is completed you can proceed with running the NCA inference step using the following command

```bash
python main.py --config_path=configs/growing_demo.yaml --mode=evaluate --output_video_path=demo.mp4
2023
```

Make sure to update the `weights_dir` in the configuration file with the correct path to the downloaded checkpoint. Specify the `output_video_path` as the location to save the NCA propagation in video format. When running in evaluate mode, the NCA process will begin from the initial seed state and continue for the number of steps defined in the configuration file as `num_nca_steps`.

## ⛏️ Built Using <a name = "built_using"></a>

- [Jax]() - Database
- [Flax]() - Server Framework
- [Optax]() - Web Framework
- [TensorboardX]() - Server Environment


## 🎉 Acknowledgements <a name = "acknowledgement"></a>

- Hat tip to the Distill team for their [blog post](https://distill.pub/2020/growing-ca/) on Neural Cellular Automata; an inspiration on how to write a blog post. 
