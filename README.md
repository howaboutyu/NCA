<!-- <h3 align="center">Project Title</h3> -->



<div align="center">

[![pytest](https://github.com/howaboutyu/NCA/actions/workflows/pytest.yml/badge.svg)](https://github.com/howaboutyu/NCA/actions/workflows/pytest.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

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
- [Checkpoints](#checkpoints)

## 🧐 About <a name = "about"></a>

This project presents a Jax implementation of the Neural Cellular Automata (NCA) algorithm, based on the concepts outlined in the Distill paper [Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca/).  While the original implementation was developed using Tensorflow, this project serves as a re-implementation specifically tailored for Jax.


## 🏁 Getting Started <a name = "getting_started"></a>

These instructions will get you setup to train the model on your local machine, preferably with a GPU. 



### Installing

#### Without docker 

To install Jax refer to the [Jax documentation](https://github.com/google/jax#installation)

Install the other dependencies with pip

```
pip install -r requirements.txt
```


#### With GPU docker (recommended)

To build the Docker image and attach to it, run the following commands:


```bash
sudo make docker-build

# Attach to the docker container
sudo make start-devel
```


This docker is based on the [nvcr.io/nvidia/tensorflow:22.09-tf2-py](https://docs.nvidia.com/deeplearning/frameworks/tensorflow-release-notes/rel-22-09.html#rel-22-09) image. It has CUDA 11.8 and cuDNN 8.6, ensure that you satisfy the [driver requirements](https://docs.nvidia.com/deeplearning/frameworks/tensorflow-release-notes/rel-22-09.html#rel-22-09).



#### 🔧 Running the tests <a name = "tests"></a>

To verify if everything is installed correctly, you can run the following command to perform tests using pytest:

```bash
pytest
```

## 🎈 Usage <a name="usage"></a>

### Training

To train the model run the following command

```bash
python main.py --config configs/growing_nca.yaml 
``` 

Configuration settings can be defined using YAML files. The default configuration file to reproduce the results mentioned in the paper can be found at `configs/growing_nca_with_damage.yaml`. For all the default configurations, please refer to nca/configs.py.

To specify your own target image, you can modify the `target_filename` field in the YAML file to the desired image filename. Please ensure that the image has an alpha channel.


### Inference

To perform inference on a trained model, execute the following command. 


```bash
python main.py --config_path=configs/growing_demo.yaml --mode=evaluate --output_video_path=demo.mp4
2023
```

Please ensure that you update the `weights_dir` field in the configuration file with the accurate path to the downloaded checkpoint. Additionally, specify the `output_video_path` to determine the location where the NCA propagation will be saved in video format.

### 🔖 Checkpoints  <a name="checkpoints"></a>



| Checkpoint                                                                                                                             | Description                     |
|---------------------------------------------------------------------------------------------------------------------------------------|---------------------------------|
| [checkpoint_squinting_face_with_tongue](https://github.com/howaboutyu/NCA/releases/download/v1.0.0-squinting-face-with-tongue/checkpoint_squinting_face_with_tongue) | Squinting face with tongue model 😝 |

