<!-- <h3 align="center">Project Title</h3> -->



<div align="center">

[![pytest](https://github.com/howaboutyu/NCA/actions/workflows/pytest.yml/badge.svg)](https://github.com/howaboutyu/NCA/actions/workflows/pytest.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

</div>


---

<p align="center"> A Jax implementation of Neural Cellular Automata 
    <br> 
    <img src="./docs/gifs/output_half.gif" alt="NCA">
    <img src="./docs/gifs/pokemon_64ch_64x64_nosyn.gif" alt="64-channel 64x64 Pokémon reconstruction without synapses">

</p>


<h3 align="center">Inference gallery (conditional Pokémon)</h3>

<div align="center">
<table>
  <tr>
    <td align="center">
      <img src="./docs/gifs/inference_all_pokemon/pokemon_00_bulbasaur_cutout.gif" width="150" alt="Bulbasaur inference"><br/>
      Bulbasaur
    </td>
    <td align="center">
      <img src="./docs/gifs/inference_all_pokemon/pokemon_01_charmander_cutout.gif" width="150" alt="Charmander inference"><br/>
      Charmander
    </td>
    <td align="center">
      <img src="./docs/gifs/inference_all_pokemon/pokemon_02_squirtle_cutout.gif" width="150" alt="Squirtle inference"><br/>
      Squirtle
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="./docs/gifs/inference_all_pokemon/pokemon_03_pikachu_cutout.gif" width="150" alt="Pikachu inference"><br/>
      Pikachu
    </td>
    <td align="center">
      <img src="./docs/gifs/inference_all_pokemon/pokemon_04_eevee_cutout.gif" width="150" alt="Eevee inference"><br/>
      Eevee
    </td>
  </tr>
</table>
</div>

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

For NVIDIA GPU training, use the CUDA 12 JAX extra instead:

```bash
make setup-gpu
python -c "import jax; print(jax.devices())"
python main.py --config_path configs/growing_nca.yaml
```

The device check should report a `CudaDevice`. The Dockerfile provides the same
CUDA 12 setup for a containerized run.


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
python main.py --config_path configs/growing_nca.yaml
``` 

Configuration settings can be defined using YAML files. The default configuration file to reproduce the results mentioned in the paper can be found at `configs/growing_nca_with_damage.yaml`. For all the default configurations, please refer to nca/configs.py.

To specify your own target image, you can modify the `target_filename` field in the YAML file to the desired image filename. Please ensure that the image has an alpha channel.

Select the perception implementation with `perception_method`: `sobel` (the original two-convolution path), `sobel_fused` (an equivalent fused convolution), `sobel_second` (adds second derivatives), `sobel_multiscale` (combines 3x3 and 5x5 gradients), or `learned` (a trainable 3x3 convolutional perception block). Set `perception_method: sobel_second` and `nonlocal_connections: true` to add long-range context. Use `nonlocal_mode: global` for a pooled whole-grid summary, or `nonlocal_mode: token_attention` for gated attention to an 8x8 spatial token grid. Initial states support `seed_pattern: single`, `random`, or a compact `pokeball` icon; set `seed_size` to control the icon diameter in cells.

```bash
python scripts/benchmark_perception.py --steps 100 --nca-steps 32
```


### Inference

To perform inference on a trained model, execute the following command. 


```bash
python main.py --config_path=configs/growing_nca.yaml --mode=evaluate --output_video_path=demo.mp4
```

Please ensure that you update the `weights_dir` field in the configuration file with the accurate path to the downloaded checkpoint. Additionally, specify the `output_video_path` to determine the location where the NCA propagation will be saved in video format.

If the model is trained with conditional Pokémon targets, run all identities in one command with cutout simulation:

```bash
python main.py --config_path=path/to/config.yaml --mode=evaluate_all_pokemon --output_dir=./runs/inference_all_pokemon
```

This command writes one `pokemon_<id>_<name>_cutout.mp4` file per configured Pokémon target under the given output directory (defaulting to your chosen `./runs/...` path). These videos are runtime outputs and are intentionally excluded from git tracking.

### 🔖 Checkpoints  <a name="checkpoints"></a>



| Checkpoint                                                                                                                             | Description                     |
|---------------------------------------------------------------------------------------------------------------------------------------|---------------------------------|
| [checkpoint_squinting_face_with_tongue](https://github.com/howaboutyu/NCA/releases/download/v1.0.0-squinting-face-with-tongue/checkpoint_squinting_face_with_tongue) | Squinting face with tongue model 😝 |

