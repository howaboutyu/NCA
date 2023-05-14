from context import *

import pytest
import jax
import numpy as np
import os
from nca.utils import make_gif, make_video, download_pokemon_emoji


def test_make_gif():
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (10, 32, 32, 3))
    x = (x + 1.0) / 2.0
    x = (x * 255).astype(np.uint8)

    x_array = [x_ for x_ in x]

    make_gif(x_array, "/tmp/test.gif")

    assert os.path.exists("/tmp/test.gif")


def test_make_video():
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (10, 320, 320, 3))
    x = (x + 1.0) / 2.0
    x = (x * 255).astype(np.uint8)

    x_array = [x_ for x_ in x]

    make_video(x_array, "/tmp/test.mp4")

    assert os.path.exists("/tmp/test.mp4")


def test_download_pokemon_emoji():
    emoji_dict = download_pokemon_emoji("/tmp/pokemon_emoji")
    assert os.path.exists("/tmp/pokemon_emoji")

    assert "pikachu" in emoji_dict
    assert ".png" in emoji_dict["pikachu"]
