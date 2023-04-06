import jax
import jax.numpy as jnp
from flax.training import checkpoints, train_state
import optax
from dataclasses import dataclass
import cv2
import numpy as np

from nca.model import UpdateModel
from nca.nca import create_perception_kernel, perceive, cell_update


@dataclass
class Config:
    dimensions: tuple = (64, 64)
    model_output_len: int = 16

    num_epochs: int = 100
    steps_per_epoch: int = 100
    learning_rate: float = 1e-5


config = Config()

# create learning rate
learning_rate_schedule = optax.cosine_decay_schedule(
    init_value=config.learning_rate,
    decay_steps=config.num_epochs * config.steps_per_epoch,
)


model = UpdateModel(model_output_len=config.model_output_len)

key = jax.random.PRNGKey(0)

# dimensions = (64, 64) + (config.model_output_len * 3,))
params = model.init(
    key, jax.random.normal(key, (1, 64, 64, config.model_output_len * 3))
)


# create train state
tx = optax.adam(learning_rate=learning_rate_schedule)
state = train_state.TrainState.create(apply_fn=model, params=params, tx=tx)


def get_data():
    img = cv2.imread("emoji_imgs/skier.png", -1) / 255.0
    img = cv2.resize(img, config.dimensions)
    return img


def train_step(state, key, state_grid, target, num_steps=64):
    kernel_x, kernel_y = create_perception_kernel(
        input_size=config.model_output_len,
        output_size=config.model_output_len,
        use_oihw_layout=True,
    )

    def loss_fn(params, state_grid):
        for _ in range(num_steps):
            state_grid = cell_update(
                key,
                state_grid,
                model.apply,
                params,
                kernel_x,
                kernel_y,
                update_prob=0.5,
            )

        pred_grid = state_grid[:, :4]
        return jax.numpy.mean(jax.numpy.square(pred_grid - target))

    grad_fn = jax.value_and_grad(loss_fn)

    loss, grad = grad_fn(state.params, state_grid)
    print(f"loss : {loss}")

    state = state.apply_gradients(grads=grad)

    return state, loss


state_grid = np.zeros((1, 64, 64, 16), dtype=np.float32)
state_grid[0, 64 // 2, 64 // 2, 3:] = 1.0
target = get_data()

# NHWC -> NCHW
state_grid = np.transpose(state_grid, (0, 3, 1, 2))
target = np.transpose(target, (2, 0, 1))
target = np.expand_dims(target, axis=0)

for epoch in range(config.num_epochs):
    state, loss = train_step(state, key, state_grid, target)
