import jax
import jax.numpy as jnp
from flax.training import checkpoints, train_state
import optax
from dataclasses import dataclass
import cv2
import numpy as np

from nca.model import UpdateModel
from nca.nca import create_perception_kernel, perceive, cell_update


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
    key,
    jax.random.normal(
        key,
        (1, config.dimensions[0], config.dimensions[1], config.model_output_len * 3),
    ),
)


# create train state
tx = optax.adam(learning_rate=config.learning_rate)
state = train_state.TrainState.create(apply_fn=model, params=params, tx=tx)


def get_data():
    img = cv2.imread("emoji_imgs/smile.png", cv2.IMREAD_UNCHANGED)
    assert img.shape[-1] == 4

    alpha = img[..., -1] > 0
    img = img[..., :3] * alpha[..., np.newaxis]

    img = cv2.resize(img, config.dimensions) / 255.0

    cv2.imwrite("target.png", img * 255)

    return img


@jax.jit
def train_step(state, key, state_grid, target):
    kernel_x, kernel_y = create_perception_kernel(
        input_size=config.model_output_len,
        output_size=config.model_output_len,
        use_oihw_layout=True,
    )

    def loss_fn(params, state_grid, key):
        def body_fun(i, vals):
            key, state_grid = vals
            subkey, key = jax.random.split(key)
            state_grid = cell_update(
                subkey,
                state_grid,
                model.apply,
                params,
                kernel_x,
                kernel_y,
                update_prob=0.5,
            )
            return (key, state_grid)

        num_steps = jax.random.randint(key, (1,), 64, 97)[0]
        (key, state_grid) = jax.lax.fori_loop(0, 64, body_fun, (key, state_grid))

        pred_rgb = state_grid[:, :3]
        alpha = state_grid[:, 3:4]
        alpha = jnp.clip(alpha, 0, 1)

        pred_rgb = pred_rgb * alpha
        return jax.numpy.mean(jax.numpy.square(pred_rgb - target))

    grad_fn = jax.value_and_grad(loss_fn)

    loss, grad = grad_fn(state.params, state_grid, key)

    print(f"loss : {loss}")

    state = state.apply_gradients(grads=grad)

    return state, loss


@jax.jit
def render_state(state, key, state_grid):
    kernel_x, kernel_y = create_perception_kernel(
        input_size=config.model_output_len,
        output_size=config.model_output_len,
        use_oihw_layout=True,
    )

    pred_rgb_array = []
    for _ in range(64):
        state_grid = cell_update(
            key,
            state_grid,
            model.apply,
            state.params,
            kernel_x,
            kernel_y,
            update_prob=0.5,
        )
        key, _ = jax.random.split(key)

        pred_rgb = state_grid[:, :3]
        alpha = state_grid[:, 3:4]

        alpha = jnp.clip(alpha, 0, 1)

        pred_rgb = pred_rgb * alpha

        pred_rgb_array.append(pred_rgb)

    return pred_rgb_array


for epoch in range(config.num_epochs):
    state_grid = np.zeros(
        (config.batch_size, config.dimensions[0], config.dimensions[1], 16),
        dtype=np.float32,
    )
    state_grid[:, config.dimensions[0] // 2, config.dimensions[1] // 2, 3:] = 1.0
    target = np.asarray([get_data()] * config.batch_size)

    # NHWC -> NCHW
    state_grid = np.transpose(state_grid, (0, 3, 1, 2))
    target = np.transpose(target, (0, 3, 1, 2))

    print(f"Epoch: {epoch}")
    state, loss = train_step(state, key, state_grid, target)
    print(f"Loss : {loss}")
    key, _ = jax.random.split(key)

    # Render
    if epoch % 1_000 != 0:
        continue
    predicted_sequence = render_state(state, key, state_grid)

    for cc, img in enumerate(predicted_sequence):
        img_uint8 = np.asarray(img * 255).astype(np.uint8)

        img_uint8 = img_uint8.transpose(0, 2, 3, 1)[0]
        print(f"max predicted: {np.max(img_uint8)}")
        # import pdb; pdb.set_trace()
        cv2.imwrite(f"img_{str(cc).zfill(5)}.png", img_uint8)
