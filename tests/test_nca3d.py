import jax
import jax.numpy as jnp

from nca.nca3d import (
    UpdateModel3D,
    cell_update_3d,
    render_selected_plane_3d,
    seed_volume_3d,
)


def test_3d_update_and_projection_shapes():
    key = jax.random.PRNGKey(0)
    state = seed_volume_3d(2, 16, 12, 12, 12)
    model = UpdateModel3D(state_channels=16, hidden_channels=16)
    params = model.init(key, jnp.transpose(state, (0, 2, 3, 4, 1)))

    next_state = cell_update_3d(key, state, model, params, update_probability=1.0)
    rendered = render_selected_plane_3d(next_state, axis="z", index=6)

    assert next_state.shape == state.shape
    assert rendered.shape == (2, 4, 12, 12)
    assert jnp.array_equal(rendered, jnp.clip(next_state[:, :4, 6], 0.0, 1.0))
    assert render_selected_plane_3d(next_state, axis="y", index=6).shape == rendered.shape
    assert render_selected_plane_3d(next_state, axis="x", index=6).shape == rendered.shape
    assert jnp.all(rendered >= 0.0)
    assert jnp.all(rendered <= 1.0)
