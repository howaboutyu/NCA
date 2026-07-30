import jax
import jax.numpy as jnp

from nca.nca4d import (
    UpdateModel4D,
    cell_update_4d,
    project_xy_over_a_4d,
    render_xy_plane_4d,
    seed_volume_4d,
)


def test_4d_update_and_projection_shapes():
    key = jax.random.PRNGKey(0)
    state = seed_volume_4d(2, 8, 5, 6, 7, 3, seed_depth_index=1, seed_axis_index=2)
    model = UpdateModel4D(state_channels=8, hidden_channels=12)
    params = model.init(key, jnp.transpose(state, (0, 2, 3, 4, 5, 1)))

    next_state = cell_update_4d(key, state, model, params, update_probability=1.0)
    selected = render_xy_plane_4d(next_state, z_index=1, a_index=2)
    projected = project_xy_over_a_4d(next_state, z_index=1)

    assert next_state.shape == state.shape
    assert selected.shape == (2, 4, 6, 7)
    assert projected.shape == (2, 4, 6, 7)
    assert jnp.array_equal(selected, jnp.clip(next_state[:, :4, 1, :, :, 2], 0.0, 1.0))
    assert jnp.all(projected >= 0.0)
    assert jnp.all(projected <= 1.0)


def test_4d_conditional_embedding_and_invalid_seed_axis():
    key = jax.random.PRNGKey(1)
    state = seed_volume_4d(1, 8, 4, 4, 4, 2)
    model = UpdateModel4D(state_channels=8, hidden_channels=8, pokemon_vocab_size=3)
    ids = jnp.asarray([2], dtype=jnp.int32)
    params = model.init(key, jnp.transpose(state, (0, 2, 3, 4, 5, 1)), pokemon_ids=ids)
    updated = cell_update_4d(key, state, model, params, update_probability=1.0, pokemon_ids=ids)
    assert updated.shape == state.shape

    try:
        seed_volume_4d(1, 8, 4, 4, 4, 2, seed_axis_index=2)
    except ValueError as exc:
        assert "seed_axis_index" in str(exc)
    else:
        raise AssertionError("invalid A-axis seed index was accepted")
