from nca.config import NCAConfig, write_config, load_config
from nca.utils import download_pokemon_emoji


def get_pokemon_configs(
    pokemon_name: str,
    target_filename: str,
) -> NCAConfig:
    """Returns a NCAConfig object for a given pokemon.

    Args:
        pokemon_name: The name of the pokemon.
        target_filename: The filename of the target image.

    Returns:
        A NCAConfig object.
    """

    config = NCAConfig()
    config.batch_size = 4
    config.n_damage = 1
    config.target_filename = target_filename
    config.weights_dir = f"checkpoints/{pokemon_name}"
    config.checkpoint_dir = f"checkpoints/{pokemon_name}"
    config.validation_video_dir = f"validation_videos/{pokemon_name}"
    config.log_dir = f"logs/{pokemon_name}"
    config.evaluation_video_file = f"/tmp/evaluation_video_{pokemon_name}.mp4"

    return config


def main():
    pokemon_dict = download_pokemon_emoji()

    for pokemon_name, target_filename in pokemon_dict.items():
        config = get_pokemon_configs(pokemon_name, target_filename)
        write_config(config, f"pokemon_configs/{pokemon_name}.yaml")


if __name__ == "__main__":
    main()
