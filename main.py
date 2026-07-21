from absl import app, flags  # type: ignore

from nca.trainer import train_and_evaluate, evaluate, evaluate_all_pokemon
from nca.config import load_config

FLAGS = flags.FLAGS
flags.DEFINE_string("config_path", None, "config file path")
flags.DEFINE_enum(
    "mode",
    "train_and_eval",
    ["train_and_eval", "evaluate", "evaluate_all_pokemon"],
    "Mode can be train_and_eval, evaluate, or evaluate_all_pokemon",
)
flags.DEFINE_string(
    "output_video_path",
    None,
    "Output video path to save the rendered NCA (required when mode is evaluate)",
)
flags.DEFINE_string(
    "output_dir",
    None,
    "Output directory for rendered Pokémon-specific inference MP4s",
)

flags.mark_flag_as_required("config_path")


def main(argv):
    del argv

    config = load_config(FLAGS.config_path)

    if FLAGS.mode == "train_and_eval":
        train_and_evaluate(config)
    elif FLAGS.mode == "evaluate":
        if not FLAGS.output_video_path:
            raise ValueError(
                "Output video path must be specified when running in evaluation mode."
            )
        evaluate(config, FLAGS.output_video_path)
    elif FLAGS.mode == "evaluate_all_pokemon":
        if not FLAGS.output_dir:
            raise ValueError(
                "Output directory must be specified when running in evaluate_all_pokemon mode."
            )
        evaluate_all_pokemon(config, FLAGS.output_dir)


if __name__ == "__main__":
    app.run(main)
