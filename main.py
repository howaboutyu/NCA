import tensorflow as tf  # type: ignore
from absl import app, flags  # type: ignore
import wandb

from nca.trainer import train_and_evaluate, evaluate
from nca.config import load_config

FLAGS = flags.FLAGS
flags.DEFINE_string("config_path", None, "config file path")
flags.DEFINE_enum(
    "mode",
    "train_and_eval",
    ["train_and_eval", "evaluate"],
    "Mode can be train_and_eval or evaluate",
)
flags.DEFINE_string(
    "output_video_path",
    None,
    "Output video path to save the rendered NCA (required when mode is evaluate)",
)

flags.mark_flag_as_required("config_path")


def main(argv):
    del argv

    tf.config.experimental.set_visible_devices([], "GPU")

    config = load_config(FLAGS.config_path)
    wandb.init(project="nca", sync_tensorboard=True, config=config)

    if FLAGS.mode == "train_and_eval":
        train_and_evaluate(config)
    elif FLAGS.mode == "evaluate":
        if not FLAGS.output_video_path:
            raise ValueError(
                "Output video path must be specified when running in evaluation mode."
            )
        evaluate(config, FLAGS.output_video_path)
    wandb.finish()


if __name__ == "__main__":
    app.run(main)
