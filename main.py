import tensorflow as tf
from absl import app, flags


from nca.trainer import train_and_evaluate
from nca.config import load_config


FLAGS = flags.FLAGS
flags.DEFINE_string("config_path", None, "config file path")
flags.DEFINE_string("mode", "train_and_eval", "can be train_and_eval or render")


flags.mark_flag_as_required("config_path")


def main(argv):
    tf.config.experimental.set_visible_devices([], "GPU")

    if FLAGS.mode == "train_and_eval":
        config = load_config(FLAGS.config_path)

        train_and_evaluate(FLAGS.config_path)


if __name__ == "__main__":
    app.run(main)
