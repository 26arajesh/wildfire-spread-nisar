"""Training entrypoint for wildfire-spread-nisar.

Example usage:
    python -m src.train --tfrecord data/train.tfrecord --epochs 10 --batch_size 32
"""

import argparse
import os
from datetime import datetime

from src.data import dataset as ds
from src.models import cnn


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--tfrecord",
        required=True,
        help="Path to TFRecord file or directory containing TFRecords",
    )
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--output_dir", default="artifacts")
    p.add_argument("--target_size", type=int, nargs=2, default=(128, 128))
    p.add_argument("--num_classes", type=int, default=2)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    train_ds = ds.make_dataset(
        args.tfrecord,
        target_size=tuple(args.target_size),
        batch_size=args.batch_size,
        shuffle=True,
        repeat=False,
        augment=True,
        cache=False,
    )

    model = cnn.build_model(
        input_shape=(args.target_size[0], args.target_size[1], 3),
        num_classes=args.num_classes,
    )

    log_dir = os.path.join(
        args.output_dir, "logs", datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    ckpt_path = os.path.join(args.output_dir, "checkpoints", "model_{epoch:02d}.h5")
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
        ckpt_path, save_weights_only=False, save_best_only=False
    )

    model.fit(train_ds, epochs=args.epochs, callbacks=[tensorboard_cb, ckpt_cb])

    final_path = os.path.join(args.output_dir, "model_final.h5")
    model.save(final_path)
    print("Saved final model to", final_path)


if __name__ == "__main__":
    import tensorflow as tf

    main()
