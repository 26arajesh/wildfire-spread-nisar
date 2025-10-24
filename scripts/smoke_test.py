"""Quick smoke tests: build model and run a tiny batch through it."""

from src.models import cnn
from src.data import dataset as ds
import tensorflow as tf


def run_smoke(tfrecord_path: str = None):
    print("Building model...")
    model = cnn.build_model()
    model.summary()

    if tfrecord_path:
        print("Loading a tiny dataset from:", tfrecord_path)
        d = ds.make_dataset(
            tfrecord_path, batch_size=2, repeat=False, augment=False, cache=False
        )
        for images, labels in d.take(1):
            print("Images shape:", images.shape)
            preds = model(images)
            print("Preds shape:", preds.shape)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--tfrecord", default=None)
    args = p.parse_args()
    run_smoke(args.tfrecord)
