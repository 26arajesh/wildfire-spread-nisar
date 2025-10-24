"""TFRecord dataset loader and preprocessing utilities.

Expect each TFRecord example to contain serialized image bytes (or raw floats) and a label.
This is a lightweight, configurable loader intended as a starting point.
"""

from typing import Optional, Dict, Any

import tensorflow as tf


def _parse_example(example_proto: tf.Tensor, feature_description: Dict[str, Any]):
    return tf.io.parse_single_example(example_proto, feature_description)


def _decode_image(image_bytes: tf.Tensor, target_shape: Optional[tuple] = None):
    # Accept raw JPEG/PNG bytes or raw float list
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    image = tf.image.convert_image_dtype(image, tf.float32)  # 0..1
    if target_shape is not None:
        image = tf.image.resize(image, target_shape)
    return image


def make_dataset(
    tfrecord_paths,
    image_key: str = "image_raw",
    label_key: str = "label",
    target_size: Optional[tuple] = (128, 128),
    batch_size: int = 32,
    shuffle: bool = True,
    repeat: bool = False,
    augment: bool = False,
    cache: bool = False,
):
    """Create a tf.data.Dataset from TFRecord(s).

    Args:
        tfrecord_paths: single path or list of paths to TFRecord files.
        image_key: feature name for the image bytes.
        label_key: feature name for the label (int or float).
        target_size: (h, w) to resize images to.
        batch_size: batch size.
        shuffle: whether to shuffle.
        repeat: whether to repeat indefinitely.
        augment: whether to apply simple augmentations.
        cache: whether to cache dataset in memory.

    Returns:
        tf.data.Dataset yielding (images, labels).
    """

    if isinstance(tfrecord_paths, str):
        tfrecord_paths = [tfrecord_paths]

    feature_description = {
        image_key: tf.io.FixedLenFeature([], tf.string),
        label_key: tf.io.FixedLenFeature([], tf.int64),
    }

    ds = tf.data.TFRecordDataset(tfrecord_paths)

    if shuffle:
        ds = ds.shuffle(buffer_size=1000)

    ds = ds.map(
        lambda x: _parse_example(x, feature_description),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    def _extract(example):
        img = _decode_image(example[image_key], target_shape=target_size)
        lbl = tf.cast(example[label_key], tf.int32)
        return img, lbl

    ds = ds.map(_extract, num_parallel_calls=tf.data.AUTOTUNE)

    if augment:

        def _augment(img, lbl):
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_flip_up_down(img)
            img = tf.image.random_brightness(img, 0.1)
            return img, lbl

        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)

    if cache:
        ds = ds.cache()

    if repeat:
        ds = ds.repeat()

    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def inspect_tfrecord(
    tfrecord_path: str,
    n: int = 5,
    image_key: str = "image_raw",
    label_key: str = "label",
):
    """Utility to print a few examples from a TFRecord (for sanity checks)."""
    feature_description = {
        image_key: tf.io.FixedLenFeature([], tf.string),
        label_key: tf.io.FixedLenFeature([], tf.int64),
    }
    ds = tf.data.TFRecordDataset([tfrecord_path])
    it = ds.take(n)
    for raw in it:
        ex = tf.io.parse_single_example(raw, feature_description)
        print({k: type(v) for k, v in ex.items()})
