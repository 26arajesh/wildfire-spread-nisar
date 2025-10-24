import tensorflow as tf

tfrecord_path = "data/train.tfrecords"

for example in tf.python_io.tf_record_iterator(tfrecord_path):
    print(tf.train.Example.FromString(example))
