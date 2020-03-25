import tensorflow as tf


def test_example_function():
    out = tf.constant(3) + tf.constant(4)
    assert 7 == out
