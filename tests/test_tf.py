from unittest import TestCase
import tensorflow as tf


class Test(TestCase):
    def test_example_function(self):
        out = tf.constant(3) + tf.constant(4)
        self.assertEqual(7, out)
