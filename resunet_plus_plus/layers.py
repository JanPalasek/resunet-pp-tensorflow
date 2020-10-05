from tensorflow.keras.layers import *
import tensorflow as tf


class HeightWidthFlatten(Layer):
    def __init__(self, **kwargs):
        super(HeightWidthFlatten, self).__init__(**kwargs)

        self.reshape = None

    def build(self, input_shape):
        self.reshape = Reshape(target_shape=(input_shape[1] * input_shape[2], input_shape[3]))

    def call(self, inputs, training=None):
        return self.reshape(inputs)