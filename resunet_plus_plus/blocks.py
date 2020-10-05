from tensorflow.keras.layers import *
from resunet_plus_plus.layers import *
import tensorflow as tf


class ResBlock(Layer):
    def __init__(self, filters, strides, **kwargs):
        super(ResBlock, self).__init__(**kwargs)
        self.filters = filters
        self.strides = strides

        self.bn1 = BatchNormalization()
        self.relu1 = ReLU()
        self.conv1 = Conv2D(filters=filters, kernel_size=3, strides=strides, padding="same", use_bias=False)

        self.bn2 = BatchNormalization()
        self.relu2 = ReLU()
        self.conv2 = Conv2D(filters=filters, kernel_size=3, strides=1, padding="same", use_bias=False)

        self.conv_skip = Conv2D(filters=filters, kernel_size=1, strides=strides, padding="same", use_bias=False)
        self.bn_skip = BatchNormalization()

        self.add = Add()

    def call(self, inputs, training=False, **kwargs):
        x = inputs
        x = self.bn1(x, training=training)
        x = self.relu1(x)
        x = self.conv1(x)

        x = self.bn2(x, training=training)
        x = self.relu2(x)
        x = self.conv2(x)

        skip = self.conv_skip(inputs)
        skip = self.bn_skip(skip, training=training)

        res = self.add([x, skip])
        return res

    def get_config(self):
        return dict(filters=self.filters, strides=self.strides, **super(ResBlock, self).get_config())


class SqueezeAndExcitationBlock(Layer):
    def __init__(self, channels, ratio=16, **kwargs):
        super(SqueezeAndExcitationBlock, self).__init__(**kwargs)

        self.channels = channels
        self.ratio = ratio

        self.gap = GlobalAveragePooling2D()
        self.dense1 = Dense(self.channels // self.ratio, activation="relu")
        self.dense2 = Dense(self.channels, activation="sigmoid")

        self.reshape = Reshape(target_shape=(1, 1, self.channels))

        self.mul = Multiply()

    def call(self, inputs, training=False, **kwargs):
        x = inputs

        x = self.gap(x)
        x = self.dense1(x)
        x = self.dense2(x)

        x = self.reshape(x)

        res = self.mul([inputs, x])
        return res

    def get_config(self):
        return dict(channels=self.channels, ratio=self.ratio, **super(SqueezeAndExcitationBlock, self).get_config())


class SelfAttentionBlock(Layer):
    def __init__(self, ratio, **kwargs):
        super(SelfAttentionBlock, self).__init__(**kwargs)

        self.ratio = ratio

        self.conv_query_1 = None
        self.conv_key_1 = None
        self.conv_value_1 = None

        self.hw_flatten_query_1 = HeightWidthFlatten()
        self.hw_flatten_key_1 = HeightWidthFlatten()
        self.hw_flatten_value_1 = HeightWidthFlatten()

        self.alpha = Softmax()

        self.reshape_1 = None
        self.conv_1 = None

        self.gamma_weight = self.add_weight(
            shape=(1,), initializer=tf.initializers.Constant(0),
            trainable=True)

    def build(self, input_shape):
        channels = input_shape[-1]
        self.conv_query_1 = Conv2D(filters=channels // self.ratio, kernel_size=1, strides=1, padding="same")
        self.conv_key_1 = Conv2D(filters=channels // self.ratio, kernel_size=1, strides=1, padding="same")
        self.conv_value_1 = Conv2D(filters=channels, kernel_size=1, strides=1, padding="same")

        self.reshape_1 = Reshape(target_shape=(input_shape[1], input_shape[2], channels))

        self.conv_1 = Conv2D(filters=channels, kernel_size=1, strides=1, padding="same")

    def call(self, inputs, training=None):
        query = self.conv_query_1(inputs, training=training)
        query = self.hw_flatten_query_1(query)

        key = self.conv_key_1(inputs, training=training)
        key = self.hw_flatten_key_1(key)

        relevance = tf.matmul(key, query, transpose_b=True)
        weights = self.alpha(relevance)

        value = self.conv_value_1(inputs, training=training)
        value = self.hw_flatten_value_1(value)

        # weight value and reshape
        x = tf.matmul(weights, value)

        x = self.reshape_1(x)
        x = self.conv_1(x)

        # add residual connection
        output = self.gamma_weight * x + inputs

        return output

    def get_config(self):
        return dict(filters=self.filters, **super(SelfAttentionBlock, self).get_config())


class SeparableConvolutionBlock(Layer):
    def __init__(self, filters, strides, dilation_rate, **kwargs):
        super(SeparableConvolutionBlock, self).__init__(**kwargs)

        self.filters = filters
        self.strides = strides
        self.dilation_rate = dilation_rate

        self.depth_conv_1 = DepthwiseConv2D(kernel_size=3, strides=strides,
                                            dilation_rate=dilation_rate, padding="same",
                                            use_bias=False)
        self.bn_1 = BatchNormalization()
        self.relu_1 = ReLU()

        self.conv_1 = Conv2D(filters=filters, kernel_size=1, strides=1, padding="same", use_bias=False)
        self.bn_2 = BatchNormalization()
        self.relu_2 = ReLU()

    def call(self, inputs, training=None) -> Layer:
        x = inputs

        x = self.depth_conv_1(x, training=training)
        x = self.bn_1(x, training=training)
        x = self.relu_1(x)

        x = self.conv_1(x, training=training)
        x = self.bn_2(x, training=training)
        x = self.relu_2(x)

        return x


class AtrousSpatialPyramidPooling(Layer):
    def __init__(self, filters, dilation_rates, **kwargs):
        super(AtrousSpatialPyramidPooling, self).__init__(**kwargs)

        self.filters = filters
        self.dilation_rates = dilation_rates

        self.sep_conv_block_1 = SeparableConvolutionBlock(filters=filters, strides=1,
                                                          dilation_rate=1)

        self.sep_conv_block_2 = SeparableConvolutionBlock(filters=filters, strides=1,
                                                          dilation_rate=dilation_rates[0])

        self.sep_conv_block_3 = SeparableConvolutionBlock(filters=filters, strides=1,
                                                          dilation_rate=dilation_rates[1])

        self.sep_conv_block_4 = SeparableConvolutionBlock(filters=filters, strides=1,
                                                          dilation_rate=dilation_rates[2])

        self.gap_1 = GlobalAveragePooling2D()
        self.reshape_1 = None
        self.conv_1 = Conv2D(filters=filters, kernel_size=1, padding="same", use_bias=False)
        self.bn_1 = BatchNormalization()
        self.relu_1 = ReLU()
        self.upsample_1 = None

        # merged layers
        self.concatenate_1 = Concatenate()
        self.conv_2 = Conv2D(filters=filters, kernel_size=1, padding="same", use_bias=False)
        self.bn_2 = BatchNormalization()
        self.relu_2 = ReLU()

    def build(self, input_shape):
        orig_size = input_shape[1:-1]
        channels = input_shape[-1]

        self.upsample_1 = UpSampling2D(size=orig_size, interpolation="bilinear")
        self.reshape_1 = Reshape(target_shape=(1, 1, channels))

    def call(self, inputs, training=None):
        x = inputs

        x1 = self.sep_conv_block_1(x, training=training)
        x2 = self.sep_conv_block_2(x, training=training)
        x3 = self.sep_conv_block_3(x, training=training)
        x4 = self.sep_conv_block_4(x, training=training)

        x5 = self.gap_1(x)
        x5 = self.reshape_1(x5)
        x5 = self.conv_1(x5)
        x5 = self.bn_1(x5, training=training)
        x5 = self.relu_1(x5)
        x5 = self.upsample_1(x5)

        x = self.concatenate_1([x1, x2, x3, x4, x5])
        x = self.conv_2(x)
        x = self.bn_2(x, training=training)
        x = self.relu_2(x)

        return x
