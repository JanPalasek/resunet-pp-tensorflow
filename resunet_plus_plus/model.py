from tensorflow.keras.models import Model
from tensorflow.keras.layers import *

from resunet_plus_plus.blocks import *


def ResUNetPlusPlus(input_shape, classes: int, filters_root: int = 64, depth: int = 3):
    input = Input(shape=input_shape)

    layer = input

    # ENCODER
    encoder_blocks = []

    filters = filters_root
    branch = Conv2D(filters=filters, kernel_size=3, strides=1, padding="same", use_bias=False)(layer)
    branch = BatchNormalization()(branch)
    branch = ReLU()(branch)
    branch = Conv2D(filters=filters, kernel_size=3, strides=1, padding="same", use_bias=True)(branch)
    layer = Add()([branch, layer])
    encoder_blocks.append(layer)

    layer = SqueezeAndExcitationBlock(filters)(layer)

    for _ in range(depth - 1):
        filters *= 2
        layer = ResBlock(filters, strides=2)(layer)
        encoder_blocks.append(layer)

        layer = SqueezeAndExcitationBlock(filters)(layer)

    # BRIDGE
    # filters *= 2
    layer = AtrousSpatialPyramidPooling(filters, dilation_rates=[2, 4, 8])(layer)

    # DECODER
    for i in range(2, depth + 1):
        filters //= 2
        skip_block_connection = encoder_blocks[-i]

        conv_skip = Conv2D(filters=filters, kernel_size=3, strides=2, padding="same", use_bias=False)(skip_block_connection)
        conv_skip = BatchNormalization()(conv_skip)

        layer = Concatenate()([layer, conv_skip])
        layer = SelfAttentionBlock(ratio=8)(layer)
        layer = UpSampling2D()(layer)
        layer = Concatenate()([layer, skip_block_connection])
        layer = ResBlock(filters, strides=1)(layer)

    layer = AtrousSpatialPyramidPooling(filters, dilation_rates=[2, 4, 8])(layer)
    layer = Conv2D(filters=classes, kernel_size=1, strides=1, padding="same")(layer)
    layer = Softmax()(layer)

    model = tf.keras.Model(inputs=input, outputs=layer)
    return model
