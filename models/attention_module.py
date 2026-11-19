"""
Convolutional Block Attention Module (CBAM) for Human Gait Recognition
Includes:
  1. Channel Attention: Focuses on 'what' feature maps carry identity traits
  2. Spatial Attention: Focuses on 'where' motion is salient (limb kinematics)
"""

import tensorflow as tf
from tensorflow.keras import layers

def channel_attention(input_feature, ratio: int = 8):
    """
    Computes channel attention map using both Global Average and Global Max Pooling.
    """
    channel = input_feature.shape[-1]
    
    # Shared MLP bottleneck layers
    dense_1 = layers.Dense(channel // ratio, activation="relu", kernel_initializer="he_normal")
    dense_2 = layers.Dense(channel, kernel_initializer="he_normal")

    # Global Average Pooling branch
    avg_pool = layers.GlobalAveragePooling2D()(input_feature)
    avg_pool = layers.Reshape((1, 1, channel))(avg_pool)
    avg_out = dense_2(dense_1(avg_pool))

    # Global Max Pooling branch
    max_pool = layers.GlobalMaxPooling2D()(input_feature)
    max_pool = layers.Reshape((1, 1, channel))(max_pool)
    max_out = dense_2(dense_1(max_pool))

    # Combine and calibrate
    scale = layers.Activation("sigmoid")(layers.Add()([avg_out, max_out]))
    return layers.Multiply()([input_feature, scale])


def spatial_attention(input_feature):
    """
    Computes spatial attention map using channel-wise pooling and a 7x7 convolution.
    """
    avg_pool = layers.Lambda(lambda x: tf.reduce_mean(x, axis=3, keepdims=True))(input_feature)
    max_pool = layers.Lambda(lambda x: tf.reduce_max(x, axis=3, keepdims=True))(input_feature)
    concat = layers.Concatenate(axis=3)([avg_pool, max_pool])

    scale = layers.Conv2D(
        filters=1,
        kernel_size=7,
        padding="same",
        activation="sigmoid",
        kernel_initializer="he_normal",
        use_bias=False
    )(concat)
    return layers.Multiply()([input_feature, scale])


def cbam_block(input_feature, ratio: int = 8):
    """
    Sequential combination of Channel Attention followed by Spatial Attention.
    """
    feat = channel_attention(input_feature, ratio=ratio)
    feat = spatial_attention(feat)
    return feat
