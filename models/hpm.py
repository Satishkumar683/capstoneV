"""
Horizontal Pyramid Mapping (HPM) for Human Gait Recognition
Partitions feature maps into multi-scale horizontal stripes to capture
fine-grained anatomical part movement (head, torso, legs, feet) along the vertical axis.
"""

import tensorflow as tf
from tensorflow.keras import layers

def horizontal_pyramid_mapping(feature_map, num_parts: int = 6):
    """
    Slices a 4D feature map (B, H, W, C) along the height dimension into num_parts horizontal stripes.
    Applies both Average Pooling and Max Pooling per stripe and concatenates the resulting vectors.
    """
    feats = []
    h = tf.shape(feature_map)[1]

    for i in range(num_parts):
        start = tf.cast(i * h // num_parts, tf.int32)
        end = tf.cast((i + 1) * h // num_parts, tf.int32)
        part = feature_map[:, start:end, :, :]

        # 1D pooled features across height and width for this horizontal strip
        avg_p = tf.reduce_mean(part, axis=[1, 2])
        max_p = tf.reduce_max(part, axis=[1, 2])

        feats.append(avg_p)
        feats.append(max_p)

    return tf.concat(feats, axis=-1)
