"""
ResNet-50 + CBAM + Horizontal Pyramid Mapping (HPM) Architecture
"""

from typing import Tuple
import tensorflow as tf
from tensorflow.keras import layers, models, applications, regularizers
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.losses import CategoricalCrossentropy

from .attention_module import cbam_block
from .hpm import horizontal_pyramid_mapping

def build_crossview_gait_model(
    num_classes: int,
    image_size: Tuple[int, int] = (224, 224),
    num_parts: int = 6,
    dropout_rate: float = 0.5,
    initial_lr: float = 1e-4,
    weight_decay: float = 1e-4,
    total_steps: int = 1000
) -> models.Model:
    """
    Constructs the end-to-end Cross-View Gait Recognition model.
    """
    inp = layers.Input((*image_size, 3), name="input_image")

    # Light data augmentation (horizontal flip preserving upright gait posture)
    aug = layers.RandomFlip("horizontal", name="gait_flip_aug")(inp)

    # ResNet-50 Backbone
    base = applications.ResNet50(weights="imagenet", include_top=False, input_tensor=aug)

    # Freeze earlier stages, fine-tune conv5 and BatchNorm layers
    for layer in base.layers:
        layer.trainable = False
    for layer in base.layers:
        if layer.name.startswith("conv5_") or isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

    # Attention Guidance via CBAM
    attn_features = cbam_block(base.output, ratio=8)

    # Horizontal Pyramid Mapping (HPM)
    hpm_features = layers.Lambda(
        lambda f: horizontal_pyramid_mapping(f, num_parts=num_parts),
        name="hpm_layer"
    )(attn_features)

    # Classification & Embedding Head
    x = layers.BatchNormalization(name="hpm_bn")(hpm_features)
    x = layers.Dense(512, activation="relu", kernel_regularizer=regularizers.l2(5e-4), name="fc_embedding")(x)
    x = layers.Dropout(dropout_rate, name="embedding_dropout")(x)
    out = layers.Dense(num_classes, activation="softmax", name="classification_output")(x)

    # Optimizer with Cosine Decay Schedule
    lr_schedule = CosineDecayRestarts(
        initial_learning_rate=initial_lr,
        first_decay_steps=total_steps,
        t_mul=2.0,
        m_mul=1.0,
        alpha=0.0
    )
    optimizer = AdamW(learning_rate=lr_schedule, weight_decay=weight_decay)

    model = models.Model(inputs=inp, outputs=out, name="CrossView_Gait_ResNet50_CBAM_HPM")
    model.compile(
        optimizer=optimizer,
        loss=CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"]
    )
    return model
