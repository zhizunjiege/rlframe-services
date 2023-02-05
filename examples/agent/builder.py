from typing import Dict

import tensorflow as tf


def func() -> Dict[str, tf.keras.Model]:
    """Build networks for the model."""
    inputs = tf.keras.Input(shape=(12,), name='inputs')
    x = tf.keras.layers.Dense(256, activation='relu')(inputs)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    outputs = tf.keras.layers.Dense(8, activation='linear')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return {
        'online': model,
        'target': tf.keras.models.clone_model(model),
    }
