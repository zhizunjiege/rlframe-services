import tensorflow as tf


def func():
    """Build networks for the model."""
    inputs = tf.keras.Input(shape=(4,), name='inputs')
    x = tf.keras.layers.Dense(16, activation='relu')(inputs)
    outputs = tf.keras.layers.Dense(2, activation='linear')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return {
        'online': model,
        'target': tf.keras.models.clone_model(model),
    }
