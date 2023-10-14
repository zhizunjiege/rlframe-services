from typing import List, Union

from keras import layers, models
import tensorflow as tf


class MLPModel(models.Model):

    def __init__(
        self,
        name: str,
        trainable: bool,
        hidden_sizes: List[int],
        hidden_activation: str,
        output_sizes: Union[int, List[int]],
        output_activation: Union[str, List[str]],
    ):
        super().__init__(name=name)

        self.hiddens = [layers.Dense(size, hidden_activation, trainable=trainable) for size in hidden_sizes]
        if isinstance(output_sizes, list):
            if not isinstance(output_activation, list):
                output_activation = [output_activation] * len(output_sizes)
            self.outputs = [
                layers.Dense(size, acti, trainable=trainable) for size, acti in zip(output_sizes, output_activation)
            ]
        else:
            self.outputs = layers.Dense(output_sizes, output_activation, trainable=trainable)

    @tf.function
    def call(
        self,
        inputs: Union[tf.Tensor, List[tf.Tensor]],
        training: bool = None,
    ) -> Union[tf.Tensor, List[tf.Tensor]]:
        if isinstance(inputs, list):
            inputs = tf.concat(inputs, axis=1)
        x = inputs
        for layer in self.hiddens:
            x = layer(x, training=training)
        if isinstance(self.outputs, list):
            return [output(x, training=training) for output in self.outputs]
        else:
            return self.outputs(x, training=training)
