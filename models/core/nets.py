from typing import List, Union

from keras.layers import Dense, Input
from keras.models import Model
import tensorflow as tf


class MLPModel(Model):

    def __init__(
        self,
        name: str,
        trainable: bool,
        input_sizes: Union[int, List[int]],
        hidden_sizes: List[int],
        hidden_activation: str,
        output_sizes: Union[int, List[int]],
        output_activation: Union[str, List[str]],
    ):
        if isinstance(input_sizes, list):
            inputs = [Input(shape=(size,)) for size in input_sizes]
            hiddens = tf.concat(inputs, axis=1)
        else:
            inputs = Input(shape=(input_sizes,))
            hiddens = inputs
        for size in hidden_sizes:
            hiddens = Dense(size, hidden_activation, trainable=trainable)(hiddens)
        if isinstance(output_sizes, list):
            if not isinstance(output_activation, list):
                output_activation = [output_activation] * len(output_sizes)
            outputs = [Dense(size, acti, trainable=trainable)(hiddens) for size, acti in zip(output_sizes, output_activation)]
        else:
            outputs = Dense(output_sizes, output_activation, trainable=trainable)(hiddens)
        super().__init__(inputs=inputs, outputs=outputs, name=name)
