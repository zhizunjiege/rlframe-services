import unittest

import numpy as np
import tensorflow as tf

from models.core.nets import MLPModel


class MLPModelTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        ...

    @classmethod
    def tearDownClass(cls):
        ...

    def test_00_init(self):
        model = MLPModel(
            name='test0',
            trainable=True,
            input_sizes=12,
            hidden_sizes=[256, 256],
            hidden_activation='relu',
            output_sizes=8,
            output_activation='softmax',
        )
        self.assertTrue(model.trainable)
        self.assertEqual(model.name, 'test0')

    def test_01_call_case1(self):
        model = MLPModel(
            name='test1',
            trainable=True,
            input_sizes=12,
            hidden_sizes=[256, 256],
            hidden_activation='relu',
            output_sizes=8,
            output_activation='tanh',
        )
        inputs = np.random.random((64, 12))
        outputs = model(inputs, training=True)
        self.assertIsInstance(outputs, tf.Tensor)
        self.assertEqual(outputs.shape, (64, 8))

    def test_02_call_case2(self):
        model = MLPModel(
            name='test2',
            trainable=False,
            input_sizes=[100, 8],
            hidden_sizes=[64],
            hidden_activation='relu',
            output_sizes=1,
            output_activation='linear',
        )
        inputs1 = np.random.random((1, 100))
        inputs2 = np.random.random((1, 8))
        outputs = model([inputs1, inputs2], training=False)
        self.assertIsInstance(outputs, tf.Tensor)
        self.assertEqual(outputs.shape, (1, 1))

    def test_03_weights(self):
        model = MLPModel(
            name='test3',
            trainable=False,
            input_sizes=100,
            hidden_sizes=[64],
            hidden_activation='relu',
            output_sizes=8,
            output_activation='softmax',
        )
        weights = model.get_weights()
        self.assertEqual(len(weights), 4)

        inputs = np.random.random((1, 100))
        model(inputs, training=False)

        model.set_weights(weights)
