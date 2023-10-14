import unittest

import numpy as np
import tensorflow as tf

from models.core.models import MLPModel


class MLPModelTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        ...

    @classmethod
    def tearDownClass(cls):
        ...

    def test_00_init(self):
        model = MLPModel(
            name='test',
            trainable=True,
            hidden_sizes=[256, 256],
            hidden_activation='relu',
            output_sizes=8,
            output_activation='tanh',
        )
        self.assertTrue(model.trainable)
        self.assertEqual(model.name, 'test')
        self.assertEqual(len(model.layers), 3)

    def test_01_call_case1(self):
        model = MLPModel(
            name='test1',
            trainable=True,
            hidden_sizes=[256, 256],
            hidden_activation='relu',
            output_sizes=1,
            output_activation='tanh',
        )
        inputs = np.random.random((64, 12))
        outputs = model(inputs, training=True)
        self.assertIsInstance(outputs, tf.Tensor)
        self.assertEqual(outputs.shape, (64, 1))

    def test_02_call_case2(self):
        model = MLPModel(
            name='test2',
            trainable=False,
            hidden_sizes=[64],
            hidden_activation='relu',
            output_sizes=8,
            output_activation='softmax',
        )
        inputs = np.random.random((1, 100))
        outputs = model(inputs, training=False)
        self.assertIsInstance(outputs, tf.Tensor)
        self.assertEqual(outputs.shape, (1, 8))
