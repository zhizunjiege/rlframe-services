import time
import unittest

import numpy as np
import tensorflow as tf

from models.ppo import MLPDiscretePolicy, MLPContinuousPolicy, MLPMultiDiscretePolicy, MLPHybridPolicy


class MLPDiscretePolicyTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = MLPDiscretePolicy(
            name='test_discrete',
            trainable=True,
            hidden_layers=[256, 256],
            act_dim=8,
        )

    @classmethod
    def tearDownClass(cls):
        cls.model = None

    def test_00_call_case1(self):
        obs = np.random.random((1, 12))
        t1 = time.time()
        for _ in range(1000):
            act, logp = self.model(obs, training=True)
        t2 = time.time()
        print(f'1x12x256x256x8 nn 1000 call time in train: {t2 - t1:.2f}s')
        self.assertIsInstance(act, tf.Tensor)
        self.assertEqual(act.shape, (1,))
        self.assertEqual(act.dtype, tf.int32)
        self.assertIsInstance(logp, tf.Tensor)
        self.assertEqual(logp.shape, (1,))
        self.assertEqual(logp.dtype, tf.float32)
        self.assertGreater(tf.reduce_min(logp), -1e2)
        self.assertLess(tf.reduce_max(logp), -1e-8)

    def test_00_call_case2(self):
        obs = np.random.random((64, 12))
        t1 = time.time()
        for _ in range(1000):
            act, logp = self.model(obs, training=False)
        t2 = time.time()
        print(f'64x12x256x256x8 nn 1000 call time in infer: {t2 - t1:.2f}s')
        self.assertIsInstance(act, tf.Tensor)
        self.assertEqual(act.shape, (64,))
        self.assertEqual(act.dtype, tf.int32)
        self.assertIsNone(logp)

    def test_01_logp_case1(self):
        obs = np.random.random((64, 12))
        act = np.random.randint(0, 8, (64, 1))
        t1 = time.time()
        for _ in range(1000):
            logp = self.model.logp(obs, act)
        t2 = time.time()
        print(f'64x12x256x256x8 nn 1000 logp time: {t2 - t1:.2f}s')
        self.assertIsInstance(logp, tf.Tensor)
        self.assertEqual(logp.shape, (64,))
        self.assertEqual(logp.dtype, tf.float32)
        self.assertGreater(tf.reduce_min(logp), -1e2)
        self.assertLess(tf.reduce_max(logp), -1e-8)

    @unittest.expectedFailure
    def test_01_logp_case2(self):
        obs = np.random.random((64, 12))
        act = np.random.randint(0, 8, (64,))
        logp = self.model.logp(obs, act)
        self.assertIsInstance(logp, tf.Tensor)
        self.assertEqual(logp.shape, (64,))
        self.assertEqual(logp.dtype, tf.float32)
        self.assertGreater(tf.reduce_min(logp), -1e2)
        self.assertLess(tf.reduce_max(logp), -1e-8)


class MLPContinuousPolicyTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = MLPContinuousPolicy(
            name='test_continuous',
            trainable=True,
            hidden_layers=[256, 256],
            act_dim=1,
        )

    @classmethod
    def tearDownClass(cls):
        cls.model = None

    def test_00_call_case1(self):
        obs = np.random.random((1, 12))
        t1 = time.time()
        for _ in range(1000):
            act, logp = self.model(obs, training=True)
        t2 = time.time()
        print(f'1x12x256x256x1 nn 1000 call time in train: {t2 - t1:.2f}s')
        self.assertIsInstance(act, tf.Tensor)
        self.assertEqual(act.shape, (1, 1))
        self.assertEqual(act.dtype, tf.float32)
        self.assertIsInstance(logp, tf.Tensor)
        self.assertEqual(logp.shape, (1,))
        self.assertEqual(logp.dtype, tf.float32)
        self.assertGreater(tf.reduce_min(logp), -1e2)
        self.assertLess(tf.reduce_max(logp), -1e-8)

    def test_00_call_case2(self):
        obs = np.random.random((64, 12))
        t1 = time.time()
        for _ in range(1000):
            act, logp = self.model(obs, training=False)
        t2 = time.time()
        print(f'64x12x256x256x1 nn 1000 call time in infer: {t2 - t1:.2f}s')
        self.assertIsInstance(act, tf.Tensor)
        self.assertEqual(act.shape, (64, 1))
        self.assertEqual(act.dtype, tf.float32)
        self.assertIsNone(logp)

    def test_01_logp_case1(self):
        obs = np.random.random((64, 12))
        act = np.random.random((64, 1)).astype(np.float32)
        t1 = time.time()
        for _ in range(1000):
            logp = self.model.logp(obs, act)
        t2 = time.time()
        print(f'64x12x256x256x1 nn 1000 logp time: {t2 - t1:.2f}s')
        self.assertIsInstance(logp, tf.Tensor)
        self.assertEqual(logp.shape, (64,))
        self.assertEqual(logp.dtype, tf.float32)
        self.assertGreater(tf.reduce_min(logp), -1e2)
        self.assertLess(tf.reduce_max(logp), -1e-8)

    @unittest.expectedFailure
    def test_01_logp_case2(self):
        obs = np.random.random((64, 12))
        act = np.random.random((64, )).astype(np.int32)
        logp = self.model.logp(obs, act)
        self.assertIsInstance(logp, tf.Tensor)
        self.assertEqual(logp.shape, (64,))
        self.assertEqual(logp.dtype, tf.float32)
        self.assertGreater(tf.reduce_min(logp), -1e2)
        self.assertLess(tf.reduce_max(logp), -1e-8)


class MLPMultiDiscretePolicyTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = MLPMultiDiscretePolicy(
            name='test_multi_discrete',
            trainable=True,
            hidden_layers=[256, 256],
            act_dims=[2, 4, 8],
        )

    @classmethod
    def tearDownClass(cls):
        cls.model = None

    def test_00_call_case1(self):
        obs = np.random.random((1, 12))
        t1 = time.time()
        for _ in range(1000):
            act, logp = self.model(obs, training=True)
        t2 = time.time()
        print(f'1x12x256x256x(2+4) nn 1000 call time in train: {t2 - t1:.2f}s')
        self.assertIsInstance(act, tf.Tensor)
        self.assertEqual(act.shape, (1, 3))
        self.assertEqual(act.dtype, tf.int32)
        self.assertIsInstance(logp, tf.Tensor)
        self.assertEqual(logp.shape, (1,))
        self.assertEqual(logp.dtype, tf.float32)
        self.assertGreater(tf.reduce_min(logp), -1e2)
        self.assertLess(tf.reduce_max(logp), -1e-8)

    def test_00_call_case2(self):
        obs = np.random.random((64, 12))
        t1 = time.time()
        for _ in range(1000):
            act, logp = self.model(obs, training=False)
        t2 = time.time()
        print(f'64x12x256x256x(2+4) nn 1000 call time in infer: {t2 - t1:.2f}s')
        self.assertIsInstance(act, tf.Tensor)
        self.assertEqual(act.shape, (64, 3))
        self.assertEqual(act.dtype, tf.int32)
        self.assertIsNone(logp)

    def test_01_logp_case1(self):
        obs = np.random.random((64, 12))
        act = np.random.randint(0, 2, (64, 3))
        t1 = time.time()
        for _ in range(1000):
            logp = self.model.logp(obs, act)
        t2 = time.time()
        print(f'64x12x256x256x(2+4) nn 1000 logp time: {t2 - t1:.2f}s')
        self.assertIsInstance(logp, tf.Tensor)
        self.assertEqual(logp.shape, (64,))
        self.assertEqual(logp.dtype, tf.float32)
        self.assertGreater(tf.reduce_min(logp), -1e2)
        self.assertLess(tf.reduce_max(logp), -1e-8)

    @unittest.expectedFailure
    def test_01_logp_case2(self):
        obs = np.random.random((64, 12))
        act = np.random.randint(0, 2, (64,))
        logp = self.model.logp(obs, act)
        self.assertIsInstance(logp, tf.Tensor)
        self.assertEqual(logp.shape, (64,))
        self.assertEqual(logp.dtype, tf.float32)
        self.assertGreater(tf.reduce_min(logp), -1e2)
        self.assertLess(tf.reduce_max(logp), -1e-8)


class MLPHybridPolicyTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = MLPHybridPolicy(
            name='test_hybrid',
            trainable=True,
            hidden_layers=[256, 256],
            act_dims=[
                [1, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 1, 1, 1],
            ],
        )

    @classmethod
    def tearDownClass(cls):
        cls.model = None

    def test_00_call_case1(self):
        obs = np.random.random((1, 12))
        t1 = time.time()
        for _ in range(1000):
            act, logp = self.model(obs, training=True)
        t2 = time.time()
        print(f'1x12x256x256x(3+1+0+3) nn 1000 call time in train: {t2 - t1:.2f}s')
        self.assertIsInstance(act, tf.Tensor)
        self.assertEqual(act.shape, (1, 5))
        self.assertEqual(act.dtype, tf.float32)
        self.assertIsInstance(logp, tf.Tensor)
        self.assertEqual(logp.shape, (1,))
        self.assertEqual(logp.dtype, tf.float32)
        self.assertGreater(tf.reduce_min(logp), -1e2)

    def test_00_call_case2(self):
        obs = np.random.random((64, 12))
        t1 = time.time()
        for _ in range(1000):
            act, logp = self.model(obs, training=False)
        t2 = time.time()
        print(f'64x12x256x256x(3+1+0+3) nn 1000 call time in infer: {t2 - t1:.2f}s')
        self.assertIsInstance(act, tf.Tensor)
        self.assertEqual(act.shape, (64, 5))
        self.assertEqual(act.dtype, tf.float32)
        self.assertIsNone(logp)

    def test_01_logp_case1(self):
        obs = np.random.random((64, 12))
        act_discrete = np.random.randint(0, 3, (64, 1)).astype(np.float32)
        act_continuous = np.random.random((64, 4)).astype(np.float32)
        act = np.concatenate([act_discrete, act_continuous], axis=1)
        t1 = time.time()
        for _ in range(1000):
            logp = self.model.logp(obs, act)
        t2 = time.time()
        print(f'64x12x256x256x(3+1+0+3) nn 1000 logp time: {t2 - t1:.2f}s')
        self.assertIsInstance(logp, tf.Tensor)
        self.assertEqual(logp.shape, (64,))
        self.assertEqual(logp.dtype, tf.float32)
        self.assertGreater(tf.reduce_min(logp), -1e2)

    @unittest.expectedFailure
    def test_01_logp_case2(self):
        obs = np.random.random((64, 12))
        act_discrete = np.random.randint(0, 3, (64,)).astype(np.float32)
        act_continuous = np.random.random((64, 4)).astype(np.float32)
        act = np.concatenate([act_discrete, act_continuous], axis=1)
        logp = self.model.logp(obs, act)
        self.assertIsInstance(logp, tf.Tensor)
        self.assertEqual(logp.shape, (64,))
        self.assertEqual(logp.dtype, tf.float32)
        self.assertGreater(tf.reduce_min(logp), -1e2)
