import json
import pickle
import time
import unittest

import grpc
import numpy as np

from protos import simenv_pb2
from protos import simenv_pb2_grpc
from protos import types_pb2


class SimenvServiceTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.channel = grpc.insecure_channel("localhost:10001")
        cls.stub = simenv_pb2_grpc.SimenvStub(channel=cls.channel)

    @classmethod
    def tearDownClass(cls):
        cls.stub.ResetService(types_pb2.CommonRequest())
        cls.channel.close()
        cls.stub = None

    def test_00_queryservice(self):
        res = self.stub.QueryService(types_pb2.CommonRequest())
        self.assertEqual(res.state, types_pb2.ServiceState.State.UNINITED)

    def test_01_simenvconfig(self):
        try:
            res = self.stub.GetSimenvConfig(types_pb2.CommonRequest())
        except grpc.RpcError as e:
            self.assertEqual(e.code(), grpc.StatusCode.FAILED_PRECONDITION)

        req = simenv_pb2.SimenvConfig()
        with open('examples/simenv/states_inputs_func.py', 'r') as f1, \
             open('examples/simenv/outputs_actions_func.py', 'r') as f2, \
             open('examples/simenv/reward_func.py', 'r') as f3, \
             open('examples/simenv/hypers_dqn.json', 'r') as f4, \
             open('examples/simenv/builder_dqn.py', 'r') as f5, \
             open('examples/simenv/structures_dqn.json', 'r') as f6:
            req.training = False
            req.states_inputs_func = f1.read()
            req.outputs_actions_func = f2.read()
            req.reward_func = f3.read()
            req.type = 'DQN'
            req.hypers = f4.read()
            req.builder = f5.read()
            req.structures = f6.read()

        res = self.stub.SetSimenvConfig(req)
        self.assertEqual(res.code, 0)

        res = self.stub.QueryService(types_pb2.CommonRequest())
        self.assertEqual(res.state, types_pb2.ServiceState.State.INITED)

        res = self.stub.GetSimenvConfig(types_pb2.CommonRequest())
        self.assertEqual(res.type, 'DQN')

        self.stub.ResetService(types_pb2.CommonRequest())

        req.training = True
        req.builder = ''
        res = self.stub.SetSimenvConfig(req)
        self.assertEqual(res.code, 0)

    def test_02_simenvmode(self):
        res = self.stub.GetSimenvMode(types_pb2.CommonRequest())
        self.assertTrue(res.training)

        res = self.stub.SetSimenvMode(simenv_pb2.SimenvMode(training=True))
        self.assertEqual(res.code, 0)

    def test_03_modelweights(self):
        res = self.stub.GetModelWeights(types_pb2.CommonRequest())
        weights = pickle.loads(res.weights)
        self.assertTrue('online' in weights)
        self.assertTrue('target' in weights)

        res = self.stub.SetModelWeights(simenv_pb2.ModelWeights(weights=res.weights))
        self.assertEqual(res.code, 0)

    def test_04_modelbuffer(self):
        res = self.stub.GetModelBuffer(types_pb2.CommonRequest())
        buffer = pickle.loads(res.buffer)
        self.assertEqual(buffer['size'], 0)
        self.assertEqual(buffer['data']['acts_buf'].shape, (0, 1))

        res = self.stub.SetModelBuffer(simenv_pb2.ModelBuffer(buffer=res.buffer))
        self.assertEqual(res.code, 0)

    def test_05_modelstatus(self):
        res = self.stub.GetModelStatus(types_pb2.CommonRequest())
        status = json.loads(res.status)
        self.assertEqual(status['react_steps'], 0)
        self.assertEqual(status['train_steps'], 0)

        res = self.stub.SetModelStatus(simenv_pb2.ModelStatus(status=res.status))
        self.assertEqual(res.code, 0)

    def test_06_getaction(self):
        info = {
            'states': {
                'model1': [{
                    'input1': np.random.rand(20).tolist(),
                }],
            },
            'done': False,
        }
        req = types_pb2.JsonString(json=json.dumps(info))

        t1 = time.time()
        for _ in range(5000):
            res = self.stub.GetAction(req)
            action = json.loads(res.json)['actions']
        t2 = time.time()
        print()
        print(f'Time cost: {t2 - t1} s, FPS: {5000 / (t2 - t1)}')
        self.assertEqual(type(action['model1']['output1']), int)
