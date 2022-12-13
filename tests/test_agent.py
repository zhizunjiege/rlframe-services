import json
import pickle
import time
import unittest

import grpc
import numpy as np

from protos import agent_pb2
from protos import agent_pb2_grpc
from protos import types_pb2


class AgentServicerTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.channel = grpc.insecure_channel("localhost:50051")
        cls.stub = agent_pb2_grpc.AgentStub(cls.channel)

    @classmethod
    def tearDownClass(cls):
        cls.stub.ResetServer(types_pb2.CommonRequest())
        cls.channel.close()

    def test_00_resetserver(self):
        res = self.stub.ResetServer(types_pb2.CommonRequest())
        self.assertEqual(res.code, 0)

    def test_01_agentconfig(self):
        req = agent_pb2.AgentConfig()
        with open('examples/states_inputs_func.py', 'r') as f1, \
             open('examples/outputs_actions_func.py', 'r') as f2, \
             open('examples/reward_func.py', 'r') as f3, \
             open('examples/hypers.json', 'r') as f4, \
             open('examples/builder.py', 'r') as f5, \
             open('examples/structures.json', 'r') as f6:
            req.training = False
            req.states_inputs_func = f1.read()
            req.outputs_actions_func = f2.read()
            req.reward_func = f3.read()
            req.type = 'DQN'
            req.hypers = f4.read()
            req.builder = f5.read()
            req.structures = f6.read()

        res = self.stub.SetAgentConfig(req)
        self.assertEqual(res.code, 0)

        res = self.stub.GetAgentConfig(types_pb2.CommonRequest())
        self.assertEqual(res.type, 'DQN')

        self.stub.ResetServer(types_pb2.CommonRequest())

        req.training = True
        req.builder = ''
        res = self.stub.SetAgentConfig(req)
        self.assertEqual(res.code, 0)

    def test_02_agentmode(self):
        res = self.stub.SetAgentMode(agent_pb2.AgentMode(training=True))
        self.assertEqual(res.code, 0)

        res = self.stub.GetAgentMode(types_pb2.CommonRequest())
        self.assertTrue(res.training)

    def test_03_agentweight(self):
        res = self.stub.GetAgentWeight(types_pb2.CommonRequest())
        weights = pickle.loads(res.weights)
        self.assertTrue('online' in weights)
        self.assertTrue('target' in weights)

        res = self.stub.SetAgentWeight(agent_pb2.AgentWeight(weights=res.weights))
        self.assertEqual(res.code, 0)

    def test_04_agentbuffer(self):
        res = self.stub.GetAgentBuffer(types_pb2.CommonRequest())
        buffer = pickle.loads(res.buffer)
        self.assertEqual(buffer['size'], 0)
        self.assertEqual(buffer['data']['acts_buf'].shape, (0, 1))

        res = self.stub.SetAgentBuffer(agent_pb2.AgentBuffer(buffer=res.buffer))
        self.assertEqual(res.code, 0)

    def test_05_agentstatus(self):
        res = self.stub.GetAgentStatus(types_pb2.CommonRequest())
        status = json.loads(res.status)
        self.assertEqual(status['react_steps'], 0)
        self.assertEqual(status['train_steps'], 0)
        self.assertEqual(status['states_dim'], 20)
        self.assertEqual(status['actions_num'], 8)

        res = self.stub.SetAgentStatus(agent_pb2.AgentStatus(status=res.status))
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
