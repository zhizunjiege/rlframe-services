import json
import pickle
import time
import unittest

import grpc
import numpy as np

from protos import agent_pb2
from protos import agent_pb2_grpc
from protos import types_pb2


class AgentServiceTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.channel = grpc.insecure_channel("localhost:10001")
        cls.stub = agent_pb2_grpc.AgentStub(channel=cls.channel)

    @classmethod
    def tearDownClass(cls):
        cls.stub.ResetService(types_pb2.CommonRequest())
        cls.channel.close()
        cls.stub = None

    def test_00_queryservice(self):
        res = self.stub.QueryService(types_pb2.CommonRequest())
        self.assertEqual(res.state, types_pb2.ServiceState.State.UNINITED)

    def test_01_agentconfig(self):
        try:
            res = self.stub.GetAgentConfig(types_pb2.CommonRequest())
        except grpc.RpcError as e:
            self.assertEqual(e.code(), grpc.StatusCode.FAILED_PRECONDITION)

        req = agent_pb2.AgentConfig()
        with open('examples/agent/states_inputs_func.py', 'r') as f1, \
             open('examples/agent/outputs_actions_func.py', 'r') as f2, \
             open('examples/agent/reward_func.py', 'r') as f3, \
             open('examples/agent/hypers_dqn.json', 'r') as f4, \
             open('examples/agent/builder_dqn.py', 'r') as f5, \
             open('examples/agent/structures_dqn.json', 'r') as f6:
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

        res = self.stub.QueryService(types_pb2.CommonRequest())
        self.assertEqual(res.state, types_pb2.ServiceState.State.INITED)

        res = self.stub.GetAgentConfig(types_pb2.CommonRequest())
        self.assertEqual(res.type, 'DQN')

        self.stub.ResetService(types_pb2.CommonRequest())

        req.training = True
        req.builder = ''
        res = self.stub.SetAgentConfig(req)
        self.assertEqual(res.code, 0)

    def test_02_agentmode(self):
        res = self.stub.GetAgentMode(types_pb2.CommonRequest())
        self.assertTrue(res.training)

        res = self.stub.SetAgentMode(agent_pb2.AgentMode(training=True))
        self.assertEqual(res.code, 0)

    def test_03_modelweights(self):
        res = self.stub.GetModelWeights(types_pb2.CommonRequest())
        weights = pickle.loads(res.weights)
        self.assertTrue('online' in weights)
        self.assertTrue('target' in weights)

        res = self.stub.SetModelWeights(agent_pb2.ModelWeights(weights=res.weights))
        self.assertEqual(res.code, 0)

    def test_04_modelbuffer(self):
        res = self.stub.GetModelBuffer(types_pb2.CommonRequest())
        buffer = pickle.loads(res.buffer)
        self.assertEqual(buffer['size'], 0)
        self.assertEqual(buffer['data']['acts_buf'].shape, (0, 1))

        res = self.stub.SetModelBuffer(agent_pb2.ModelBuffer(buffer=res.buffer))
        self.assertEqual(res.code, 0)

    def test_05_modelstatus(self):
        res = self.stub.GetModelStatus(types_pb2.CommonRequest())
        status = json.loads(res.status)
        self.assertEqual(status['react_steps'], 0)
        self.assertEqual(status['train_steps'], 0)

        res = self.stub.SetModelStatus(agent_pb2.ModelStatus(status=res.status))
        self.assertEqual(res.code, 0)

    def test_06_getaction(self):
        info = {
            'states': {
                'model1': [{
                    'output1': np.random.rand(20).tolist(),
                }],
            },
            'terminated': False,
            'truncated': False,
        }
        req = types_pb2.JsonString(json=json.dumps(info))

        t1 = time.time()
        for _ in range(5000):
            res = self.stub.GetAction(req)
            action = json.loads(res.json)['actions']
        t2 = time.time()
        print()
        print(f'Time cost: {t2 - t1} s, FPS: {5000 / (t2 - t1)}')
        self.assertEqual(type(action['model1']['input1']), int)
