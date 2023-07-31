import json
import pickle
import queue
import random
import time
import unittest

from google.protobuf import json_format
import grpc

from protos import agent_pb2
from protos import agent_pb2_grpc
from protos import types_pb2


class AgentServiceTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.channel = grpc.insecure_channel('localhost:10002')
        cls.stub = agent_pb2_grpc.AgentStub(channel=cls.channel)

    @classmethod
    def tearDownClass(cls):
        cls.stub.ResetService(types_pb2.CommonRequest())
        cls.channel.close()

    def test_00_queryservice(self):
        res = self.stub.QueryService(types_pb2.CommonRequest())
        self.assertEqual(res.state, types_pb2.ServiceState.State.UNINITED)

    def test_01_agentconfig(self):
        try:
            res = self.stub.GetAgentConfig(types_pb2.CommonRequest())
        except grpc.RpcError as e:
            self.assertEqual(e.code(), grpc.StatusCode.FAILED_PRECONDITION)

        req = agent_pb2.AgentConfig()
        with open('tests/examples/agent/configs.json', 'r') as f1, \
             open('tests/examples/agent/states_inputs_func.py', 'r') as f2, \
             open('tests/examples/agent/outputs_actions_func.py', 'r') as f3, \
             open('tests/examples/agent/reward_func.py', 'r') as f4:
            configs = json.load(f1)
            req.training = configs['training']
            req.name = configs['name']
            req.hypers = json.dumps(configs['hypers'])
            req.sifunc = f2.read()
            req.oafunc = f3.read()
            req.rewfunc = f4.read()
            for hook in configs['hooks']:
                pointer = req.hooks.add()
                pointer.name = hook['name']
                pointer.args = json.dumps(hook['args'])
        self.stub.SetAgentConfig(req)

        res = self.stub.QueryService(types_pb2.CommonRequest())
        self.assertEqual(res.state, types_pb2.ServiceState.State.INITED)

        res = self.stub.GetAgentConfig(types_pb2.CommonRequest())
        self.assertEqual(res.name, 'DQN')

    def test_02_agentmode(self):
        res = self.stub.GetAgentMode(types_pb2.CommonRequest())
        self.assertTrue(res.training)

        self.stub.SetAgentMode(agent_pb2.AgentMode(training=True))

    def test_03_modelweights(self):
        res = self.stub.GetModelWeights(types_pb2.CommonRequest())
        weights = pickle.loads(res.weights)
        self.assertTrue('online' in weights)
        self.assertTrue('target' in weights)

        self.stub.SetModelWeights(agent_pb2.ModelWeights(weights=res.weights))

    def test_04_modelbuffer(self):
        res = self.stub.GetModelBuffer(types_pb2.CommonRequest())
        buffer = pickle.loads(res.buffer)
        self.assertEqual(buffer['size'], 0)
        self.assertEqual(buffer['acts_buf'].shape, (0, 1))

        self.stub.SetModelBuffer(agent_pb2.ModelBuffer(buffer=res.buffer))

    def test_05_modelstatus(self):
        res = self.stub.GetModelStatus(types_pb2.CommonRequest())
        status = json.loads(res.status)
        self.assertEqual(status['react_steps'], 0)
        self.assertEqual(status['train_steps'], 0)

        self.stub.SetModelStatus(agent_pb2.ModelStatus(status=res.status))

    def test_06_getaction(self):
        info = {
            'states': {
                'example_uav': {
                    'entities': [{
                        'params': {
                            'longitude': {
                                'vdouble': 0.0
                            },
                            'latitude': {
                                'vdouble': 0.0
                            },
                            'altitude': {
                                'vdouble': 0.0
                            },
                            'speed': {
                                'vdouble': 0.0
                            },
                            'azimuth': {
                                'vdouble': 0.0
                            },
                        }
                    }],
                },
                'example_sub': {
                    'entities': [{
                        'params': {
                            'longitude': {
                                'vdouble': 0.0
                            },
                            'latitude': {
                                'vdouble': 0.0
                            },
                            'altitude': {
                                'vdouble': 0.0
                            },
                            'speed': {
                                'vdouble': 0.0
                            },
                            'azimuth': {
                                'vdouble': 0.0
                            },
                        }
                    }],
                }
            },
            'terminated': False,
            'truncated': False,
            'reward': 0.0,
        }
        req = json_format.ParseDict(info, types_pb2.SimState())

        t1 = time.time()
        for _ in range(100):
            req_count = 100
            req_queue = queue.SimpleQueue()
            req_queue.put(req)
            req.truncated = False
            req.reward = random.random()
            for res in self.stub.GetAction(iter(req_queue.get, None)):
                res = json_format.MessageToDict(res, preserving_proto_field_name=True)
                before = time.perf_counter_ns()
                while time.perf_counter_ns() - before < 0.01 * 1e9:
                    pass
                if req_count > 0:
                    if req_count == 1:
                        req.truncated = True
                    req_queue.put(req)
                else:
                    req_queue.put(None)
                req_count -= 1
        t2 = time.time()
        print(f'Time cost: {t2 - t1} s, FPS: {100 * 100 / (t2 - t1)}')

        azimuth = res['actions']['example_uav']['entities'][0]['params']['azimuth']['vdouble']
        print(f'Azimuth: {azimuth}')
        self.assertIsInstance(azimuth, float)
