import json
import pickle
import queue
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
        with open('tests/examples/agent/hypers.json', 'r') as f1, \
             open('tests/examples/agent/states_inputs_func.py', 'r') as f2, \
             open('tests/examples/agent/outputs_actions_func.py', 'r') as f3, \
             open('tests/examples/agent/reward_func.py', 'r') as f4:
            req.training = True
            hypers = json.load(f1)
            req.type = hypers['type']
            req.hypers = json.dumps(hypers['hypers'])
            req.sifunc = f2.read()
            req.oafunc = f3.read()
            req.rewfunc = f4.read()
        self.stub.SetAgentConfig(req)

        res = self.stub.QueryService(types_pb2.CommonRequest())
        self.assertEqual(res.state, types_pb2.ServiceState.State.INITED)

        res = self.stub.GetAgentConfig(types_pb2.CommonRequest())
        self.assertEqual(res.type, hypers['type'])

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
        self.assertEqual(buffer['data']['acts_buf'].shape, (0, 1))

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
                                'double_value': 0.0
                            },
                            'latitude': {
                                'double_value': 0.0
                            },
                            'altitude': {
                                'double_value': 0.0
                            },
                            'speed': {
                                'double_value': 0.0
                            },
                            'azimuth': {
                                'double_value': 0.0
                            },
                        }
                    }],
                },
                'example_sub': {
                    'entities': [{
                        'params': {
                            'longitude': {
                                'double_value': 0.0
                            },
                            'latitude': {
                                'double_value': 0.0
                            },
                            'altitude': {
                                'double_value': 0.0
                            },
                            'speed': {
                                'double_value': 0.0
                            },
                            'azimuth': {
                                'double_value': 0.0
                            },
                        }
                    }],
                }
            },
            'terminated': False,
            'truncated': False,
        }
        req = json_format.ParseDict(info, types_pb2.SimState())

        req_count = 5000
        req_queue = queue.SimpleQueue()
        req_queue.put(req)
        req_count -= 1
        t1 = time.time()
        for res in self.stub.GetAction(iter(req_queue.get, None)):
            res = json_format.MessageToDict(res, preserving_proto_field_name=True)
            before = time.perf_counter_ns()
            while time.perf_counter_ns() - before < 10000000:
                pass
            if req_count > 0:
                req_queue.put(req)
            else:
                req_queue.put(None)
            req_count -= 1
        t2 = time.time()
        print(f'Time cost: {t2 - t1} s, FPS: {5000 / (t2 - t1)}')

        azimuth = res['actions']['example_uav']['entities'][0]['params']['azimuth']['double_value']
        print(f'Azimuth: {azimuth}')
        self.assertIsInstance(azimuth, float)
