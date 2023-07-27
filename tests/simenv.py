import json
import time
import unittest

import grpc

from protos import simenv_pb2
from protos import simenv_pb2_grpc
from protos import types_pb2


class SimenvServiceTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with open('tests/examples/simenv/service.json', 'r') as f:
            service = json.load(f)
        cls.channel = grpc.insecure_channel(f'{service["host"]}:{service["port"]}')
        cls.stub = simenv_pb2_grpc.SimenvStub(channel=cls.channel)

    @classmethod
    def tearDownClass(cls):
        cls.stub.ResetService(types_pb2.CommonRequest())
        cls.channel.close()

    def test_00_queryservice(self):
        res = self.stub.QueryService(types_pb2.CommonRequest())
        self.assertEqual(res.state, types_pb2.ServiceState.State.UNINITED)

    def test_01_simenvconfig(self):
        try:
            res = self.stub.GetSimenvConfig(types_pb2.CommonRequest())
        except grpc.RpcError as e:
            self.assertEqual(e.code(), grpc.StatusCode.FAILED_PRECONDITION)

        req = simenv_pb2.SimenvConfig()
        with open('tests/examples/simenv/configs.json', 'r') as f1, \
             open('tests/examples/simenv/sim_term_func.cpp', 'r') as f2:
            configs = json.load(f1)
            req.name = configs['name']
            configs['args']['proxy']['sim_term_func'] = f2.read()
            req.args = json.dumps(configs['args'])
        self.stub.SetSimenvConfig(req)

        res = self.stub.QueryService(types_pb2.CommonRequest())
        self.assertEqual(res.state, types_pb2.ServiceState.State.INITED)

        res = self.stub.GetSimenvConfig(types_pb2.CommonRequest())
        self.assertEqual(res.name, 'CQSIM')

    def test_02_simcontrol(self):
        cmd = simenv_pb2.SimCmd()

        cmd.type = 'init'
        self.stub.SimControl(cmd)

        cmd.type = 'start'
        self.stub.SimControl(cmd)
        time.sleep(3)

        cmd.type = 'pause'
        self.stub.SimControl(cmd)

        cmd.type = 'param'
        cmd.params = json.dumps({'speed_ratio': 100})
        self.stub.SimControl(cmd)

        cmd.type = 'step'
        self.stub.SimControl(cmd)

        cmd.type = 'resume'
        self.stub.SimControl(cmd)
        time.sleep(3)

        cmd.type = 'episode'
        self.stub.SimControl(cmd)
        time.sleep(10)

        cmd.type = 'stop'
        self.stub.SimControl(cmd)

    def test_03_simmonitor(self):
        res = self.stub.SimMonitor(types_pb2.CommonRequest())
        print('state: ', res.state)
        print('data: ', json.loads(res.data))
        print('logs: ', json.loads(res.logs))
