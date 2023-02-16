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

        with open('examples/simenv/args.json', 'r') as f:
            args = f.read()

        req = simenv_pb2.SimenvConfig()
        req.type = args['type']
        req.args = json.dumps(args['args'])
        res = self.stub.SetSimenvConfig(req)
        self.assertEqual(res.code, 0)

        res = self.stub.QueryService(types_pb2.CommonRequest())
        self.assertEqual(res.state, types_pb2.ServiceState.State.INITED)

        res = self.stub.GetSimenvConfig(types_pb2.CommonRequest())
        self.assertEqual(res.type, args['type'])

    def test_02_simcontrol(self):
        cmd = simenv_pb2.SimCmd()

        with open('examples/simenv/configs.json', 'r') as f1, open('examples/simenv/sim_term_func.cc', 'r') as f2:
            sim_params = json.load(f1)
            sim_params['proxy']['sim_term_func'] = f2.read()

        cmd.type = 'init'
        cmd.params = json.dumps(sim_params)
        self.stub.SimControl(cmd)

        cmd.type = 'start'
        cmd.params = '{}'
        self.stub.SimControl(cmd)
        time.sleep(3)

        cmd.type = 'pause'
        cmd.params = '{}'
        self.stub.SimControl(cmd)

        cmd.type = 'param'
        cmd.params = json.dumps({'speed_ratio': 100})
        self.stub.SimControl(cmd)

        cmd.type = 'step'
        cmd.params = '{}'
        self.stub.SimControl(cmd)

        cmd.type = 'resume'
        cmd.params = '{}'
        self.stub.SimControl(cmd)
        time.sleep(3)

        cmd.type = 'episode'
        cmd.params = '{}'
        self.stub.SimControl(cmd)
        time.sleep(10)

        cmd.type = 'stop'
        cmd.params = '{}'
        self.stub.SimControl(cmd)

    def test_03_simmonitor(self):
        res = self.stub.SimMonitor(types_pb2.CommonRequest())
        print('state: ', res.state)
        print('data: ', json.loads(res.data))
        print('logs: ', json.loads(res.logs))
