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

        req = simenv_pb2.SimenvConfig()
        req.type = 'CQSim'
        req.args = json.dumps({'engine_addr': 'localhost:50041'})
        res = self.stub.SetSimenvConfig(req)
        self.assertEqual(res.code, 0)

        res = self.stub.QueryService(types_pb2.CommonRequest())
        self.assertEqual(res.state, types_pb2.ServiceState.State.INITED)

        res = self.stub.GetSimenvConfig(types_pb2.CommonRequest())
        self.assertEqual(res.type, 'CQSim')

    def test_02_simcontrol(self):
        cmd = simenv_pb2.SimCmd()

        cmd.type = simenv_pb2.SimCmd.Type.INIT
        cmd.params = json.dumps({
            'exp_design_id': 28,
            'sim_start_time': int(time.time()),
            'sim_duration': 30,
            'time_step': 1000,
            'speed_ratio': 10,
        })
        self.stub.SimControl(cmd)

        cmd.type = simenv_pb2.SimCmd.Type.START
        cmd.params = json.dumps({})
        self.stub.SimControl(cmd)
        time.sleep(3)

        cmd.type = simenv_pb2.SimCmd.Type.PAUSE
        self.stub.SimControl(cmd)
        time.sleep(1)

        cmd.type = simenv_pb2.SimCmd.Type.PARAM
        cmd.params = json.dumps({'speed_ratio': 100})
        self.stub.SimControl(cmd)

        cmd.type = simenv_pb2.SimCmd.Type.STEP
        self.stub.SimControl(cmd)

        cmd.type = simenv_pb2.SimCmd.Type.RESUME
        self.stub.SimControl(cmd)
        time.sleep(3)

        print(self.stub.SimMonitor(types_pb2.CommonRequest()))

        cmd.type = simenv_pb2.SimCmd.Type.DONE
        self.stub.SimControl(cmd)
        time.sleep(10)

        print(self.stub.SimMonitor(types_pb2.CommonRequest()))

        cmd.type = simenv_pb2.SimCmd.Type.STOP
        self.stub.SimControl(cmd)

    def test_03_simmonitor(self):
        print(self.stub.SimMonitor(types_pb2.CommonRequest()))
