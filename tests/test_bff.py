import json
import time
import unittest

import grpc

from protos import bff_pb2
from protos import bff_pb2_grpc
from protos import types_pb2


class BFFServicerTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.stub = bff_pb2_grpc.BFFStub(grpc.insecure_channel("localhost:50050"))
        cls.addr = 'localhost:50051'

    @classmethod
    def tearDownClass(cls):
        cls.stub = None

    def test_00_registeragent(self):
        res = self.stub.RegisterAgent(bff_pb2.AgentsInfo.Info(addr=self.addr))
        self.assertEqual(res.code, 0)

        res = self.stub.UnRegisterAgent(bff_pb2.AgentsInfo.Info(addr=self.addr))
        self.assertEqual(res.msg, '')

    def test_01_getagentsinfo(self):
        self.stub.RegisterAgent(bff_pb2.AgentsInfo.Info(addr=self.addr))
        res = self.stub.GetAgentsInfo(types_pb2.CommonRequest())
        self.assertEqual(res.infos[0].addr, self.addr)

    def test_02_agentsconfig(self):
        req = bff_pb2.AgentsConfig()
        with open('examples/states_inputs_func.py', 'r') as f1, \
             open('examples/outputs_actions_func.py', 'r') as f2, \
             open('examples/reward_func.py', 'r') as f3, \
             open('examples/hypers.json', 'r') as f4, \
             open('examples/builder.py', 'r') as f5:
            req.configs[self.addr].training = True
            req.configs[self.addr].states_inputs_func = f1.read()
            req.configs[self.addr].outputs_actions_func = f2.read()
            req.configs[self.addr].reward_func = f3.read()
            req.configs[self.addr].type = 'DQN'
            req.configs[self.addr].hypers = f4.read()
            req.configs[self.addr].builder = f5.read()
            req.configs[self.addr].structures = ''

        res = self.stub.SetAgentsConfig(req)
        self.assertEqual(res.code, 0)

        res = self.stub.GetAgentsConfig(types_pb2.CommonRequest())
        self.assertEqual(len(res.configs), 1)

    @unittest.skip('skip')
    def test_03_simconfig(self):
        with open('examples/sample_done_func.py', 'r') as f:
            sample_done_func = f.read()
        req = bff_pb2.SimConfig(
            exp_design_id=1,
            time_steps=1000,
            speed_ratio=1,
            sim_start_time=time.time(),
            sim_duration=10,
            exp_repeat_times=1,
            sample_done_func=sample_done_func,
        )
        res = self.stub.SetSimConfig(req)
        self.assertEqual(res.code, 0)

        res = self.stub.GetSimConfig(types_pb2.CommonRequest())
        self.assertEqual(res.exp_design_id, 1)

    def test_04_proxyconfig(self):
        with open('examples/proxy.json', 'r') as f:
            proxy = json.load(f)
        req = bff_pb2.ProxyConfig()
        for addr, config in proxy['configs'].items():
            for type, model in config['models'].items():
                req.configs[addr].models[type].name = model['name']
                for input, param in model['input_params'].items():
                    req.configs[addr].models[type].input_params[input].name = param['name']
                    req.configs[addr].models[type].input_params[input].type = param['type']
                    req.configs[addr].models[type].input_params[input].value = param['value']
                for output, param in model['output_params'].items():
                    req.configs[addr].models[type].output_params[output].name = param['name']
                    req.configs[addr].models[type].output_params[output].type = param['type']
                    req.configs[addr].models[type].output_params[output].value = param['value']
        req.sim_steps_ratio = proxy['sim_steps_ratio']
        res = self.stub.SetProxyConfig(req)
        self.assertEqual(res.code, 0)

        res = self.stub.GetProxyConfig(types_pb2.CommonRequest())
        self.assertEqual(len(res.configs), 1)

    @unittest.skip('skip')
    def test_05_control(self):
        req = bff_pb2.ControlCommand()

        req.cmd = bff_pb2.ControlCommand.Type.START
        res = self.stub.Control(req)
        self.assertEqual(res.code, 0)

        req.cmd = bff_pb2.ControlCommand.Type.SUSPEND
        res = self.stub.Control(req)
        self.assertEqual(res.code, 0)

        req.cmd = bff_pb2.ControlCommand.Type.CONTINUE
        res = self.stub.Control(req)
        self.assertEqual(res.code, 0)

    @unittest.skip('skip')
    def test_06_simstatus(self):
        for res in self.stub.GetSimStatus(types_pb2.CommonRequest()):
            self.assertEqual(res.srv_state, 1)

    @unittest.skip('skip')
    def test_07_proxychat(self):

        def generator():
            info = {'states': {'example': [1, 2, 3, 4]}}
            req = types_pb2.JsonString(json=json.dumps(info))
            for _ in range(10000):
                yield req
            yield types_pb2.JsonString(json=json.dumps(info))

        t1 = time.time()
        for res in self.stub.ProxyChat(generator()):
            info = json.loads(res.json)
            self.assertEqual(type(info['done']), bool)
        t2 = time.time()
        print()
        print(f'Time cost: {t2 - t1} s, FPS: {10000 / (t2 - t1)}')
