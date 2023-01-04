import json
import time
import unittest

import grpc
import numpy as np

from protos import bff_pb2
from protos import bff_pb2_grpc
from protos import simenv_pb2
from protos import types_pb2


class BFFServicerTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.channel = grpc.insecure_channel("localhost:10000")
        cls.stub = bff_pb2_grpc.BFFStub(channel=cls.channel)
        cls.ids = []

    @classmethod
    def tearDownClass(cls):
        cls.stub.ResetService(bff_pb2.ServiceIdList(ids=cls.ids))
        cls.stub.ResetServer(types_pb2.CommonRequest())
        cls.channel.close()
        cls.stub = None

    def test_00_registerservice(self):
        agent_service = bff_pb2.ServiceInfo(
            type=bff_pb2.ServiceInfo.Type.AGENT,
            name='agent',
            ip='localhost',
            port=10001,
            desc='agent service',
        )
        simenv_service = bff_pb2.ServiceInfo(
            type=bff_pb2.ServiceInfo.Type.SIMENV,
            name='simenv',
            ip='localhost',
            port=10002,
            desc='simenv service',
        )
        res = self.stub.RegisterService(bff_pb2.ServiceInfoList(services=[agent_service, simenv_service]))
        self.assertEqual(len(res.ids), 2)
        res = self.stub.UnRegisterService(bff_pb2.ServiceIdList(ids=[]))
        self.assertEqual(res.code, 0)

        res = self.stub.RegisterService(bff_pb2.ServiceInfoList(services=[agent_service, simenv_service]))
        self.ids += res.ids

    def test_01_serviceinfo(self):
        res = self.stub.GetServiceInfo(bff_pb2.ServiceIdList(ids=[self.ids[0]]))
        self.assertIn(self.ids[0], res.services)
        res = self.stub.SetServiceInfo(res)
        self.assertEqual(res.code, 0)

    def test_02_routeconfig(self):
        with open('examples/route/data_config.json', 'r') as f1, \
             open('examples/route/route_config.json', 'r') as f2, \
             open('examples/route/sim_term_func.py', 'r') as f3:
            data_config = json.load(f1)
            route_config = json.load(f2)
            sim_term_func = f3.read()

        req = bff_pb2.DataConfig()
        for type, struct in data_config['types'].items():
            for field, value in struct.items():
                req.types[type].fields[field] = value
        for type, model in data_config['data'].items():
            for input, param in model['input_params'].items():
                req.data[type].input_params[input].type = param['type']
                req.data[type].input_params[input].value = json.dumps(param['value'])
            for output, param in model['output_params'].items():
                req.data[type].output_params[output].type = param['type']
                req.data[type].output_params[output].value = '{}'
        res = self.stub.SetDataConfig(req)
        self.assertEqual(res.code, 0)
        res = self.stub.GetDataConfig(types_pb2.CommonRequest())
        self.assertIn('model1', res.data)

        req = bff_pb2.RouteConfig()
        for simenv, route in route_config['routes'].items():
            for agent, config in route['configs'].items():
                req.routes[simenv].configs[agent].models.extend(config['models'])
        req.sim_term_func = sim_term_func
        req.sim_step_ratio = route_config['sim_step_ratio']
        res = self.stub.SetRouteConfig(req)
        self.assertEqual(res.code, 0)
        res = self.stub.GetRouteConfig(types_pb2.CommonRequest())
        self.assertIn(self.ids[1], res.routes)

    def test_03_resetservice(self):
        res = self.stub.ResetService(bff_pb2.ServiceIdList(ids=self.ids))
        self.assertEqual(res.code, 0)

    def test_04_queryservice(self):
        res = self.stub.QueryService(bff_pb2.ServiceIdList(ids=self.ids))
        self.assertEqual(res.states[self.ids[0]].state, types_pb2.ServiceState.State.UNINITED)
        self.assertEqual(res.states[self.ids[1]].state, types_pb2.ServiceState.State.UNINITED)

    def test_05_agentconfig(self):
        req = bff_pb2.AgentConfigMap()
        with open('examples/agent/states_inputs_func.py', 'r') as f1, \
             open('examples/agent/outputs_actions_func.py', 'r') as f2, \
             open('examples/agent/reward_func.py', 'r') as f3, \
             open('examples/agent/hypers_dqn.json', 'r') as f4, \
             open('examples/agent/builder_dqn.py', 'r') as f5, \
             open('examples/agent/structures_dqn.json', 'r') as f6:
            req.configs[self.ids[0]].training = True
            req.configs[self.ids[0]].states_inputs_func = f1.read()
            req.configs[self.ids[0]].outputs_actions_func = f2.read()
            req.configs[self.ids[0]].reward_func = f3.read()
            req.configs[self.ids[0]].type = 'DQN'
            req.configs[self.ids[0]].hypers = f4.read()
            req.configs[self.ids[0]].builder = f5.read()
            req.configs[self.ids[0]].structures = f6.read()
        res = self.stub.SetAgentConfig(req)
        self.assertEqual(res.code, 0)
        res = self.stub.GetAgentConfig(bff_pb2.ServiceIdList(ids=[]))
        self.assertIn(self.ids[0], res.configs)

    def test_06_agentmode(self):
        req = bff_pb2.AgentModeMap()
        req.modes[self.ids[0]].training = True
        res = self.stub.SetAgentMode(req)
        self.assertEqual(res.code, 0)
        res = self.stub.GetAgentMode(bff_pb2.ServiceIdList(ids=[]))
        self.assertIn(self.ids[0], res.modes)

    def test_07_modelweights(self):
        res = self.stub.GetModelWeights(bff_pb2.ServiceIdList(ids=[]))
        self.assertIn(self.ids[0], res.weights)
        res = self.stub.SetModelWeights(res)
        self.assertEqual(res.code, 0)

    def test_08_modelbuffer(self):
        res = self.stub.GetModelBuffer(bff_pb2.ServiceIdList(ids=[]))
        self.assertIn(self.ids[0], res.buffers)
        res = self.stub.SetModelBuffer(res)
        self.assertEqual(res.code, 0)

    def test_09_modelstatus(self):
        res = self.stub.GetModelStatus(bff_pb2.ServiceIdList(ids=[]))
        self.assertIn(self.ids[0], res.status)
        res = self.stub.SetModelStatus(res)
        self.assertEqual(res.code, 0)

    def test_10_simenvconfig(self):
        req = bff_pb2.SimenvConfigMap()
        req.configs[self.ids[1]].type = 'CQSim'
        req.configs[self.ids[1]].args = json.dumps({'engine_addr': 'localhost:50041'})
        res = self.stub.SetSimenvConfig(req)
        self.assertEqual(res.code, 0)
        res = self.stub.GetSimenvConfig(bff_pb2.ServiceIdList(ids=[]))
        self.assertIn(self.ids[1], res.configs)

    def test_11_simcontrol(self):
        req = bff_pb2.SimCmdMap()

        req.cmds[self.ids[1]].type = simenv_pb2.SimCmd.Type.INIT
        req.cmds[self.ids[1]].params = json.dumps({
            'exp_design_id': 28,
            'sim_start_time': int(time.time()),
            'sim_duration': 30,
            'time_step': 1000,
            'speed_ratio': 10,
        })
        self.stub.SimControl(req)

        req.cmds[self.ids[1]].type = simenv_pb2.SimCmd.Type.START
        req.cmds[self.ids[1]].params = json.dumps({})
        self.stub.SimControl(req)

        time.sleep(5)

        req.cmds[self.ids[1]].type = simenv_pb2.SimCmd.Type.STOP
        self.stub.SimControl(req)

    def test_12_simmonitor(self):
        print(self.stub.SimMonitor(types_pb2.CommonRequest()))

    def test_13_proxychat(self):
        req_info = {
            'states': {
                'model1': [{
                    'output1': np.random.rand(20).tolist(),
                }]
            },
            'truncated': False,
        }
        req = types_pb2.JsonString(json=json.dumps(req_info))

        def generator():
            for _ in range(5000):
                yield req

        t1 = time.time()
        for res in self.stub.ProxyChat(generator(), metadata=[('id', self.ids[1])]):
            res_info = json.loads(res.json)
        t2 = time.time()

        self.assertIn('actions', res_info)
        print(f'Time cost: {t2 - t1} s, FPS: {5000 / (t2 - t1)}')
