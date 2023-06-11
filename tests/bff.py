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
        cls.channel = grpc.insecure_channel('localhost:10000')
        cls.stub = bff_pb2_grpc.BFFStub(channel=cls.channel)
        cls.ids = []

    @classmethod
    def tearDownClass(cls):
        cls.stub.ResetService(bff_pb2.ServiceIdList(ids=cls.ids))
        cls.stub.ResetServer(types_pb2.CommonRequest())
        cls.channel.close()
        cls.stub = None

    def test_00_registerservice(self):
        with open('tests/examples/services.json', 'r') as f:
            services = json.load(f)
        services = {id: bff_pb2.ServiceInfo(**service) for id, service in services.items()}
        req = bff_pb2.ServiceInfoMap(services=services)

        self.stub.RegisterService(req)
        self.stub.UnRegisterService(bff_pb2.ServiceIdList(ids=[]))
        self.stub.RegisterService(req)
        self.ids += list(services.keys())

    def test_01_serviceinfo(self):
        res = self.stub.GetServiceInfo(bff_pb2.ServiceIdList(ids=self.ids[0:1]))
        self.assertIn(self.ids[0], res.services)
        self.stub.SetServiceInfo(res)

    @unittest.skip('Deprecated')
    def test_02_routeconfig(self):
        with open('tests/examples/routes.json', 'r') as f:
            routes = json.load(f)

        req = bff_pb2.RouteConfig()
        for simenv, agents in routes.items():
            for agent, models in agents.items():
                req.routes[simenv].configs[agent].models.extend(models)

        self.stub.SetRouteConfig(req)
        res = self.stub.GetRouteConfig(types_pb2.CommonRequest())
        self.assertIn(self.ids[0], res.routes)

    def test_03_resetservice(self):
        self.stub.ResetService(bff_pb2.ServiceIdList(ids=self.ids))

    def test_04_queryservice(self):
        res = self.stub.QueryService(bff_pb2.ServiceIdList(ids=self.ids))
        self.assertEqual(res.states[self.ids[0]].state, types_pb2.ServiceState.State.UNINITED)
        self.assertEqual(res.states[self.ids[1]].state, types_pb2.ServiceState.State.UNINITED)

    def test_05_agentconfig(self):
        req = bff_pb2.AgentConfigMap()
        with open('tests/examples/agent/hypers.json', 'r') as f1, \
             open('tests/examples/agent/states_inputs_func.py', 'r') as f2, \
             open('tests/examples/agent/outputs_actions_func.py', 'r') as f3, \
             open('tests/examples/agent/reward_func.py', 'r') as f4, \
             open('tests/examples/agent/hooks.json', 'r') as f5:
            req.configs[self.ids[1]].training = True
            req.configs[self.ids[1]].name = 'DQN'
            req.configs[self.ids[1]].hypers = f1.read()
            req.configs[self.ids[1]].sifunc = f2.read()
            req.configs[self.ids[1]].oafunc = f3.read()
            req.configs[self.ids[1]].rewfunc = f4.read()
            hooks = json.load(f5)
            for hook in hooks:
                pointer = req.configs[self.ids[1]].hooks.add()
                pointer.name = hook['name']
                pointer.args = json.dumps(hook['args'])

        self.stub.SetAgentConfig(req)
        res = self.stub.GetAgentConfig(bff_pb2.ServiceIdList(ids=[]))
        self.assertIn(self.ids[1], res.configs)

    def test_06_agentmode(self):
        req = bff_pb2.AgentModeMap()
        req.modes[self.ids[1]].training = True
        self.stub.SetAgentMode(req)
        res = self.stub.GetAgentMode(bff_pb2.ServiceIdList(ids=[]))
        self.assertIn(self.ids[1], res.modes)

    def test_07_modelweights(self):
        res = self.stub.GetModelWeights(bff_pb2.ServiceIdList(ids=[]))
        self.assertIn(self.ids[1], res.weights)
        self.stub.SetModelWeights(res)

    def test_08_modelbuffer(self):
        res = self.stub.GetModelBuffer(bff_pb2.ServiceIdList(ids=[]))
        self.assertIn(self.ids[1], res.buffers)
        self.stub.SetModelBuffer(res)

    def test_09_modelstatus(self):
        res = self.stub.GetModelStatus(bff_pb2.ServiceIdList(ids=[]))
        self.assertIn(self.ids[1], res.status)
        self.stub.SetModelStatus(res)

    def test_10_simenvconfig(self):
        req = bff_pb2.SimenvConfigMap()
        with open('tests/examples/simenv/args.json', 'r') as f1, \
             open('tests/examples/simenv/sim_term_func.cpp', 'r') as f2:
            req.configs[self.ids[0]].name = 'CQSIM'
            args = json.load(f1)
            args['proxy']['sim_term_func'] = f2.read()
            req.configs[self.ids[0]].args = json.dumps(args)
        self.stub.SetSimenvConfig(req)
        res = self.stub.GetSimenvConfig(bff_pb2.ServiceIdList(ids=[]))
        self.assertIn(self.ids[0], res.configs)

    def test_11_simcontrol(self):
        req = bff_pb2.SimCmdMap()

        req.cmds[self.ids[0]].type = 'init'
        self.stub.SimControl(req)

        req.cmds[self.ids[0]].type = 'start'
        self.stub.SimControl(req)

        time.sleep(30)

        req.cmds[self.ids[0]].type = 'stop'
        self.stub.SimControl(req)

    def test_12_simmonitor(self):
        res = self.stub.SimMonitor(bff_pb2.ServiceIdList(ids=[]))
        for id, info in res.infos.items():
            print('id: ', id)
            print('state: ', info.state)
            print('data: ', json.loads(info.data))
            print('logs: ', json.loads(info.logs))
