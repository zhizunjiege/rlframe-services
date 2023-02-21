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
        with open('examples/services.json', 'r') as f:
            services = json.load(f)
        services = [bff_pb2.ServiceInfo(**service) for service in services]

        res = self.stub.RegisterService(bff_pb2.ServiceInfoList(services=services))
        self.assertEqual(len(res.ids), 2)
        res = self.stub.UnRegisterService(bff_pb2.ServiceIdList(ids=[]))
        self.assertEqual(res.code, 0)
        res = self.stub.RegisterService(bff_pb2.ServiceInfoList(services=services))
        self.ids += res.ids

    def test_01_serviceinfo(self):
        res = self.stub.GetServiceInfo(bff_pb2.ServiceIdList(ids=self.ids[0:1]))
        self.assertIn(self.ids[0], res.services)
        res = self.stub.SetServiceInfo(res)
        self.assertEqual(res.code, 0)

    def test_02_routeconfig(self):
        with open('examples/routes.json', 'r') as f:
            routes = json.load(f)

        req = bff_pb2.RouteConfig()
        for simenv, agents in routes.items():
            for agent, models in agents.items():
                req.routes[simenv].configs[agent].models.extend(models)

        res = self.stub.SetRouteConfig(req)
        self.assertEqual(res.code, 0)
        res = self.stub.GetRouteConfig(types_pb2.CommonRequest())
        self.assertIn(self.ids[0], res.routes)

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
             open('examples/agent/hypers.json', 'r') as f4, \
             open('examples/agent/structs.json', 'r') as f5, \
             open('examples/agent/builder.py', 'r') as f6:
            req.configs[self.ids[1]].training = True
            req.configs[self.ids[1]].states_inputs_func = f1.read()
            req.configs[self.ids[1]].outputs_actions_func = f2.read()
            req.configs[self.ids[1]].reward_func = f3.read()
            hypers = json.load(f4)
            req.configs[self.ids[1]].type = hypers['type']
            req.configs[self.ids[1]].hypers = json.dumps(hypers['hypers'])
            req.configs[self.ids[1]].structs = f5.read()
            req.configs[self.ids[1]].builder = f6.read()

        res = self.stub.SetAgentConfig(req)
        self.assertEqual(res.code, 0)
        res = self.stub.GetAgentConfig(bff_pb2.ServiceIdList(ids=[]))
        self.assertIn(self.ids[1], res.configs)

    def test_06_agentmode(self):
        req = bff_pb2.AgentModeMap()
        req.modes[self.ids[1]].training = True
        res = self.stub.SetAgentMode(req)
        self.assertEqual(res.code, 0)
        res = self.stub.GetAgentMode(bff_pb2.ServiceIdList(ids=[]))
        self.assertIn(self.ids[1], res.modes)

    def test_07_modelweights(self):
        res = self.stub.GetModelWeights(bff_pb2.ServiceIdList(ids=[]))
        self.assertIn(self.ids[1], res.weights)
        res = self.stub.SetModelWeights(res)
        self.assertEqual(res.code, 0)

    def test_08_modelbuffer(self):
        res = self.stub.GetModelBuffer(bff_pb2.ServiceIdList(ids=[]))
        self.assertIn(self.ids[1], res.buffers)
        res = self.stub.SetModelBuffer(res)
        self.assertEqual(res.code, 0)

    def test_09_modelstatus(self):
        res = self.stub.GetModelStatus(bff_pb2.ServiceIdList(ids=[]))
        self.assertIn(self.ids[1], res.status)
        res = self.stub.SetModelStatus(res)
        self.assertEqual(res.code, 0)

    def test_10_simenvconfig(self):
        with open('examples/simenv/args.json', 'r') as f:
            args = json.load(f)

        req = bff_pb2.SimenvConfigMap()
        req.configs[self.ids[0]].type = args['type']
        req.configs[self.ids[0]].args = json.dumps(args['args'])
        res = self.stub.SetSimenvConfig(req)
        self.assertEqual(res.code, 0)
        res = self.stub.GetSimenvConfig(bff_pb2.ServiceIdList(ids=[]))
        self.assertIn(self.ids[0], res.configs)

    def test_11_simcontrol(self):
        req = bff_pb2.SimCmdMap()

        with open('examples/simenv/configs.json', 'r') as f1, open('examples/simenv/sim_term_func.cc', 'r') as f2:
            sim_params = json.load(f1)
            sim_params['proxy']['sim_term_func'] = f2.read()

        req.cmds[self.ids[0]].type = 'init'
        req.cmds[self.ids[0]].params = json.dumps(sim_params)
        self.stub.SimControl(req)

        req.cmds[self.ids[0]].type = 'start'
        req.cmds[self.ids[0]].params = '{}'
        self.stub.SimControl(req)

        time.sleep(30)

        req.cmds[self.ids[0]].type = 'stop'
        req.cmds[self.ids[0]].params = '{}'
        self.stub.SimControl(req)

    def test_12_simmonitor(self):
        res = self.stub.SimMonitor(bff_pb2.ServiceIdList(ids=[]))
        for id, info in res.infos.items():
            print('id: ', id)
            print('state: ', info.state)
            print('data: ', json.loads(info.data))
            print('logs: ', json.loads(info.logs))
