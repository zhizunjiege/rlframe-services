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

    def test_00_registerservice(self):
        with open('tests/examples/agent/service.json', 'r') as f1, \
             open('tests/examples/simenv/service.json', 'r') as f2:
            agent_service = json.load(f1)
            simenv_service = json.load(f2)
        services = {
            service['id']: bff_pb2.ServiceInfo(
                type=service['type'],
                name=service['name'],
                host=service['host'],
                port=service['port'],
                desc=service['desc'],
            ) for service in [agent_service, simenv_service]
        }
        req = bff_pb2.ServiceInfoMap(services=services)

        self.stub.RegisterService(req)
        self.stub.UnRegisterService(bff_pb2.ServiceIdList(ids=[]))
        self.stub.RegisterService(req)
        self.ids += list(services.keys())

    def test_01_serviceinfo(self):
        res = self.stub.GetServiceInfo(bff_pb2.ServiceIdList(ids=self.ids[0:1]))
        self.assertIn(self.ids[0], res.services)
        self.stub.SetServiceInfo(res)

    def test_02_resetservice(self):
        self.stub.ResetService(bff_pb2.ServiceIdList(ids=self.ids))

    def test_03_queryservice(self):
        res = self.stub.QueryService(bff_pb2.ServiceIdList(ids=self.ids))
        self.assertEqual(res.states[self.ids[0]].state, types_pb2.ServiceState.State.UNINITED)
        self.assertEqual(res.states[self.ids[1]].state, types_pb2.ServiceState.State.UNINITED)

    def test_04_agentconfig(self):
        req = bff_pb2.AgentConfigMap()
        with open('tests/examples/agent/configs.json', 'r') as f1, \
             open('tests/examples/agent/states_inputs_func.py', 'r') as f2, \
             open('tests/examples/agent/outputs_actions_func.py', 'r') as f3, \
             open('tests/examples/agent/reward_func.py', 'r') as f4:
            configs = json.load(f1)
            req.configs[self.ids[0]].training = configs['training']
            req.configs[self.ids[0]].name = configs['name']
            req.configs[self.ids[0]].hypers = json.dumps(configs['hypers'])
            req.configs[self.ids[0]].sifunc = f2.read()
            req.configs[self.ids[0]].oafunc = f3.read()
            req.configs[self.ids[0]].rewfunc = f4.read()
            for hook in configs['hooks']:
                pointer = req.configs[self.ids[0]].hooks.add()
                pointer.name = hook['name']
                pointer.args = json.dumps(hook['args'])

        self.stub.SetAgentConfig(req)
        res = self.stub.GetAgentConfig(bff_pb2.ServiceIdList(ids=[]))
        self.assertIn(self.ids[0], res.configs)

    def test_05_agentmode(self):
        req = bff_pb2.AgentModeMap()
        req.modes[self.ids[0]].training = True
        self.stub.SetAgentMode(req)
        res = self.stub.GetAgentMode(bff_pb2.ServiceIdList(ids=[]))
        self.assertIn(self.ids[0], res.modes)

    def test_06_modelweights(self):
        res = self.stub.GetModelWeights(bff_pb2.ServiceIdList(ids=[]))
        self.assertIn(self.ids[0], res.weights)
        self.stub.SetModelWeights(res)

    def test_07_modelbuffer(self):
        res = self.stub.GetModelBuffer(bff_pb2.ServiceIdList(ids=[]))
        self.assertIn(self.ids[0], res.buffers)
        self.stub.SetModelBuffer(res)

    def test_08_modelstatus(self):
        res = self.stub.GetModelStatus(bff_pb2.ServiceIdList(ids=[]))
        self.assertIn(self.ids[0], res.status)
        self.stub.SetModelStatus(res)

    def test_09_simenvconfig(self):
        req = bff_pb2.SimenvConfigMap()
        with open('tests/examples/simenv/configs.json', 'r') as f1, \
             open('tests/examples/simenv/sim_term_func.cpp', 'r') as f2:
            configs = json.load(f1)
            req.configs[self.ids[1]].name = configs['name']
            configs['args']['proxy']['sim_term_func'] = f2.read()
            req.configs[self.ids[1]].args = json.dumps(configs['args'])
        self.stub.SetSimenvConfig(req)
        res = self.stub.GetSimenvConfig(bff_pb2.ServiceIdList(ids=[]))
        self.assertIn(self.ids[1], res.configs)

    def test_10_simcontrol(self):
        req = bff_pb2.SimCmdMap()

        req.cmds[self.ids[1]].type = 'init'
        self.stub.SimControl(req)

        req.cmds[self.ids[1]].type = 'start'
        self.stub.SimControl(req)

        time.sleep(30)

        req.cmds[self.ids[1]].type = 'stop'
        self.stub.SimControl(req)

    def test_11_simmonitor(self):
        res = self.stub.SimMonitor(bff_pb2.ServiceIdList(ids=[]))
        for id, info in res.infos.items():
            print('id: ', id)
            print('state: ', info.state)
            print('data: ', json.loads(info.data))
            print('logs: ', json.loads(info.logs))
