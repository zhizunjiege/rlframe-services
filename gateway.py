import argparse
from concurrent import futures
import itertools
import json
import pickle

import grpc

from protos import agent_pb2
from protos import agent_pb2_grpc
from protos import engine_pb2
from protos import engine_pb2_grpc
from protos import gateway_pb2
from protos import gateway_pb2_grpc
from protos import types_pb2


class GatewayServicer(gateway_pb2_grpc.GatewayServicer):

    def __init__(self, engine):
        self.__reset_all()

        self.engine = engine_pb2_grpc.SimControllerStub(channel=grpc.insecure_channel(engine))

    def __reset_all(self):
        self.agents = {}

        self.param_configs = None
        self.route_configs = None
        self.sim_configs = None

        self.sdfunc = None

        self.srv_state = gateway_pb2.SimStatus.ServerState.STOPPED

    def RegisterAgent(self, request, context):
        self.agents[request.addr] = agent_pb2_grpc.AgentStub(channel=grpc.insecure_channel(request.addr))
        return types_pb2.CommonResponse()

    def UnRegisterAgent(self, request, context):
        del self.agents[request.addr]
        return types_pb2.CommonResponse()

    def GetAgentsInfo(self, request, context):
        infos = [gateway_pb2.AgentInfo(addr=addr) for addr in list(self.agents.keys())]
        return gateway_pb2.AgentsInfo(infos=infos)

    def GetParamConfig(self, request, context):
        return self.param_configs or gateway_pb2.ParamConfig()

    def SetParamConfig(self, request, context):
        self.param_configs = request
        return types_pb2.CommonResponse()

    def GetRouteConfig(self, request, context):
        return self.route_configs or gateway_pb2.RouteConfig()

    def SetRouteConfig(self, request, context):
        self.route_configs = request
        return types_pb2.CommonResponse()

    def GetSimConfig(self, request, context):
        return self.sim_configs or gateway_pb2.SimConfig()

    def SetSimConfig(self, request, context):
        self.sim_configs = request

        sdfunc_src = request.sample_done_func + '\ndone = func(states)'
        self.sdfunc = compile(source=sdfunc_src, model='exec')

        return types_pb2.CommonResponse()

    def Control(self, request, context):
        # TODO: control the simulation
        return types_pb2.CommonResponse()

    def GetSimStatus(self, request, context):
        for info in self.engine.GetSysInfo(types_pb2.CommonRequest()):
            status = gateway_pb2.SimStatus(
                srv_state=self.srv_state,
                sim_cur_time=info.sim_current_time.seconds,
                sim_duration=info.sim_duration.seconds,
                real_duration=info.real_duration.seconds,
                real_speed_ratio=info.real_speed_ratio,
                cur_sample_id=info.current_sample_id,
                exp_repeated_time=1,  # TODO
                sim_steps_remainder=0,  # TODO
            )
            yield status
        return gateway_pb2.SimStatus()

    def GetAgentsConfig(self, request, context):
        addrs = request.addrs if len(request.addrs) > 0 else list(self.agents.keys())
        configs = {}
        for addr in addrs:
            stub = self.agents[addr]
            configs[addr] = stub.GetAgentConfig(types_pb2.CommonRequest())
        return gateway_pb2.AgentsConfig(configs=configs)

    def SetAgentsConfig(self, request, context):
        configs = request.configs
        for addr in configs:
            stub = self.agents[addr]
            stub.SetAgentConfig(configs[addr])
        return types_pb2.CommonResponse()

    def RstAgentsConfig(self, request, context):
        for addr in self.agents:
            stub = self.agents[addr]
            stub.RstAgentConfig(types_pb2.CommonRequest())
        return types_pb2.CommonResponse()

    def GetAgentsMode(self, request, context):
        addrs = request.addrs if len(request.addrs) > 0 else list(self.agents.keys())
        modes = {}
        for addr in addrs:
            stub = self.agents[addr]
            modes[addr] = stub.GetAgentMode(types_pb2.CommonRequest())
        return gateway_pb2.AgentsMode(modes=modes)

    def SetAgentsMode(self, request, context):
        modes = request.modes
        for addr in modes:
            stub = self.agents[addr]
            stub.SetAgentMode(modes[addr])
        return types_pb2.CommonResponse()

    def GetAgentsWeight(self, request, context):
        addrs = request.addrs if len(request.addrs) > 0 else list(self.agents.keys())
        weights = {}
        for addr in addrs:
            stub = self.agents[addr]
            weights[addr] = stub.GetAgentWeight(types_pb2.CommonRequest())
        return gateway_pb2.AgentsWeight(weights=weights)

    def SetAgentsWeight(self, request, context):
        weights = request.weights
        for addr in weights:
            stub = self.agents[addr]
            stub.SetAgentWeight(weights[addr])
        return types_pb2.CommonResponse()

    def GetAgentsBuffer(self, request, context):
        addrs = request.addrs if len(request.addrs) > 0 else list(self.agents.keys())
        buffers = {}
        for addr in addrs:
            stub = self.agents[addr]
            buffers[addr] = stub.GetAgentBuffer(types_pb2.CommonRequest())
        return gateway_pb2.AgentsBuffer(buffers=buffers)

    def SetAgentsBuffer(self, request, context):
        buffers = request.buffers
        for addr in buffers:
            stub = self.agents[addr]
            stub.SetAgentBuffer(buffers[addr])
        return types_pb2.CommonResponse()

    def GetAgentsStatus(self, request, context):
        addrs = request.addrs if len(request.addrs) > 0 else list(self.agents.keys())
        status = {}
        for addr in addrs:
            stub = self.agents[addr]
            status[addr] = stub.GetAgentStatus(types_pb2.CommonRequest())
        return gateway_pb2.AgentsStatus(status=status)

    def SetAgentsStatus(self, request, context):
        status = request.status
        for addr in status:
            stub = self.agents[addr]
            stub.SetAgentStatus(status[addr])
        return types_pb2.CommonResponse()

    def GetAction(self, request_iterator, context):

        def source_generator():
            for request in request_iterator:
                result = {'states': json.loads(request.states), 'done': False}
                exec(self.sdfunc, result)
                # TODO: Route states to agents
                yield agent_pb2.PickleBytes(pkl=pickle.dumps(result))

        src_its = itertools.tee(source_generator(), len(self.agents))
        tgt_its = [self.agents[addr].GetAction(src_its[index]) for index, addr in enumerate(self.agents)]

        while True:
            try:
                actions = {}
                for it in tgt_its:
                    val = next(it)
                    result = pickle.loads(val.pkl)
                    actions.update(result['actions'])
                yield gateway_pb2.JsonString(json=json.dumps(actions))
            except StopIteration:
                return gateway_pb2.JsonString()


def gateway_server(engine, ip, port, max_workers):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    gateway_pb2_grpc.add_GatewayServicer_to_server(GatewayServicer(engine=engine), server)
    server.add_insecure_port(f'{ip}:{port}')
    server.start()
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        ...


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run an gateway service.')
    parser.add_argument('-e', '--engine', type=str, default='localhost:50041', help='Engine address')
    parser.add_argument('-i', '--ip', type=str, default='0.0.0.0', type=str, help='IP address to listen on.')
    parser.add_argument('-p', '--port', default=0, type=int, help='Port to listen on.')
    parser.add_argument('-w', '--work', default=10, type=int, help='Max workers.')
    args = parser.parse_args()
    gateway_server(args.engine, args.ip, args.port, args.work)
