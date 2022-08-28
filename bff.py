import argparse
from concurrent import futures
import json

import grpc

from protos import agent_pb2_grpc
from protos import engine_pb2
from protos import engine_pb2_grpc
from protos import bff_pb2
from protos import bff_pb2_grpc
from protos import types_pb2


class BFFServicer(bff_pb2_grpc.BFFServicer):

    def __init__(self, engine):
        self.__reset_all()

        self.engine = engine_pb2_grpc.SimControllerStub(channel=grpc.insecure_channel(engine))

    def __reset_all(self):
        self.agents = {}

        self.proxy_configs = None
        self.sim_configs = None

        self.sdfunc = None

        self.srv_state = bff_pb2.SimStatus.ServerState.STOPPED
        self.exp_repeated_time = 0  # TODO

    def GetProxyConfig(self, request, context):
        return self.proxy_configs or bff_pb2.ProxyConfig()

    def SetProxyConfig(self, request, context):
        self.proxy_configs = request
        return types_pb2.CommonResponse()

    def ProxyChat(self, request_iterator, context):
        for request in request_iterator:
            result = {'states': json.loads(request.states), 'done': False}
            exec(self.sdfunc, result)
            done = result['done']
            if done:
                self.engine.Control(engine_pb2.ControlCmd(run_cmd=engine_pb2.ControlCmd.RunCmdType.STOP_CURRENT_SAMPLE))
                return bff_pb2.JsonString(json.dumps(None))
            else:
                yield types_pb2.JsonString(json=json.dumps({'done': done}))

    def Control(self, request, context):
        if request.cmd == bff_pb2.ControlCommand.Type.START:
            sample = engine_pb2.InitInfo.MultiSample(exp_design_id=self.sim_configs.exp_design_id)
            self.engine.Init(engine_pb2.InitInfo(multi_sample_config=sample))
            self.engine.Control(engine_pb2.ControlCmd(sim_start_time=self.sim_configs.sim_start_time))
            self.engine.Control(engine_pb2.ControlCmd(sim_duration=self.sim_configs.sim_duration))
            self.engine.Control(engine_pb2.ControlCmd(time_step=self.sim_configs.time_step))
            self.engine.Control(engine_pb2.ControlCmd(speed_ratio=self.sim_configs.speed_ratio))
            self.engine.Control(engine_pb2.ControlCmd(run_cmd=engine_pb2.ControlCmd.RunCmdType.START))
            self.srv_state = bff_pb2.SimStatus.ServerState.RUNNING
        elif request.cmd == bff_pb2.ControlCommand.Type.SUSPEND:
            self.engine.Control(engine_pb2.ControlCmd(run_cmd=engine_pb2.ControlCmd.RunCmdType.SUSPEND))
            self.srv_state = bff_pb2.SimStatus.ServerState.SUSPENDED
        elif request.cmd == bff_pb2.ControlCommand.Type.CONTINUE:
            self.engine.Control(engine_pb2.ControlCmd(run_cmd=engine_pb2.ControlCmd.RunCmdType.CONTINUE))
            self.srv_state = bff_pb2.SimStatus.ServerState.RUNNING
        else:
            self.engine.Control(engine_pb2.ControlCmd(run_cmd=engine_pb2.ControlCmd.RunCmdType.STOP))
            self.__reset_all()
        return types_pb2.CommonResponse()

    def GetSimConfig(self, request, context):
        return self.sim_configs or bff_pb2.SimConfig()

    def SetSimConfig(self, request, context):
        self.sim_configs = request

        sdfunc_src = request.sample_done_func + '\ndone = func(states)'
        self.sdfunc = compile(source=sdfunc_src, model='exec')

        return types_pb2.CommonResponse()

    def GetSimStatus(self, request, context):
        for info in self.engine.GetSysInfo(types_pb2.CommonRequest()):
            status = bff_pb2.SimStatus(
                srv_state=self.srv_state,
                sim_cur_time=info.sim_current_time.seconds,
                sim_duration=info.sim_duration.seconds,
                real_duration=info.real_duration.seconds,
                real_speed_ratio=info.real_speed_ratio,
                cur_sample_id=info.current_sample_id,
                exp_repeated_time=self.exp_repeated_time,
            )
            yield status
        return bff_pb2.SimStatus()

    def RegisterAgent(self, request, context):
        self.agents[request.addr] = agent_pb2_grpc.AgentStub(channel=grpc.insecure_channel(request.addr))
        return types_pb2.CommonResponse()

    def UnRegisterAgent(self, request, context):
        del self.agents[request.addr]
        return types_pb2.CommonResponse()

    def GetAgentsInfo(self, request, context):
        infos = [bff_pb2.AgentsInfo.Info(addr=addr) for addr in self.agents.keys()]
        return bff_pb2.AgentsInfo(infos=infos)

    def GetAgentsConfig(self, request, context):
        addrs = request.addrs if len(request.addrs) > 0 else list(self.agents.keys())
        configs = {}
        for addr in addrs:
            stub = self.agents[addr]
            configs[addr] = stub.GetAgentConfig(types_pb2.CommonRequest())
        return bff_pb2.AgentsConfig(configs=configs)

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
        return bff_pb2.AgentsMode(modes=modes)

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
        return bff_pb2.AgentsWeight(weights=weights)

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
        return bff_pb2.AgentsBuffer(buffers=buffers)

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
        return bff_pb2.AgentsStatus(status=status)

    def SetAgentsStatus(self, request, context):
        status = request.status
        for addr in status:
            stub = self.agents[addr]
            stub.SetAgentStatus(status[addr])
        return types_pb2.CommonResponse()


def bff_server(engine, ip, port, max_workers):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    bff_pb2_grpc.add_BFFServicer_to_server(BFFServicer(engine=engine), server)
    server.add_insecure_port(f'{ip}:{port}')
    server.start()
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        ...


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run an bff service.')
    parser.add_argument('-e', '--engine', type=str, default='localhost:50041', help='Engine address')
    parser.add_argument('-i', '--ip', type=str, default='0.0.0.0', help='IP address to listen on.')
    parser.add_argument('-p', '--port', type=int, default=0, help='Port to listen on.')
    parser.add_argument('-w', '--work', type=int, default=10, help='Max workers.')
    args = parser.parse_args()
    bff_server(args.engine, args.ip, args.port, args.work)
