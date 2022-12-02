import argparse
from concurrent import futures
import hashlib
import json

import grpc

from protos import agent_pb2_grpc
from protos import bff_pb2
from protos import bff_pb2_grpc
from protos import types_pb2

from simenvs import SimEnvs


class BFFServicer(bff_pb2_grpc.BFFServicer):

    def __init__(self):
        self.__reset_all()

    def __reset_all(self):
        self.services = {}

        self.data_config = None

        self.route_config = None
        self.sim_done_func = None
        self.sim_steps = 0  # TODO

        self.state = bff_pb2.SimInfo.State.STOPPED

        self.simenvs = {}
        self.agents = {}

    def ResetServer(self, request, context):
        self.__reset_all()
        return types_pb2.CommonResponse()

    def RegisterService(self, request, context):
        ids = []
        for service in request.services:
            key = f'{service.type}-{service.subtype}-{service.name} {service.ip}:{service.port}'
            sha256 = hashlib.sha256()
            sha256.update(key.encode('utf-8'))
            id = sha256.hexdigest()
            self.services[id] = service
            ids.append(id)
            if service.type == bff_pb2.ServiceType.SIMENV:
                params = json.loads(service.params) if service.params else {}
                self.simenvs[id] = SimEnvs[service.subtype](id=id, **params)
            else:
                channel = grpc.insecure_channel(f'{service.ip}:{service.port}')
                self.agents[id] = agent_pb2_grpc.AgentStub(channel)
        return bff_pb2.ServiceIdList(ids=ids)

    def UnRegisterService(self, request, context):
        ids = request.ids if len(request.ids) > 0 else list(self.services.keys())
        for id in ids:
            del self.services[id]
            if id in self.simenvs:
                self.simenvs[id].close()
                del self.simenvs[id]
            else:
                self.agents[id].ResetServer(types_pb2.CommonRequest())
                del self.agents[id]
        return types_pb2.CommonResponse()

    def GetServiceInfo(self, request, context):
        services = {}
        ids = request.ids if len(request.ids) > 0 else list(self.services.keys())
        for id in ids:
            services[id] = self.services[id]
        return bff_pb2.ServiceInfoMap(services=services)

    def SetServiceInfo(self, request, context):
        for id in request.services:
            self.services[id] = request.services[id]
        return types_pb2.CommonResponse()

    def GetDataConfig(self, request, context):
        return self.data_config

    def SetDataConfig(self, request, context):
        self.data_config = request
        return types_pb2.CommonResponse()

    def ProxyChat(self, request_iterator, context):
        metadata = context.invocation_metadata()
        if len(metadata) > 0 and metadata[0][0] == 'id':
            id = metadata[0][1]
        else:
            ip = context.peer().split(':')[1:]
            for id_ in self.simenvs:
                if self.services[id_].ip == ip:
                    id = id_
                    break
                else:
                    id = ''

        response = types_pb2.JsonString()
        if id in self.simenvs:
            route = self.route_config[id]
            for request in request_iterator:
                info = json.loads(request.json)
                exec(self.sim_done_func, info)
                states, done, actions = info['states'], info['done'], {}
                for id_ in route.configs:
                    states_ = {}
                    for model in route.configs[id_].models:
                        states_[model] = states[model]
                    result = self.agents[id_].GetAction({'states': states_, 'done': done})
                    if not done:
                        actions_ = json.loads(result.json)['actions']
                        actions.update(actions_)
                response.json = json.dumps({'actions': actions})
                if done:
                    self.simenvs[id].control('done', {})
                    return response
                else:
                    yield response
        else:
            response.json = json.dumps({'error': 'Invalid id!'})
            return response

    def GetRouteConfig(self, request, context):
        return self.route_config

    def SetRouteConfig(self, request, context):
        self.route_config = request
        sdfunc_src = request.sim_done_func + '\ndone = func(states)'
        self.sim_done_func = compile(sdfunc_src, '', 'exec')
        return types_pb2.CommonResponse()

    def SimControl(self, request, context):
        if request.cmd == bff_pb2.SimCmd.Type.START:
            cmd = 'start'
            self.state = bff_pb2.SimInfo.State.RUNNING
        elif request.cmd == bff_pb2.SimCmd.Type.PAUSE:
            cmd = 'pause'
            self.state = bff_pb2.SimInfo.State.SUSPENDED
        elif request.cmd == bff_pb2.SimCmd.Type.STEP:
            cmd = 'step'
            self.state = bff_pb2.SimInfo.State.SUSPENDED
        elif request.cmd == bff_pb2.SimCmd.Type.RESUME:
            cmd = 'resume'
            self.state = bff_pb2.SimInfo.State.RUNNING
        elif request.cmd == bff_pb2.SimCmd.Type.STOP:
            cmd = 'stop'
            self.state = bff_pb2.SimInfo.State.STOPPED
        else:
            cmd = 'param'
        for id in self.simenvs:
            if id in request.params:
                params = json.loads(request.params[id])
            else:
                params = {}
            self.simenvs[id].control(cmd, params)
        return types_pb2.CommonResponse()

    def SimMonitor(self, request, context):
        data, logs = {}, {}
        for id in self.simenvs:
            data_, logs_ = self.simenvs[id].monitor()
            data[id], logs[id] = json.dumps(data_), json.dumps(logs_)
        return bff_pb2.SimInfo(state=self.state, data=data, logs=logs)

    def GetAgentConfig(self, request, context):
        configs = {}
        ids = request.ids if len(request.ids) > 0 else list(self.agents.keys())
        for id in ids:
            stub = self.agents[id]
            configs[id] = stub.GetAgentConfig(types_pb2.CommonRequest())
        return bff_pb2.AgentConfigMap(configs=configs)

    def SetAgentConfig(self, request, context):
        for id in request.configs:
            stub = self.agents[id]
            stub.SetAgentConfig(request.configs[id])
        return types_pb2.CommonResponse()

    def GetAgentMode(self, request, context):
        modes = {}
        ids = request.ids if len(request.ids) > 0 else list(self.agents.keys())
        for id in ids:
            stub = self.agents[id]
            modes[id] = stub.GetAgentMode(types_pb2.CommonRequest())
        return bff_pb2.AgentModeMap(modes=modes)

    def SetAgentMode(self, request, context):
        for id in request.modes:
            stub = self.agents[id]
            stub.SetAgentMode(request.modes[id])
        return types_pb2.CommonResponse()

    def GetAgentWeight(self, request, context):
        weights = {}
        ids = request.ids if len(request.ids) > 0 else list(self.agents.keys())
        for id in ids:
            stub = self.agents[id]
            weights[id] = stub.GetAgentWeight(types_pb2.CommonRequest())
        return bff_pb2.AgentWeightMap(weights=weights)

    def SetAgentWeight(self, request, context):
        for id in request.weights:
            stub = self.agents[id]
            stub.SetAgentWeight(request.weights[id])
        return types_pb2.CommonResponse()

    def GetAgentBuffer(self, request, context):
        buffers = {}
        ids = request.ids if len(request.ids) > 0 else list(self.agents.keys())
        for id in ids:
            stub = self.agents[id]
            buffers[id] = stub.GetAgentBuffer(types_pb2.CommonRequest())
        return bff_pb2.AgentBufferMap(buffers=buffers)

    def SetAgentBuffer(self, request, context):
        for id in request.buffers:
            stub = self.agents[id]
            stub.SetAgentBuffer(request.buffers[id])
        return types_pb2.CommonResponse()

    def GetAgentStatus(self, request, context):
        status = {}
        ids = request.ids if len(request.ids) > 0 else list(self.agents.keys())
        for id in ids:
            stub = self.agents[id]
            status[id] = stub.GetAgentStatus(types_pb2.CommonRequest())
        return bff_pb2.AgentStatusMap(status=status)

    def SetAgentStatus(self, request, context):
        for id in request.status:
            stub = self.agents[id]
            stub.SetAgentStatus(request.status[id])
        return types_pb2.CommonResponse()


def bff_server(ip, port, max_workers):
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        options=[
            ('grpc.max_send_message_length', 256 * 1024 * 1024),
            ('grpc.max_receive_message_length', 256 * 1024 * 1024),
        ],
    )
    bff_pb2_grpc.add_BFFServicer_to_server(BFFServicer(), server)
    port = server.add_insecure_port(f'{ip}:{port}')
    server.start()
    print(f'BFF server started at {ip}:{port}')
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        ...


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run an bff service.')
    parser.add_argument('-i', '--ip', type=str, default='0.0.0.0', help='IP address to listen on.')
    parser.add_argument('-p', '--port', type=int, default=0, help='Port to listen on.')
    parser.add_argument('-w', '--work', type=int, default=10, help='Max workers.')
    args = parser.parse_args()
    bff_server(args.ip, args.port, args.work)
