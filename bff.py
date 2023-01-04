import argparse
from concurrent import futures
import hashlib
import json

import grpc

from protos import agent_pb2_grpc
from protos import bff_pb2
from protos import bff_pb2_grpc
from protos import simenv_pb2
from protos import simenv_pb2_grpc
from protos import types_pb2


class BFFServicer(bff_pb2_grpc.BFFServicer):

    def __init__(self):
        self.reset()

    def reset(self):
        self.services = {}

        self.data_config = None
        self.route_config = None
        self.sim_term_func = None

        self.agents = {}
        self.simenvs = {}

    def unknown_id(self, id, context):
        context.abort(grpc.StatusCode.INVALID_ARGUMENT, f'Unknown service id: {id}')

    def ResetServer(self, request, context):
        self.reset()
        return types_pb2.CommonResponse()

    def RegisterService(self, request, context):
        ids = []
        for service in request.services:
            key = f'{service.type}-{service.name} {service.ip}:{service.port}'
            sha256 = hashlib.sha256()
            sha256.update(key.encode('utf-8'))
            id = sha256.hexdigest()[:8]
            self.services[id] = service
            ids.append(id)
            if service.type == bff_pb2.ServiceInfo.Type.AGENT:
                channel = grpc.insecure_channel(f'{service.ip}:{service.port}')
                self.agents[id] = agent_pb2_grpc.AgentStub(channel)
            elif service.type == bff_pb2.ServiceInfo.Type.SIMENV:
                channel = grpc.insecure_channel(f'{service.ip}:{service.port}')
                self.simenvs[id] = simenv_pb2_grpc.SimenvStub(channel)
            else:
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, f'Unknown service type: {service.type}')
        return bff_pb2.ServiceIdList(ids=ids)

    def UnRegisterService(self, request, context):
        ids = request.ids if len(request.ids) > 0 else list(self.services.keys())
        for id in ids:
            del self.services[id]
            if id in self.agents:
                self.agents[id].ResetService(types_pb2.CommonRequest())
                del self.agents[id]
            elif id in self.simenvs:
                self.simenvs[id].ResetService(types_pb2.CommonRequest())
                del self.simenvs[id]
            else:
                self.unknown_id(id, context)
        return types_pb2.CommonResponse()

    def GetServiceInfo(self, request, context):
        services = {}
        ids = request.ids if len(request.ids) > 0 else list(self.services.keys())
        for id in ids:
            if id in self.services:
                services[id] = self.services[id]
        return bff_pb2.ServiceInfoMap(services=services)

    def SetServiceInfo(self, request, context):
        for id in request.services:
            if id in self.services:
                self.services[id] = request.services[id]
        return types_pb2.CommonResponse()

    def GetDataConfig(self, request, context):
        return self.data_config

    def SetDataConfig(self, request, context):
        self.data_config = request
        return types_pb2.CommonResponse()

    def GetRouteConfig(self, request, context):
        return self.route_config

    def SetRouteConfig(self, request, context):
        self.route_config = request
        sdfunc_src = request.sim_term_func + '\nterminated = func(states)'
        self.sim_term_func = compile(sdfunc_src, '', 'exec')
        return types_pb2.CommonResponse()

    def ProxyChat(self, request_iterator, context):
        id = ''
        for datum in context.invocation_metadata():
            if datum[0] == 'id':
                id = datum[1]
                break

        if id == '':
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, 'Missing id in metadata!')
        elif id not in self.simenvs:
            self.unknown_id(id, context)
        else:
            route = self.route_config.routes[id]
            for request in request_iterator:
                info = json.loads(request.json)
                exec(self.sim_term_func, info)
                states, terminated, truncated, actions = info['states'], info['terminated'], info['truncated'], {}
                for id_ in route.configs:
                    states_ = {}
                    for model in route.configs[id_].models:
                        states_[model] = states[model]
                    agent_req = types_pb2.JsonString(json=json.dumps({
                        'states': states_,
                        'terminated': terminated,
                        'truncated': truncated,
                    }))
                    agent_res = self.agents[id_].GetAction(agent_req)
                    actions_ = json.loads(agent_res.json)['actions']
                    actions.update(actions_)
                response = types_pb2.JsonString(json=json.dumps({'actions': actions}))
                if terminated:
                    self.simenvs[id].SimControl(simenv_pb2.SimCmd(type=simenv_pb2.SimCmd.Type.EPISODE, params=json.dumps({})))
                    return response
                else:
                    yield response

    def ResetService(self, request, context):
        ids = request.ids if len(request.ids) > 0 else list(self.services.keys())
        for id in ids:
            if id in self.agents:
                self.agents[id].ResetService(types_pb2.CommonRequest())
            elif id in self.simenvs:
                self.simenvs[id].ResetService(types_pb2.CommonRequest())
            else:
                self.unknown_id(id, context)
        return types_pb2.CommonResponse()

    def QueryService(self, request, context):
        states = {}
        ids = request.ids if len(request.ids) > 0 else list(self.services.keys())
        for id in ids:
            if id in self.agents:
                states[id] = self.agents[id].QueryService(types_pb2.CommonRequest())
            elif id in self.simenvs:
                states[id] = self.simenvs[id].QueryService(types_pb2.CommonRequest())
            else:
                self.unknown_id(id, context)
        return bff_pb2.ServiceStateMap(states=states)

    def GetAgentConfig(self, request, context):
        configs = {}
        ids = request.ids if len(request.ids) > 0 else list(self.agents.keys())
        for id in ids:
            if id in self.agents:
                configs[id] = self.agents[id].GetAgentConfig(types_pb2.CommonRequest())
            else:
                self.unknown_id(id, context)
        return bff_pb2.AgentConfigMap(configs=configs)

    def SetAgentConfig(self, request, context):
        for id in request.configs:
            if id in self.agents:
                self.agents[id].SetAgentConfig(request.configs[id])
            else:
                self.unknown_id(id, context)
        return types_pb2.CommonResponse()

    def GetAgentMode(self, request, context):
        modes = {}
        ids = request.ids if len(request.ids) > 0 else list(self.agents.keys())
        for id in ids:
            if id in self.agents:
                modes[id] = self.agents[id].GetAgentMode(types_pb2.CommonRequest())
            else:
                self.unknown_id(id, context)
        return bff_pb2.AgentModeMap(modes=modes)

    def SetAgentMode(self, request, context):
        for id in request.modes:
            if id in self.agents:
                self.agents[id].SetAgentMode(request.modes[id])
            else:
                self.unknown_id(id, context)
        return types_pb2.CommonResponse()

    def GetModelWeights(self, request, context):
        weights = {}
        ids = request.ids if len(request.ids) > 0 else list(self.agents.keys())
        for id in ids:
            if id in self.agents:
                weights[id] = self.agents[id].GetModelWeights(types_pb2.CommonRequest())
            else:
                self.unknown_id(id, context)
        return bff_pb2.ModelWeightsMap(weights=weights)

    def SetModelWeights(self, request, context):
        for id in request.weights:
            if id in self.agents:
                self.agents[id].SetModelWeights(request.weights[id])
            else:
                self.unknown_id(id, context)
        return types_pb2.CommonResponse()

    def GetModelBuffer(self, request, context):
        buffers = {}
        ids = request.ids if len(request.ids) > 0 else list(self.agents.keys())
        for id in ids:
            if id in self.agents:
                buffers[id] = self.agents[id].GetModelBuffer(types_pb2.CommonRequest())
            else:
                self.unknown_id(id, context)
        return bff_pb2.ModelBufferMap(buffers=buffers)

    def SetModelBuffer(self, request, context):
        for id in request.buffers:
            if id in self.agents:
                self.agents[id].SetModelBuffer(request.buffers[id])
            else:
                self.unknown_id(id, context)
        return types_pb2.CommonResponse()

    def GetModelStatus(self, request, context):
        status = {}
        ids = request.ids if len(request.ids) > 0 else list(self.agents.keys())
        for id in ids:
            if id in self.agents:
                status[id] = self.agents[id].GetModelStatus(types_pb2.CommonRequest())
            else:
                self.unknown_id(id, context)
        return bff_pb2.ModelStatusMap(status=status)

    def SetModelStatus(self, request, context):
        for id in request.status:
            if id in self.agents:
                self.agents[id].SetModelStatus(request.status[id])
            else:
                self.unknown_id(id, context)
        return types_pb2.CommonResponse()

    def GetSimenvConfig(self, request, context):
        configs = {}
        ids = request.ids if len(request.ids) > 0 else list(self.simenvs.keys())
        for id in ids:
            if id in self.simenvs:
                configs[id] = self.simenvs[id].GetSimenvConfig(types_pb2.CommonRequest())
            else:
                self.unknown_id(id, context)
        return bff_pb2.SimenvConfigMap(configs=configs)

    def SetSimenvConfig(self, request, context):
        for id in request.configs:
            if id in self.simenvs:
                self.simenvs[id].SetSimenvConfig(request.configs[id])
            else:
                self.unknown_id(id, context)
        return types_pb2.CommonResponse()

    def SimControl(self, request, context):
        for id in request.cmds:
            if id in self.simenvs:
                self.simenvs[id].SimControl(request.cmds[id])
            else:
                self.unknown_id(id, context)
        return types_pb2.CommonResponse()

    def SimMonitor(self, request, context):
        infos = {}
        ids = request.ids if len(request.ids) > 0 else list(self.simenvs.keys())
        for id in ids:
            if id in self.simenvs:
                infos[id] = self.simenvs[id].SimMonitor(types_pb2.CommonRequest())
            else:
                self.unknown_id(id, context)
        return bff_pb2.SimInfoMap(infos=infos)


def bff_server(ip, port, max_workers, max_msg_len):
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        options=[
            ('grpc.max_send_message_length', max_msg_len * 1024 * 1024),
            ('grpc.max_receive_message_length', max_msg_len * 1024 * 1024),
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
    parser.add_argument('-w', '--work', type=int, default=10, help='Max workers in thread pool.')
    parser.add_argument('-m', '--msglen', type=int, default=256, help='Max message length in MB.')
    args = parser.parse_args()
    bff_server(args.ip, args.port, args.work, args.msglen)
