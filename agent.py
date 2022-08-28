import argparse
from concurrent import futures
import json
import pickle

import grpc

from protos import agent_pb2
from protos import agent_pb2_grpc
from protos import types_pb2

from models import default_builder, RLModels


class AgentServicer(agent_pb2_grpc.AgentServicer):

    def __init__(self):
        self.__reset_all()

    def __reset_all(self):
        self.configs = None

        self.model = None

        self.sifunc = None
        self.oafunc = None
        self.rfunc = None

        self.__reset_args()

    def __reset_args(self):
        self.sifunc_args = {'states': None, 'inputs': None}
        self.oafunc_args = {'outputs': None, 'actions': None}
        self.rfunc_args = {
            'states': None,
            'inputs': None,
            'actions': None,
            'outputs': None,
            'next_states': None,
            'next_inputs': None,
            'done': False,
            'reward': 0,
        }

    def PingPong(self, request, context):
        return types_pb2.CommonResponse()

    def GetAgentConfig(self, request, context):
        configs = self.configs or agent_pb2.AgentConfig()
        return configs

    def SetAgentConfig(self, request, context):
        self.configs = request

        sifunc_src = request.states_inputs_func + '\ninputs = func(states)'
        self.sifunc = compile(sifunc_src, '', 'exec')
        oafunc_src = request.outputs_actions_func + '\nactions = func(outputs)'
        self.oafunc = compile(oafunc_src, '', 'exec')
        if self.configs.training:
            rfunc_src = request.reward_func + '\nreward = func(states, actions, next_states, done)'
            self.rfunc = compile(rfunc_src, '', 'exec')

        if request.builder:
            result = {}
            builder_src = request.builder + '\nnetworks = func()'
            exec(builder_src, result)
            networks = result['networks']
        else:
            structures = json.loads(request.structures)
            networks = default_builder(structures)
        hypers = json.loads(request.hypers)
        self.model = RLModels[request.type](
            training=request.training,
            networks=networks,
            **hypers,
        )

        return types_pb2.CommonResponse()

    def RstAgentConfig(self, request, context):
        self.__reset_all()
        return types_pb2.CommonResponse()

    def GetAgentMode(self, request, context):
        training = self.model.training if self.model is not None else False
        return agent_pb2.AgentMode(training=training)

    def SetAgentMode(self, request, context):
        if self.model is not None:
            self.model.training = request.training
        return types_pb2.CommonResponse()

    def GetAgentWeight(self, request, context):
        weights = pickle.dumps(self.model.get_weights()) if self.model is not None else bytes()
        return agent_pb2.AgentWeight(weights=weights)

    def SetAgentWeight(self, request, context):
        if self.model is not None:
            self.model.set_weights(pickle.loads(request.weights))
        return types_pb2.CommonResponse()

    def GetAgentBuffer(self, request, context):
        buffer = pickle.dumps(self.model.get_buffer()) if self.model is not None else bytes()
        return agent_pb2.AgentBuffer(buffer=buffer)

    def SetAgentBuffer(self, request, context):
        if self.model is not None:
            self.model.set_buffer(pickle.loads(request.buffer))
        return types_pb2.CommonResponse()

    def GetAgentStatus(self, request, context):
        status = json.dumps(self.model.get_status()) if self.model is not None else ''
        return agent_pb2.AgentStatus(status=status)

    def SetAgentStatus(self, request, context):
        if self.model is not None:
            self.model.set_status(json.loads(request.status))
        return types_pb2.CommonResponse()

    def GetAction(self, request_iterator, context):
        if self.model is not None:
            for request in request_iterator:
                info = json.loads(request.json)
                states, done = info['states'], info['done']
                self.sifunc_args['states'] = states
                exec(self.sifunc, self.sifunc_args)
                inputs = self.sifunc_args['inputs']
                outputs = self.model.react(inputs)
                self.oafunc_args['outputs'] = outputs
                exec(self.oafunc, self.oafunc_args)
                actions = self.oafunc_args['actions']
                if self.model.training:
                    if self.rfunc_args['states'] is None:
                        self.rfunc_args['states'] = states
                        self.rfunc_args['inputs'] = inputs
                        self.rfunc_args['actions'] = actions
                        self.rfunc_args['outputs'] = outputs
                    else:
                        self.rfunc_args['next_states'] = states
                        self.rfunc_args['next_inputs'] = inputs
                        self.rfunc_args['done'] = done
                        exec(self.rfunc, self.rfunc_args)
                        self.model.store(
                            states=self.rfunc_args['inputs'],
                            actions=self.rfunc_args['outputs'],
                            next_states=self.rfunc_args['next_inputs'],
                            reward=self.rfunc_args['reward'],
                            done=done,
                        )
                        self.model.train()
                        self.rfunc_args['states'] = self.rfunc_args['next_states']
                        self.rfunc_args['inputs'] = self.rfunc_args['next_inputs']
                        self.rfunc_args['actions'] = actions
                        self.rfunc_args['outputs'] = outputs
                yield types_pb2.JsonString(json=json.dumps({'actions': actions}))
        self.__reset_args()
        return types_pb2.JsonString(json=json.dumps(None))


def agent_server(ip, port, max_workers):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    agent_pb2_grpc.add_AgentServicer_to_server(AgentServicer(), server)
    server.add_insecure_port(f'{ip}:{port}')
    server.start()
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        ...


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run an agent service.')
    parser.add_argument('-i', '--ip', type=str, default='0.0.0.0', help='IP address to listen on.')
    parser.add_argument('-p', '--port', type=int, default=50051, help='Port to listen on.')
    parser.add_argument('-w', '--work', type=int, default=10, help='Max workers.')
    args = parser.parse_args()
    agent_server(args.ip, args.port, args.work)
