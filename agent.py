import argparse
from concurrent import futures
import json
import pickle

import grpc

from protos import agent_pb2
from protos import agent_pb2_grpc
from protos import types_pb2

from models import RLModels
from models.utils import default_builder


class AgentServicer(agent_pb2_grpc.AgentServicer):

    def __init__(self):
        self.reset_all()

    def reset_all(self):
        self.state = types_pb2.ServiceState.State.UNINITED

        self.configs = None

        self.model = None

        self.sifunc = None
        self.oafunc = None
        self.rfunc = None

        self.reset_args()

    def reset_args(self):
        self.func_cache = {}
        self.sifunc_args = {'states': None, 'inputs': None, 'cache': self.func_cache}
        self.oafunc_args = {'outputs': None, 'actions': None, 'cache': self.func_cache}
        self.rfunc_args = {
            'states': None,
            'inputs': None,
            'actions': None,
            'outputs': None,
            'next_states': None,
            'next_inputs': None,
            'reward': 0,
            'terminated': False,
            'cache': self.func_cache,
        }

    def check_state(self, context):
        if self.state == types_pb2.ServiceState.State.UNINITED:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, 'Service not inited')

    def ResetService(self, request, context):
        if self.model is not None:
            self.model.close()
        self.reset_all()
        return types_pb2.CommonResponse()

    def QueryService(self, request, context):
        return types_pb2.ServiceState(state=self.state)

    def GetAgentConfig(self, request, context):
        self.check_state(context)
        return self.configs

    def SetAgentConfig(self, request, context):
        sifunc_src = request.states_inputs_func + '\ninputs = func(states)'
        self.sifunc = compile(sifunc_src, '', 'exec')
        oafunc_src = request.outputs_actions_func + '\nactions = func(outputs)'
        self.oafunc = compile(oafunc_src, '', 'exec')
        if request.training:
            rfunc_src = request.reward_func + '\nreward = func(states, inputs, actions, outputs,\
                 next_states, next_inputs, terminated)'

            self.rfunc = compile(rfunc_src, '', 'exec')

        hypers = json.loads(request.hypers)
        if request.builder:
            result = {}
            builder_src = request.builder + '\nnetworks = func()'
            exec(builder_src, result)
            networks = result['networks']
        else:
            structures = json.loads(request.structures)
            networks = default_builder(structures)
        self.model = RLModels[request.type](training=request.training, networks=networks, **hypers)

        self.state = types_pb2.ServiceState.State.INITED
        self.configs = request

        return types_pb2.CommonResponse()

    def GetAgentMode(self, request, context):
        self.check_state(context)
        return agent_pb2.AgentMode(training=self.model.training)

    def SetAgentMode(self, request, context):
        self.check_state(context)
        self.model.training = request.training
        return types_pb2.CommonResponse()

    def GetModelWeights(self, request, context):
        self.check_state(context)
        weights = pickle.dumps(self.model.get_weights())
        return agent_pb2.ModelWeights(weights=weights)

    def SetModelWeights(self, request, context):
        self.check_state(context)
        self.model.set_weights(pickle.loads(request.weights))
        return types_pb2.CommonResponse()

    def GetModelBuffer(self, request, context):
        self.check_state(context)
        buffer = pickle.dumps(self.model.get_buffer())
        return agent_pb2.ModelBuffer(buffer=buffer)

    def SetModelBuffer(self, request, context):
        self.check_state(context)
        self.model.set_buffer(pickle.loads(request.buffer))
        return types_pb2.CommonResponse()

    def GetModelStatus(self, request, context):
        self.check_state(context)
        status = json.dumps(self.model.get_status())
        return agent_pb2.ModelStatus(status=status)

    def SetModelStatus(self, request, context):
        self.check_state(context)
        self.model.set_status(json.loads(request.status))
        return types_pb2.CommonResponse()

    def GetAction(self, request, context):
        self.check_state(context)

        info = json.loads(request.json)
        states, terminated, truncated = info['states'], info['terminated'], info['truncated']
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
                self.rfunc_args['reward'] = 0.0
                self.rfunc_args['terminated'] = terminated
                exec(self.rfunc, self.rfunc_args)
                self.model.store(
                    states=self.rfunc_args['inputs'],
                    actions=self.rfunc_args['outputs'],
                    next_states=self.rfunc_args['next_inputs'],
                    reward=self.rfunc_args['reward'],
                    terminated=terminated,
                    truncated=truncated,
                )
                self.rfunc_args['states'] = self.rfunc_args['next_states']
                self.rfunc_args['inputs'] = self.rfunc_args['next_inputs']
                self.rfunc_args['actions'] = actions
                self.rfunc_args['outputs'] = outputs
                self.model.train()
        if terminated or truncated:
            self.reset_args()

        return types_pb2.JsonString(json=json.dumps({'actions': actions}))


def agent_server(ip, port, max_workers, max_msg_len):
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        options=[
            ('grpc.max_send_message_length', max_msg_len * 1024 * 1024),
            ('grpc.max_receive_message_length', max_msg_len * 1024 * 1024),
        ],
    )
    agent_pb2_grpc.add_AgentServicer_to_server(AgentServicer(), server)
    port = server.add_insecure_port(f'{ip}:{port}')
    server.start()
    print(f'Agent server started at {ip}:{port}')
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        ...


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run an agent service.')
    parser.add_argument('-i', '--ip', type=str, default='0.0.0.0', help='IP address to listen on.')
    parser.add_argument('-p', '--port', type=int, default=0, help='Port to listen on.')
    parser.add_argument('-w', '--work', type=int, default=10, help='Max workers in thread pool.')
    parser.add_argument('-m', '--msglen', type=int, default=256, help='Max message length in MB.')
    args = parser.parse_args()
    agent_server(args.ip, args.port, args.work, args.msglen)
