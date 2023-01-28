import argparse
from concurrent import futures
import json
import pickle

from google.protobuf import json_format
import grpc

from protos import agent_pb2
from protos import agent_pb2_grpc
from protos import types_pb2

from models import RLModels
from models.utils import default_builder


class AgentServicer(agent_pb2_grpc.AgentServicer):

    def __init__(self):
        self.reset()

    def reset(self):
        self.state = types_pb2.ServiceState.State.UNINITED

        self.configs = None

        self.model = None

        self.sifunc = None
        self.oafunc = None
        self.rfunc = None

    def check_state(self, context):
        if self.state == types_pb2.ServiceState.State.UNINITED:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, 'Service not inited')

    def ResetService(self, request, context):
        if self.model is not None:
            self.model.close()
        self.reset()
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
                 next_states, next_inputs, terminated, truncated)'

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

    def GetAction(self, request_iterator, context):
        self.check_state(context)
        sifunc_args = {'states': None, 'inputs': None}
        oafunc_args = {'outputs': None, 'actions': None}
        rfunc_args = {
            'states': None,
            'inputs': None,
            'actions': None,
            'outputs': None,
            'next_states': None,
            'next_inputs': None,
            'reward': 0,
            'terminated': False,
            'truncated': False,
        }
        for request in request_iterator:
            sim_state = json_format.MessageToDict(request)
            states, terminated, truncated = sim_state['states'], sim_state['terminated'], sim_state['truncated']
            sifunc_args['states'] = states
            exec(self.sifunc, sifunc_args)
            inputs = sifunc_args['inputs']
            outputs = self.model.react(inputs)
            oafunc_args['outputs'] = outputs
            exec(self.oafunc, oafunc_args)
            actions = oafunc_args['actions']

            if self.model.training:
                if rfunc_args['states'] is None:
                    rfunc_args['states'] = states
                    rfunc_args['inputs'] = inputs
                    rfunc_args['actions'] = actions
                    rfunc_args['outputs'] = outputs
                else:
                    rfunc_args['next_states'] = states
                    rfunc_args['next_inputs'] = inputs
                    rfunc_args['terminated'] = terminated
                    rfunc_args['truncated'] = truncated
                    exec(self.rfunc, rfunc_args)
                    self.model.store(
                        states=rfunc_args['inputs'],
                        actions=rfunc_args['outputs'],
                        next_states=rfunc_args['next_inputs'],
                        reward=rfunc_args['reward'],
                        terminated=terminated,
                        truncated=truncated,
                    )
                    rfunc_args['states'] = rfunc_args['next_states']
                    rfunc_args['inputs'] = rfunc_args['next_inputs']
                    rfunc_args['actions'] = actions
                    rfunc_args['outputs'] = outputs
                    self.model.train()

            sim_action = types_pb2.SimAction()
            response = json_format.ParseDict({'actions': actions}, sim_action)
            if terminated or truncated:
                return response
            else:
                yield response


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
