import argparse
from concurrent import futures
import json
import pickle
import signal

from google.protobuf import json_format
import grpc

from protos import agent_pb2
from protos import agent_pb2_grpc
from protos import types_pb2

from models import RLModels


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
        self.reset()
        return types_pb2.CommonResponse()

    def QueryService(self, request, context):
        return types_pb2.ServiceState(state=self.state)

    def GetAgentConfig(self, request, context):
        self.check_state(context)
        return self.configs

    def SetAgentConfig(self, request, context):
        try:
            sifunc_src = request.sifunc + '\ninputs = func(states)'
            self.sifunc = compile(sifunc_src, 'states_to_inputs', 'exec')
            oafunc_src = request.oafunc + '\nactions = func(outputs)'
            self.oafunc = compile(oafunc_src, 'outputs_to_actions', 'exec')
            if request.training:
                rfunc_src = request.rewfunc + '\nreward = func(states, inputs, actions, outputs,\
                    next_states, next_inputs, terminated, truncated)'

                self.rfunc = compile(rfunc_src, 'reward', 'exec')
        except SyntaxError as e:
            message = f'Invalid {e.filename} function, error in line {e.lineno}, column {e.offset}, {e.text}'
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, message)

        try:
            hypers = json.loads(request.hypers)
            self.model = RLModels[request.type](training=request.training, **hypers)
        except Exception as e:
            message = f'Invalid hypers for {request.type} model, info: {e}'
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, message)

        self.state = types_pb2.ServiceState.State.INITED
        self.configs = request

        return types_pb2.CommonResponse()

    def GetAgentMode(self, request, context):
        self.check_state(context)
        return agent_pb2.AgentMode(training=self.model.training)

    def SetAgentMode(self, request, context):
        self.check_state(context)
        if not self.configs.training:
            message = 'Cannot change training mode when training is initially set to False'
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, message)
        self.model.training = request.training
        return types_pb2.CommonResponse()

    def GetModelWeights(self, request, context):
        self.check_state(context)
        weights = pickle.dumps(self.model.get_weights())
        return agent_pb2.ModelWeights(weights=weights)

    def SetModelWeights(self, request, context):
        self.check_state(context)
        try:
            self.model.set_weights(pickle.loads(request.weights))
        except Exception as e:
            message = f'Invalid weights for {self.configs.type} model, info: {e}'
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, message)
        return types_pb2.CommonResponse()

    def GetModelBuffer(self, request, context):
        self.check_state(context)
        buffer = pickle.dumps(self.model.get_buffer())
        return agent_pb2.ModelBuffer(buffer=buffer)

    def SetModelBuffer(self, request, context):
        self.check_state(context)
        try:
            self.model.set_buffer(pickle.loads(request.buffer))
        except Exception as e:
            message = f'Invalid buffer for {self.configs.type} model, info: {e}'
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, message)
        return types_pb2.CommonResponse()

    def GetModelStatus(self, request, context):
        self.check_state(context)
        status = json.dumps(self.model.get_status())
        return agent_pb2.ModelStatus(status=status)

    def SetModelStatus(self, request, context):
        self.check_state(context)
        try:
            self.model.set_status(json.loads(request.status))
        except Exception as e:
            message = f'Invalid status for {self.configs.type} model, info: {e}'
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, message)
        return types_pb2.CommonResponse()

    def GetAction(self, request_iterator, context):
        self.check_state(context)
        global_caches = {}
        sifunc_args = {'states': None, 'inputs': None, 'caches': global_caches}
        oafunc_args = {'outputs': None, 'actions': None, 'caches': global_caches}
        rfunc_args = {
            'states': None,
            'inputs': None,
            'actions': None,
            'outputs': None,
            'next_states': None,
            'next_inputs': None,
            'reward': None,
            'terminated': False,
            'truncated': False,
        }
        for request in request_iterator:
            states, terminated, truncated = self.parse_state(request)

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

            response = self.wrap_action(actions)
            if terminated or truncated:
                return response
            else:
                yield response

    def parse_param(self, param):
        if 'double_value' in param:
            return param['double_value']
        elif 'int32_value' in param:
            return param['int32_value']
        elif 'bool_value' in param:
            return param['bool_value']
        elif 'string_value' in param:
            return param['string_value']
        elif 'array_value' in param:
            return [self.parse_param(item) for item in param['array_value']['items']]
        elif 'struct_value' in param:
            return {field: self.parse_param(value) for field, value in param['struct_value']['fields'].items()}

    def parse_state(self, req):
        req = json_format.MessageToDict(req, preserving_proto_field_name=True, including_default_value_fields=True)
        states, terminated, truncated = req['states'], req['terminated'], req['truncated']
        for k in states:
            states[k] = states[k]['entities']
            for i in range(len(states[k])):
                states[k][i] = states[k][i]['params']
                for j in states[k][i]:
                    states[k][i][j] = self.parse_param(states[k][i][j])
        return states, terminated, truncated

    def wrap_param(self, param):
        if isinstance(param, bool):
            return {'bool_value': param}
        elif isinstance(param, float):
            return {'double_value': param}
        elif isinstance(param, int):
            return {'int32_value': param}
        elif isinstance(param, str):
            return {'string_value': param}
        elif isinstance(param, list):
            return {'array_value': {'items': [self.wrap_param(item) for item in param]}}
        elif isinstance(param, dict):
            return {'struct_value': {'fields': {k: self.wrap_param(v) for k, v in param.items()}}}

    def wrap_action(self, actions):
        for k in actions:
            actions[k] = {'entities': actions[k]}
            for i in range(len(actions[k]['entities'])):
                actions[k]['entities'][i] = {'params': actions[k]['entities'][i]}
                for j in actions[k]['entities'][i]['params']:
                    actions[k]['entities'][i]['params'][j] = self.wrap_param(actions[k]['entities'][i]['params'][j])
        return json_format.ParseDict({'actions': actions}, types_pb2.SimAction())

    def Call(self, request, context):
        self.check_state(context)
        identity, str_data, bin_data = self.model.call(request.identity, request.str_data, request.bin_data)
        return types_pb2.CallData(identity=identity, str_data=str_data, bin_data=bin_data)


def agent_server(host, port, max_workers, max_msg_len):
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        options=[
            ('grpc.max_send_message_length', max_msg_len * 1024 * 1024),
            ('grpc.max_receive_message_length', max_msg_len * 1024 * 1024),
        ],
    )
    agent_pb2_grpc.add_AgentServicer_to_server(AgentServicer(), server)
    port = server.add_insecure_port(f'{host}:{port}')
    server.start()
    print(f'Agent server started at {host}:{port}')

    def grace_exit(*_):
        print('Agent service stopping...')
        evt = server.stop(0)
        evt.wait(1)

    signal.signal(signal.SIGINT, grace_exit)
    signal.signal(signal.SIGTERM, grace_exit)
    server.wait_for_termination()
    print('Agent server stopped.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run an agent service.')
    parser.add_argument('-i', '--host', type=str, default='0.0.0.0', help='Host to listen on.')
    parser.add_argument('-p', '--port', type=int, default=0, help='Port to listen on.')
    parser.add_argument('-w', '--worker', type=int, default=10, help='Max workers in thread pool.')
    parser.add_argument('-m', '--msglen', type=int, default=256, help='Max message length in MB.')
    args = parser.parse_args()
    agent_server(args.host, args.port, args.worker, args.msglen)
