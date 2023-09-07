import argparse
import asyncio
from concurrent import futures
import importlib
import io
import json
import logging
import os
import pickle
import shutil
import signal
import sys
import zipfile

from google.protobuf import json_format
import grpc

from protos import agent_pb2
from protos import agent_pb2_grpc
from protos import types_pb2

from hooks import AgentHooks
from models import RLModels

LOGGER_NAME = 'agent'


class AgentServicer(agent_pb2_grpc.AgentServicer):

    def __init__(self):
        self.reset()
        self.copy()

    def reset(self):
        self.state = types_pb2.ServiceState.State.UNINITED

        self.configs = None

        self.model = None

        self.sifunc = None
        self.oafunc = None
        self.rewfunc = None

        self.training = None

        self.hooks = []
        self.shared = {}

        self.episodes = 0
        self.react_steps = 0
        self.train_steps = 0

    def copy(self):
        source, target = 'models', 'data/models'
        if not os.path.exists(target):
            os.makedirs(target, exist_ok=True)
        for file in ['base.py']:
            shutil.copy(os.path.join(source, file), os.path.join(target, file))
        with open(f'{target}/__init__.py', 'w') as f:
            f.write('')

    async def check_state(self, context):
        if self.state == types_pb2.ServiceState.State.UNINITED:
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, 'Service not inited')

    async def ResetService(self, request, context):
        self.reset()
        return types_pb2.CommonResponse()

    async def QueryService(self, request, context):
        return types_pb2.ServiceState(state=self.state)

    async def GetAgentConfig(self, request, context):
        await self.check_state(context)
        return self.configs

    async def SetAgentConfig(self, request, context):
        try:
            sisrc = request.sifunc + '\ninputs = func(states)'
            self.sifunc = compile(sisrc, 'states_to_inputs', 'exec')
            oasrc = request.oafunc + '\nactions = func(outputs)'
            self.oafunc = compile(oasrc, 'outputs_to_actions', 'exec')
            rewsrc = request.rewfunc + '\nreward = func(states, inputs, actions, outputs,\
                next_states, next_inputs, terminated, truncated, reward)'

            self.rewfunc = compile(rewsrc, 'reward', 'exec')
        except SyntaxError as e:
            message = f'Invalid {e.filename} function, error in line {e.lineno}, column {e.offset}, {e.text}'
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, message)

        if request.name in RLModels:
            model_class = RLModels[request.name]
        else:
            try:
                module = importlib.import_module(f'data.models.{request.name.lower()}')
                importlib.reload(module)
                model_class = getattr(module, request.name)
            except Exception:
                message = f'{request.name} model not supported'
                await context.abort(grpc.StatusCode.INVALID_ARGUMENT, message)

        try:
            hypers = json.loads(request.hypers) if request.hypers else {}
            self.model = model_class(training=request.training, **hypers)
        except Exception as e:
            message = f'Invalid hypers for {request.name} model, info: {e}'
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, message)

        try:
            for hook in request.hooks:
                if hook.name not in AgentHooks:
                    message = f'{hook.name} hook not supported'
                    await context.abort(grpc.StatusCode.INVALID_ARGUMENT, message)
                else:
                    args = json.loads(hook.args) if hook.args else {}
                    self.hooks.append(AgentHooks[hook.name](self.model, **args))
        except Exception as e:
            message = f'Invalid args for {hook.name} hook, info: {e}'
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, message)

        self.state = types_pb2.ServiceState.State.INITED
        self.configs = request
        self.training = request.training

        return types_pb2.CommonResponse()

    async def GetAgentMode(self, request, context):
        await self.check_state(context)
        return agent_pb2.AgentMode(training=self.training)

    async def SetAgentMode(self, request, context):
        await self.check_state(context)
        if not self.configs.training:
            message = 'Cannot change training mode when training is initially set to False'
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, message)
        self.training = request.training
        return types_pb2.CommonResponse()

    async def GetModelWeights(self, request, context):
        await self.check_state(context)
        weights = pickle.dumps(self.model.get_weights())
        return agent_pb2.ModelWeights(weights=weights)

    async def SetModelWeights(self, request, context):
        await self.check_state(context)
        try:
            self.model.set_weights(pickle.loads(request.weights))
        except Exception as e:
            message = f'Invalid weights for {self.configs.name} model, info: {e}'
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, message)
        return types_pb2.CommonResponse()

    async def GetModelBuffer(self, request, context):
        await self.check_state(context)
        buffer = pickle.dumps(self.model.get_buffer())
        return agent_pb2.ModelBuffer(buffer=buffer)

    async def SetModelBuffer(self, request, context):
        await self.check_state(context)
        try:
            self.model.set_buffer(pickle.loads(request.buffer))
        except Exception as e:
            message = f'Invalid buffer for {self.configs.name} model, info: {e}'
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, message)
        return types_pb2.CommonResponse()

    async def GetModelStatus(self, request, context):
        await self.check_state(context)
        status = json.dumps(self.model.get_status())
        return agent_pb2.ModelStatus(status=status)

    async def SetModelStatus(self, request, context):
        await self.check_state(context)
        try:
            status = json.loads(request.status) if request.status else {}
            self.model.set_status(status)
        except Exception as e:
            message = f'Invalid status for {self.configs.name} model, info: {e}'
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, message)
        return types_pb2.CommonResponse()

    async def GetAction(self, request_iterator, context):
        await self.check_state(context)

        enabled = self.training
        self.model.training = self.training

        if enabled:
            for hook in self.hooks:
                hook.before_episode(self.episodes, self.shared)

        init, done = False, False
        siargs, oaargs, rewargs, caches = {}, {}, {}, {}

        task = None
        loop = asyncio.get_running_loop()
        async for request in request_iterator:
            if task is not None:
                await task
                task = None

            states, terminated, truncated, reward = self.parse_state(request)

            done = terminated or truncated

            if not done:
                if enabled:
                    for hook in self.hooks:
                        hook.before_react(self.react_steps)

            siargs['caches'] = caches
            siargs['states'] = states
            exec(self.sifunc, siargs)
            inputs = siargs['inputs']

            if not done:
                outputs = self.model.react(inputs)

                oaargs['caches'] = caches
                oaargs['outputs'] = outputs
                exec(self.oafunc, oaargs)
                actions = oaargs['actions']

                if enabled:
                    for hook in reversed(self.hooks):
                        hook.after_react(self.react_steps, siargs, oaargs)

                    self.react_steps += 1

            if init:
                if enabled:
                    rewargs['next_states'] = states
                    rewargs['next_inputs'] = inputs
                    rewargs['terminated'] = terminated
                    rewargs['truncated'] = truncated
                    rewargs['reward'] = reward
                    exec(self.rewfunc, rewargs)

                    for hook in self.hooks:
                        hook.react2train(rewargs)

                    if self.model.training:
                        self.model.store(
                            states=rewargs['inputs'],
                            actions=rewargs['outputs'],
                            next_states=rewargs['next_inputs'],
                            reward=rewargs['reward'],
                            terminated=rewargs['terminated'],
                            truncated=rewargs['truncated'],
                        )
                        task = asyncio.create_task(self.learn(loop))
            else:
                init = True

            if not done:
                if enabled:
                    rewargs['caches'] = caches
                    rewargs['states'] = states
                    rewargs['inputs'] = inputs
                    rewargs['actions'] = actions
                    rewargs['outputs'] = outputs

                response = self.wrap_action(actions)
                yield response
            else:
                if task is not None:
                    await task
                    task = None
                break

        if enabled:
            for hook in reversed(self.hooks):
                hook.after_episode(self.episodes, self.shared)

            self.episodes += 1

    async def learn(self, loop):
        for hook in self.hooks:
            hook.before_train(self.train_steps)

        infos = await loop.run_in_executor(None, self.model.train)

        for hook in reversed(self.hooks):
            hook.after_train(self.train_steps, infos)

        self.train_steps += 1

    def parse_param(self, param):
        if 'vdouble' in param:
            return param['vdouble']
        elif 'vint32' in param:
            return param['vint32']
        elif 'vbool' in param:
            return param['vbool']
        elif 'vstring' in param:
            return param['vstring']
        elif 'varray' in param:
            return [self.parse_param(item) for item in param['varray']['items']]
        elif 'vstruct' in param:
            return {field: self.parse_param(value) for field, value in param['vstruct']['fields'].items()}

    def parse_state(self, req):
        req = json_format.MessageToDict(req, preserving_proto_field_name=True, including_default_value_fields=True)
        states = req['states']
        for k in states:
            states[k] = states[k]['entities']
            for i in range(len(states[k])):
                states[k][i] = states[k][i]['params']
                for j in states[k][i]:
                    states[k][i][j] = self.parse_param(states[k][i][j])
        return states, req['terminated'], req['truncated'], req['reward']

    def wrap_param(self, param):
        if isinstance(param, bool):
            return {'vbool': param}
        elif isinstance(param, float):
            return {'vdouble': param}
        elif isinstance(param, int):
            return {'vint32': param}
        elif isinstance(param, str):
            return {'vstring': param}
        elif isinstance(param, list):
            return {'varray': {'items': [self.wrap_param(item) for item in param]}}
        elif isinstance(param, dict):
            return {'vstruct': {'fields': {k: self.wrap_param(v) for k, v in param.items()}}}

    def wrap_action(self, actions):
        for k in actions:
            actions[k] = {'entities': actions[k]}
            for i in range(len(actions[k]['entities'])):
                actions[k]['entities'][i] = {'params': actions[k]['entities'][i]}
                for j in actions[k]['entities'][i]['params']:
                    actions[k]['entities'][i]['params'][j] = self.wrap_param(actions[k]['entities'][i]['params'][j])
        return json_format.ParseDict({'actions': actions}, types_pb2.SimAction())

    async def Call(self, request, context):
        name, dstr, dbin = request.name, request.dstr, request.dbin
        if name.startswith('@'):
            name, dstr, dbin = self.call(name, dstr, dbin)
        else:
            await self.check_state(context)
            name, dstr, dbin = self.model.call(name, dstr, dbin)
        return types_pb2.CallData(name=name, dstr=dstr, dbin=dbin)

    def call(self, name, dstr, dbin):
        if name == '@custom-model':
            with zipfile.ZipFile(io.BytesIO(dbin), 'r') as zip_ref:
                zip_ref.extractall('data/models')
        return name, 'OK', b''


_cleanup_coroutines = []


async def agent_server(host, port, max_workers, max_msg_len):
    logger = logging.getLogger(LOGGER_NAME)
    server = grpc.aio.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        options=[
            ('grpc.max_send_message_length', max_msg_len * 1024 * 1024),
            ('grpc.max_receive_message_length', max_msg_len * 1024 * 1024),
        ],
    )
    agent_pb2_grpc.add_AgentServicer_to_server(AgentServicer(), server)
    port = server.add_insecure_port(f'{host}:{port}')
    await server.start()
    logger.info(f'Agent server started at {host}:{port}')

    async def grace_exit(*_):
        logger.info('Agent server stopping...')
        await server.stop(0)

    _cleanup_coroutines.append(grace_exit())
    await server.wait_for_termination()
    logger.info('Agent server stopped.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run an agent service.')
    parser.add_argument('-i', '--host', type=str, default='0.0.0.0', help='Host to listen on.')
    parser.add_argument('-p', '--port', type=int, default=0, help='Port to listen on.')
    parser.add_argument('-w', '--worker', type=int, default=10, help='Max workers in thread pool.')
    parser.add_argument('-m', '--msglen', type=int, default=256, help='Max message length in MB.')
    parser.add_argument('-l', '--loglvl', type=str, default='info', help='Log level defined in `logging`.')
    args = parser.parse_args()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(lineno)d - %(levelname)s - %(message)s'))
    logger = logging.getLogger(LOGGER_NAME)
    logger.addHandler(handler)
    logger.setLevel(args.loglvl.upper())

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    server = agent_server(args.host, args.port, args.worker, args.msglen)
    if sys.platform == "linux":
        try:
            for signame in ('SIGINT', 'SIGTERM'):
                loop.add_signal_handler(getattr(signal, signame), lambda: asyncio.create_task(_cleanup_coroutines[0]))
            loop.run_until_complete(server)
        except (Exception, KeyboardInterrupt):
            ...
        finally:
            loop.close()
    else:
        try:
            loop.run_until_complete(server)
        except (Exception, KeyboardInterrupt):
            ...
            loop.run_until_complete(*_cleanup_coroutines)
        finally:
            loop.close()
