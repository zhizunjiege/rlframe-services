import argparse
from concurrent import futures
import json
import logging
import signal

import grpc

from protos import simenv_pb2
from protos import simenv_pb2_grpc
from protos import types_pb2

from engines import SimEngines


class SimenvServicer(simenv_pb2_grpc.SimenvServicer):

    def __init__(self):
        self.reset()

    def reset(self):
        self.state = types_pb2.ServiceState.State.UNINITED

        self.configs = None

        self.engine = None

    def check_state(self, context):
        if self.state == types_pb2.ServiceState.State.UNINITED:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, 'Service not inited')

    def ResetService(self, request, context):
        self.reset()
        return types_pb2.CommonResponse()

    def QueryService(self, request, context):
        return types_pb2.ServiceState(state=self.state)

    def GetSimenvConfig(self, request, context):
        self.check_state(context)
        return self.configs

    def SetSimenvConfig(self, request, context):
        if request.name not in SimEngines:
            message = f'{request.name} engine not supported'
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, message)

        try:
            args = json.loads(request.args) if request.args else {}
            self.engine = SimEngines[request.name](**args)
        except Exception as e:
            message = f'Invalid args for {request.name} engine, info: {e}'
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, message)

        self.state = types_pb2.ServiceState.State.INITED
        self.configs = request

        return types_pb2.CommonResponse()

    def SimControl(self, request, context):
        self.check_state(context)
        try:
            params = json.loads(request.params) if request.params else {}
            self.engine.control(type=request.type, params=params)
        except Exception as e:
            message = f'Invalid command {request.type} with its params, info: {e}'
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, message)
        return types_pb2.CommonResponse()

    def SimMonitor(self, request, context):
        self.check_state(context)
        state = self.engine.state
        data_, logs_ = self.engine.monitor()
        data, logs = json.dumps(data_), json.dumps(logs_)
        return simenv_pb2.SimInfo(state=state, data=data, logs=logs)

    def Call(self, request, context):
        self.check_state(context)
        name, dstr, dbin = self.engine.call(request.name, request.dstr, request.dbin)
        return types_pb2.CallData(name=name, dstr=dstr, dbin=dbin)


def simenv_server(host, port, max_workers, max_msg_len):
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        options=[
            ('grpc.max_send_message_length', max_msg_len * 1024 * 1024),
            ('grpc.max_receive_message_length', max_msg_len * 1024 * 1024),
        ],
    )
    simenv_pb2_grpc.add_SimenvServicer_to_server(SimenvServicer(), server)
    port = server.add_insecure_port(f'{host}:{port}')
    server.start()
    logging.info(f'Simenv server started at {host}:{port}')

    def grace_exit(*_):
        logging.info('Simenv service stopping...')
        evt = server.stop(0)
        evt.wait(1)

    signal.signal(signal.SIGINT, grace_exit)
    signal.signal(signal.SIGTERM, grace_exit)
    server.wait_for_termination()
    logging.info('Simenv server stopped.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run an simenv service.')
    parser.add_argument('-i', '--host', type=str, default='0.0.0.0', help='Host to listen on.')
    parser.add_argument('-p', '--port', type=int, default=0, help='Port to listen on.')
    parser.add_argument('-w', '--worker', type=int, default=10, help='Max workers in thread pool.')
    parser.add_argument('-m', '--msglen', type=int, default=4, help='Max message length in MB.')
    parser.add_argument('-l', '--loglvl', type=str, default='info', help='Log level defined in `logging`.')
    args = parser.parse_args()

    logging.basicConfig(
        format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s',
        level=args.loglvl.upper(),
    )

    simenv_server(args.host, args.port, args.worker, args.msglen)
