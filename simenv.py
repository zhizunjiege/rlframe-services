import argparse
from concurrent import futures
import json

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
        if self.engine is not None:
            self.engine.close()
        self.reset()
        return types_pb2.CommonResponse()

    def QueryService(self, request, context):
        return types_pb2.ServiceState(state=self.state)

    def GetSimenvConfig(self, request, context):
        self.check_state(context)
        return self.configs

    def SetSimenvConfig(self, request, context):
        args = json.loads(request.args)
        self.engine = SimEngines[request.type](id=id, **args)

        self.state = types_pb2.ServiceState.State.INITED
        self.configs = request

        return types_pb2.CommonResponse()

    def SimControl(self, request, context):
        self.check_state(context)
        if request.type == simenv_pb2.SimCmd.Type.INIT:
            cmd = 'init'
        elif request.type == simenv_pb2.SimCmd.Type.START:
            cmd = 'start'
        elif request.type == simenv_pb2.SimCmd.Type.PAUSE:
            cmd = 'pause'
        elif request.type == simenv_pb2.SimCmd.Type.STEP:
            cmd = 'step'
        elif request.type == simenv_pb2.SimCmd.Type.RESUME:
            cmd = 'resume'
        elif request.type == simenv_pb2.SimCmd.Type.STOP:
            cmd = 'stop'
        elif request.type == simenv_pb2.SimCmd.Type.DONE:
            cmd = 'done'
        elif request.type == simenv_pb2.SimCmd.Type.PARAM:
            cmd = 'param'
        else:
            ...
        params = json.loads(request.params)
        self.engine.control(cmd=cmd, params=params)
        return types_pb2.CommonResponse()

    def SimMonitor(self, request, context):
        self.check_state(context)
        if self.engine.state == 'uninited':
            state = simenv_pb2.SimInfo.State.UNINITED
        elif self.engine.state == 'stopped':
            state = simenv_pb2.SimInfo.State.STOPPED
        elif self.engine.state == 'running':
            state = simenv_pb2.SimInfo.State.RUNNING
        elif self.engine.state == 'suspended':
            state = simenv_pb2.SimInfo.State.SUSPENDED
        else:
            ...
        data_, logs_ = self.engine.monitor()
        data, logs = json.dumps(data_), json.dumps(logs_)
        return simenv_pb2.SimInfo(state=state, data=data, logs=logs)


def simenv_server(ip, port, max_workers, max_msg_len):
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        options=[
            ('grpc.max_send_message_length', max_msg_len * 1024 * 1024),
            ('grpc.max_receive_message_length', max_msg_len * 1024 * 1024),
        ],
    )
    simenv_pb2_grpc.add_SimenvServicer_to_server(SimenvServicer(), server)
    port = server.add_insecure_port(f'{ip}:{port}')
    server.start()
    print(f'Simenv server started at {ip}:{port}')
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        ...


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run an simenv service.')
    parser.add_argument('-i', '--ip', type=str, default='0.0.0.0', help='IP address to listen on.')
    parser.add_argument('-p', '--port', type=int, default=0, help='Port to listen on.')
    parser.add_argument('-w', '--work', type=int, default=10, help='Max workers in thread pool.')
    parser.add_argument('-m', '--msglen', type=int, default=4, help='Max message length in MB.')
    args = parser.parse_args()
    simenv_server(args.ip, args.port, args.work, args.msglen)
