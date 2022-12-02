# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from protos import agent_pb2 as protos_dot_agent__pb2
from protos import types_pb2 as protos_dot_types__pb2


class AgentStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.ResetServer = channel.unary_unary(
            '/game.agent.Agent/ResetServer',
            request_serializer=protos_dot_types__pb2.CommonRequest.SerializeToString,
            response_deserializer=protos_dot_types__pb2.CommonResponse.FromString,
        )
        self.GetAgentConfig = channel.unary_unary(
            '/game.agent.Agent/GetAgentConfig',
            request_serializer=protos_dot_types__pb2.CommonRequest.SerializeToString,
            response_deserializer=protos_dot_agent__pb2.AgentConfig.FromString,
        )
        self.SetAgentConfig = channel.unary_unary(
            '/game.agent.Agent/SetAgentConfig',
            request_serializer=protos_dot_agent__pb2.AgentConfig.SerializeToString,
            response_deserializer=protos_dot_types__pb2.CommonResponse.FromString,
        )
        self.GetAgentMode = channel.unary_unary(
            '/game.agent.Agent/GetAgentMode',
            request_serializer=protos_dot_types__pb2.CommonRequest.SerializeToString,
            response_deserializer=protos_dot_agent__pb2.AgentMode.FromString,
        )
        self.SetAgentMode = channel.unary_unary(
            '/game.agent.Agent/SetAgentMode',
            request_serializer=protos_dot_agent__pb2.AgentMode.SerializeToString,
            response_deserializer=protos_dot_types__pb2.CommonResponse.FromString,
        )
        self.GetAgentWeight = channel.unary_unary(
            '/game.agent.Agent/GetAgentWeight',
            request_serializer=protos_dot_types__pb2.CommonRequest.SerializeToString,
            response_deserializer=protos_dot_agent__pb2.AgentWeight.FromString,
        )
        self.SetAgentWeight = channel.unary_unary(
            '/game.agent.Agent/SetAgentWeight',
            request_serializer=protos_dot_agent__pb2.AgentWeight.SerializeToString,
            response_deserializer=protos_dot_types__pb2.CommonResponse.FromString,
        )
        self.GetAgentBuffer = channel.unary_unary(
            '/game.agent.Agent/GetAgentBuffer',
            request_serializer=protos_dot_types__pb2.CommonRequest.SerializeToString,
            response_deserializer=protos_dot_agent__pb2.AgentBuffer.FromString,
        )
        self.SetAgentBuffer = channel.unary_unary(
            '/game.agent.Agent/SetAgentBuffer',
            request_serializer=protos_dot_agent__pb2.AgentBuffer.SerializeToString,
            response_deserializer=protos_dot_types__pb2.CommonResponse.FromString,
        )
        self.GetAgentStatus = channel.unary_unary(
            '/game.agent.Agent/GetAgentStatus',
            request_serializer=protos_dot_types__pb2.CommonRequest.SerializeToString,
            response_deserializer=protos_dot_agent__pb2.AgentStatus.FromString,
        )
        self.SetAgentStatus = channel.unary_unary(
            '/game.agent.Agent/SetAgentStatus',
            request_serializer=protos_dot_agent__pb2.AgentStatus.SerializeToString,
            response_deserializer=protos_dot_types__pb2.CommonResponse.FromString,
        )
        self.GetAction = channel.unary_unary(
            '/game.agent.Agent/GetAction',
            request_serializer=protos_dot_types__pb2.JsonString.SerializeToString,
            response_deserializer=protos_dot_types__pb2.JsonString.FromString,
        )


class AgentServicer(object):
    """Missing associated documentation comment in .proto file."""

    def ResetServer(self, request, context):
        """重置Agent服务
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetAgentConfig(self, request, context):
        """获取智能体配置
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def SetAgentConfig(self, request, context):
        """设置智能体配置
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetAgentMode(self, request, context):
        """获取智能体模式
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def SetAgentMode(self, request, context):
        """设置智能体模式
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetAgentWeight(self, request, context):
        """获取智能体权重
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def SetAgentWeight(self, request, context):
        """设置智能体权重
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetAgentBuffer(self, request, context):
        """获取智能体经验池
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def SetAgentBuffer(self, request, context):
        """设置智能体经验池
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetAgentStatus(self, request, context):
        """获取智能体状态
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def SetAgentStatus(self, request, context):
        """设置智能体状态
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetAction(self, request, context):
        """获取决策动作
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_AgentServicer_to_server(servicer, server):
    rpc_method_handlers = {
        'ResetServer':
            grpc.unary_unary_rpc_method_handler(
                servicer.ResetServer,
                request_deserializer=protos_dot_types__pb2.CommonRequest.FromString,
                response_serializer=protos_dot_types__pb2.CommonResponse.SerializeToString,
            ),
        'GetAgentConfig':
            grpc.unary_unary_rpc_method_handler(
                servicer.GetAgentConfig,
                request_deserializer=protos_dot_types__pb2.CommonRequest.FromString,
                response_serializer=protos_dot_agent__pb2.AgentConfig.SerializeToString,
            ),
        'SetAgentConfig':
            grpc.unary_unary_rpc_method_handler(
                servicer.SetAgentConfig,
                request_deserializer=protos_dot_agent__pb2.AgentConfig.FromString,
                response_serializer=protos_dot_types__pb2.CommonResponse.SerializeToString,
            ),
        'GetAgentMode':
            grpc.unary_unary_rpc_method_handler(
                servicer.GetAgentMode,
                request_deserializer=protos_dot_types__pb2.CommonRequest.FromString,
                response_serializer=protos_dot_agent__pb2.AgentMode.SerializeToString,
            ),
        'SetAgentMode':
            grpc.unary_unary_rpc_method_handler(
                servicer.SetAgentMode,
                request_deserializer=protos_dot_agent__pb2.AgentMode.FromString,
                response_serializer=protos_dot_types__pb2.CommonResponse.SerializeToString,
            ),
        'GetAgentWeight':
            grpc.unary_unary_rpc_method_handler(
                servicer.GetAgentWeight,
                request_deserializer=protos_dot_types__pb2.CommonRequest.FromString,
                response_serializer=protos_dot_agent__pb2.AgentWeight.SerializeToString,
            ),
        'SetAgentWeight':
            grpc.unary_unary_rpc_method_handler(
                servicer.SetAgentWeight,
                request_deserializer=protos_dot_agent__pb2.AgentWeight.FromString,
                response_serializer=protos_dot_types__pb2.CommonResponse.SerializeToString,
            ),
        'GetAgentBuffer':
            grpc.unary_unary_rpc_method_handler(
                servicer.GetAgentBuffer,
                request_deserializer=protos_dot_types__pb2.CommonRequest.FromString,
                response_serializer=protos_dot_agent__pb2.AgentBuffer.SerializeToString,
            ),
        'SetAgentBuffer':
            grpc.unary_unary_rpc_method_handler(
                servicer.SetAgentBuffer,
                request_deserializer=protos_dot_agent__pb2.AgentBuffer.FromString,
                response_serializer=protos_dot_types__pb2.CommonResponse.SerializeToString,
            ),
        'GetAgentStatus':
            grpc.unary_unary_rpc_method_handler(
                servicer.GetAgentStatus,
                request_deserializer=protos_dot_types__pb2.CommonRequest.FromString,
                response_serializer=protos_dot_agent__pb2.AgentStatus.SerializeToString,
            ),
        'SetAgentStatus':
            grpc.unary_unary_rpc_method_handler(
                servicer.SetAgentStatus,
                request_deserializer=protos_dot_agent__pb2.AgentStatus.FromString,
                response_serializer=protos_dot_types__pb2.CommonResponse.SerializeToString,
            ),
        'GetAction':
            grpc.unary_unary_rpc_method_handler(
                servicer.GetAction,
                request_deserializer=protos_dot_types__pb2.JsonString.FromString,
                response_serializer=protos_dot_types__pb2.JsonString.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler('game.agent.Agent', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


# This class is part of an EXPERIMENTAL API.
class Agent(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def ResetServer(request,
                    target,
                    options=(),
                    channel_credentials=None,
                    call_credentials=None,
                    insecure=False,
                    compression=None,
                    wait_for_ready=None,
                    timeout=None,
                    metadata=None):
        return grpc.experimental.unary_unary(request, target, '/game.agent.Agent/ResetServer',
                                             protos_dot_types__pb2.CommonRequest.SerializeToString,
                                             protos_dot_types__pb2.CommonResponse.FromString, options, channel_credentials,
                                             insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetAgentConfig(request,
                       target,
                       options=(),
                       channel_credentials=None,
                       call_credentials=None,
                       insecure=False,
                       compression=None,
                       wait_for_ready=None,
                       timeout=None,
                       metadata=None):
        return grpc.experimental.unary_unary(request, target, '/game.agent.Agent/GetAgentConfig',
                                             protos_dot_types__pb2.CommonRequest.SerializeToString,
                                             protos_dot_agent__pb2.AgentConfig.FromString, options, channel_credentials,
                                             insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def SetAgentConfig(request,
                       target,
                       options=(),
                       channel_credentials=None,
                       call_credentials=None,
                       insecure=False,
                       compression=None,
                       wait_for_ready=None,
                       timeout=None,
                       metadata=None):
        return grpc.experimental.unary_unary(request, target, '/game.agent.Agent/SetAgentConfig',
                                             protos_dot_agent__pb2.AgentConfig.SerializeToString,
                                             protos_dot_types__pb2.CommonResponse.FromString, options, channel_credentials,
                                             insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetAgentMode(request,
                     target,
                     options=(),
                     channel_credentials=None,
                     call_credentials=None,
                     insecure=False,
                     compression=None,
                     wait_for_ready=None,
                     timeout=None,
                     metadata=None):
        return grpc.experimental.unary_unary(request, target, '/game.agent.Agent/GetAgentMode',
                                             protos_dot_types__pb2.CommonRequest.SerializeToString,
                                             protos_dot_agent__pb2.AgentMode.FromString, options, channel_credentials, insecure,
                                             call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def SetAgentMode(request,
                     target,
                     options=(),
                     channel_credentials=None,
                     call_credentials=None,
                     insecure=False,
                     compression=None,
                     wait_for_ready=None,
                     timeout=None,
                     metadata=None):
        return grpc.experimental.unary_unary(request, target, '/game.agent.Agent/SetAgentMode',
                                             protos_dot_agent__pb2.AgentMode.SerializeToString,
                                             protos_dot_types__pb2.CommonResponse.FromString, options, channel_credentials,
                                             insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetAgentWeight(request,
                       target,
                       options=(),
                       channel_credentials=None,
                       call_credentials=None,
                       insecure=False,
                       compression=None,
                       wait_for_ready=None,
                       timeout=None,
                       metadata=None):
        return grpc.experimental.unary_unary(request, target, '/game.agent.Agent/GetAgentWeight',
                                             protos_dot_types__pb2.CommonRequest.SerializeToString,
                                             protos_dot_agent__pb2.AgentWeight.FromString, options, channel_credentials,
                                             insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def SetAgentWeight(request,
                       target,
                       options=(),
                       channel_credentials=None,
                       call_credentials=None,
                       insecure=False,
                       compression=None,
                       wait_for_ready=None,
                       timeout=None,
                       metadata=None):
        return grpc.experimental.unary_unary(request, target, '/game.agent.Agent/SetAgentWeight',
                                             protos_dot_agent__pb2.AgentWeight.SerializeToString,
                                             protos_dot_types__pb2.CommonResponse.FromString, options, channel_credentials,
                                             insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetAgentBuffer(request,
                       target,
                       options=(),
                       channel_credentials=None,
                       call_credentials=None,
                       insecure=False,
                       compression=None,
                       wait_for_ready=None,
                       timeout=None,
                       metadata=None):
        return grpc.experimental.unary_unary(request, target, '/game.agent.Agent/GetAgentBuffer',
                                             protos_dot_types__pb2.CommonRequest.SerializeToString,
                                             protos_dot_agent__pb2.AgentBuffer.FromString, options, channel_credentials,
                                             insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def SetAgentBuffer(request,
                       target,
                       options=(),
                       channel_credentials=None,
                       call_credentials=None,
                       insecure=False,
                       compression=None,
                       wait_for_ready=None,
                       timeout=None,
                       metadata=None):
        return grpc.experimental.unary_unary(request, target, '/game.agent.Agent/SetAgentBuffer',
                                             protos_dot_agent__pb2.AgentBuffer.SerializeToString,
                                             protos_dot_types__pb2.CommonResponse.FromString, options, channel_credentials,
                                             insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetAgentStatus(request,
                       target,
                       options=(),
                       channel_credentials=None,
                       call_credentials=None,
                       insecure=False,
                       compression=None,
                       wait_for_ready=None,
                       timeout=None,
                       metadata=None):
        return grpc.experimental.unary_unary(request, target, '/game.agent.Agent/GetAgentStatus',
                                             protos_dot_types__pb2.CommonRequest.SerializeToString,
                                             protos_dot_agent__pb2.AgentStatus.FromString, options, channel_credentials,
                                             insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def SetAgentStatus(request,
                       target,
                       options=(),
                       channel_credentials=None,
                       call_credentials=None,
                       insecure=False,
                       compression=None,
                       wait_for_ready=None,
                       timeout=None,
                       metadata=None):
        return grpc.experimental.unary_unary(request, target, '/game.agent.Agent/SetAgentStatus',
                                             protos_dot_agent__pb2.AgentStatus.SerializeToString,
                                             protos_dot_types__pb2.CommonResponse.FromString, options, channel_credentials,
                                             insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetAction(request,
                  target,
                  options=(),
                  channel_credentials=None,
                  call_credentials=None,
                  insecure=False,
                  compression=None,
                  wait_for_ready=None,
                  timeout=None,
                  metadata=None):
        return grpc.experimental.unary_unary(request, target, '/game.agent.Agent/GetAction',
                                             protos_dot_types__pb2.JsonString.SerializeToString,
                                             protos_dot_types__pb2.JsonString.FromString, options, channel_credentials,
                                             insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
