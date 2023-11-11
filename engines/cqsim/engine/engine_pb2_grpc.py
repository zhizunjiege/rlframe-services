# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from . import engine_pb2 as engine__pb2


class SimControllerStub(object):
    """仿真运行控制
    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.GetEntityList = channel.unary_unary(
            '/cqsim.control.SimController/GetEntityList',
            request_serializer=engine__pb2.EntityListRequest.SerializeToString,
            response_deserializer=engine__pb2.EntityListResponse.FromString,
        )
        self.Init = channel.unary_unary(
            '/cqsim.control.SimController/Init',
            request_serializer=engine__pb2.InitInfo.SerializeToString,
            response_deserializer=engine__pb2.CommonResponse.FromString,
        )
        self.Control = channel.unary_unary(
            '/cqsim.control.SimController/Control',
            request_serializer=engine__pb2.ControlCmd.SerializeToString,
            response_deserializer=engine__pb2.CommonResponse.FromString,
        )
        self.GetSysInfo = channel.unary_stream(
            '/cqsim.control.SimController/GetSysInfo',
            request_serializer=engine__pb2.CommonRequest.SerializeToString,
            response_deserializer=engine__pb2.SysInfoResponse.FromString,
        )
        self.GetDataSysInfo = channel.unary_stream(
            '/cqsim.control.SimController/GetDataSysInfo',
            request_serializer=engine__pb2.CommonRequest.SerializeToString,
            response_deserializer=engine__pb2.DataSysInfoResponse.FromString,
        )
        self.GetAllNode = channel.unary_unary(
            '/cqsim.control.SimController/GetAllNode',
            request_serializer=engine__pb2.CommonRequest.SerializeToString,
            response_deserializer=engine__pb2.NodeResponse.FromString,
        )
        self.GetInitedPercent = channel.unary_stream(
            '/cqsim.control.SimController/GetInitedPercent',
            request_serializer=engine__pb2.CommonRequest.SerializeToString,
            response_deserializer=engine__pb2.InitedResponse.FromString,
        )
        self.GetErrorMsg = channel.unary_stream(
            '/cqsim.control.SimController/GetErrorMsg',
            request_serializer=engine__pb2.CommonRequest.SerializeToString,
            response_deserializer=engine__pb2.ErrMsgResponse.FromString,
        )
        self.GetNodeJoinExit = channel.unary_stream(
            '/cqsim.control.SimController/GetNodeJoinExit',
            request_serializer=engine__pb2.CommonRequest.SerializeToString,
            response_deserializer=engine__pb2.NodeJoinExitResponse.FromString,
        )
        self.SetLogLevel = channel.unary_unary(
            '/cqsim.control.SimController/SetLogLevel',
            request_serializer=engine__pb2.LogLevelRequest.SerializeToString,
            response_deserializer=engine__pb2.CommonResponse.FromString,
        )
        self.SetHttpInfo = channel.unary_unary(
            '/cqsim.control.SimController/SetHttpInfo',
            request_serializer=engine__pb2.HttpInfo.SerializeToString,
            response_deserializer=engine__pb2.CommonResponse.FromString,
        )
        self.GetScenarioInfo = channel.unary_unary(
            '/cqsim.control.SimController/GetScenarioInfo',
            request_serializer=engine__pb2.CommonRequest.SerializeToString,
            response_deserializer=engine__pb2.ScenarioInfo.FromString,
        )
        self.GetDataAddress = channel.unary_unary(
            '/cqsim.control.SimController/GetDataAddress',
            request_serializer=engine__pb2.CommonRequest.SerializeToString,
            response_deserializer=engine__pb2.CommonResponse.FromString,
        )


class SimControllerServicer(object):
    """仿真运行控制
    """

    def GetEntityList(self, request, context):
        """获取想定中的实体列表 (用于节点设计)
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Init(self, request, context):
        """初始化
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Control(self, request, context):
        """发送控制指令
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetSysInfo(self, request, context):
        """获取系统信息
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetDataSysInfo(self, request, context):
        """获取数据服务系统信息
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetAllNode(self, request, context):
        """获取所有节点
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetInitedPercent(self, request, context):
        """获取分布式下初始化进度
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetErrorMsg(self, request, context):
        """持续获取错误信息
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetNodeJoinExit(self, request, context):
        """节点加入退出
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def SetLogLevel(self, request, context):
        """设置日志等级
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def SetHttpInfo(self, request, context):
        """http相关设置
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetScenarioInfo(self, request, context):
        """获取当前运行的想定信息
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetDataAddress(self, request, context):
        """获取数据服务地址
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_SimControllerServicer_to_server(servicer, server):
    rpc_method_handlers = {
        'GetEntityList':
            grpc.unary_unary_rpc_method_handler(
                servicer.GetEntityList,
                request_deserializer=engine__pb2.EntityListRequest.FromString,
                response_serializer=engine__pb2.EntityListResponse.SerializeToString,
            ),
        'Init':
            grpc.unary_unary_rpc_method_handler(
                servicer.Init,
                request_deserializer=engine__pb2.InitInfo.FromString,
                response_serializer=engine__pb2.CommonResponse.SerializeToString,
            ),
        'Control':
            grpc.unary_unary_rpc_method_handler(
                servicer.Control,
                request_deserializer=engine__pb2.ControlCmd.FromString,
                response_serializer=engine__pb2.CommonResponse.SerializeToString,
            ),
        'GetSysInfo':
            grpc.unary_stream_rpc_method_handler(
                servicer.GetSysInfo,
                request_deserializer=engine__pb2.CommonRequest.FromString,
                response_serializer=engine__pb2.SysInfoResponse.SerializeToString,
            ),
        'GetDataSysInfo':
            grpc.unary_stream_rpc_method_handler(
                servicer.GetDataSysInfo,
                request_deserializer=engine__pb2.CommonRequest.FromString,
                response_serializer=engine__pb2.DataSysInfoResponse.SerializeToString,
            ),
        'GetAllNode':
            grpc.unary_unary_rpc_method_handler(
                servicer.GetAllNode,
                request_deserializer=engine__pb2.CommonRequest.FromString,
                response_serializer=engine__pb2.NodeResponse.SerializeToString,
            ),
        'GetInitedPercent':
            grpc.unary_stream_rpc_method_handler(
                servicer.GetInitedPercent,
                request_deserializer=engine__pb2.CommonRequest.FromString,
                response_serializer=engine__pb2.InitedResponse.SerializeToString,
            ),
        'GetErrorMsg':
            grpc.unary_stream_rpc_method_handler(
                servicer.GetErrorMsg,
                request_deserializer=engine__pb2.CommonRequest.FromString,
                response_serializer=engine__pb2.ErrMsgResponse.SerializeToString,
            ),
        'GetNodeJoinExit':
            grpc.unary_stream_rpc_method_handler(
                servicer.GetNodeJoinExit,
                request_deserializer=engine__pb2.CommonRequest.FromString,
                response_serializer=engine__pb2.NodeJoinExitResponse.SerializeToString,
            ),
        'SetLogLevel':
            grpc.unary_unary_rpc_method_handler(
                servicer.SetLogLevel,
                request_deserializer=engine__pb2.LogLevelRequest.FromString,
                response_serializer=engine__pb2.CommonResponse.SerializeToString,
            ),
        'SetHttpInfo':
            grpc.unary_unary_rpc_method_handler(
                servicer.SetHttpInfo,
                request_deserializer=engine__pb2.HttpInfo.FromString,
                response_serializer=engine__pb2.CommonResponse.SerializeToString,
            ),
        'GetScenarioInfo':
            grpc.unary_unary_rpc_method_handler(
                servicer.GetScenarioInfo,
                request_deserializer=engine__pb2.CommonRequest.FromString,
                response_serializer=engine__pb2.ScenarioInfo.SerializeToString,
            ),
        'GetDataAddress':
            grpc.unary_unary_rpc_method_handler(
                servicer.GetDataAddress,
                request_deserializer=engine__pb2.CommonRequest.FromString,
                response_serializer=engine__pb2.CommonResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler('cqsim.control.SimController', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


# This class is part of an EXPERIMENTAL API.
class SimController(object):
    """仿真运行控制
    """

    @staticmethod
    def GetEntityList(request,
                      target,
                      options=(),
                      channel_credentials=None,
                      call_credentials=None,
                      insecure=False,
                      compression=None,
                      wait_for_ready=None,
                      timeout=None,
                      metadata=None):
        return grpc.experimental.unary_unary(request, target, '/cqsim.control.SimController/GetEntityList',
                                             engine__pb2.EntityListRequest.SerializeToString,
                                             engine__pb2.EntityListResponse.FromString, options, channel_credentials, insecure,
                                             call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def Init(request,
             target,
             options=(),
             channel_credentials=None,
             call_credentials=None,
             insecure=False,
             compression=None,
             wait_for_ready=None,
             timeout=None,
             metadata=None):
        return grpc.experimental.unary_unary(request, target, '/cqsim.control.SimController/Init',
                                             engine__pb2.InitInfo.SerializeToString, engine__pb2.CommonResponse.FromString,
                                             options, channel_credentials, insecure, call_credentials, compression,
                                             wait_for_ready, timeout, metadata)

    @staticmethod
    def Control(request,
                target,
                options=(),
                channel_credentials=None,
                call_credentials=None,
                insecure=False,
                compression=None,
                wait_for_ready=None,
                timeout=None,
                metadata=None):
        return grpc.experimental.unary_unary(request, target, '/cqsim.control.SimController/Control',
                                             engine__pb2.ControlCmd.SerializeToString, engine__pb2.CommonResponse.FromString,
                                             options, channel_credentials, insecure, call_credentials, compression,
                                             wait_for_ready, timeout, metadata)

    @staticmethod
    def GetSysInfo(request,
                   target,
                   options=(),
                   channel_credentials=None,
                   call_credentials=None,
                   insecure=False,
                   compression=None,
                   wait_for_ready=None,
                   timeout=None,
                   metadata=None):
        return grpc.experimental.unary_stream(request, target, '/cqsim.control.SimController/GetSysInfo',
                                              engine__pb2.CommonRequest.SerializeToString,
                                              engine__pb2.SysInfoResponse.FromString, options, channel_credentials, insecure,
                                              call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetDataSysInfo(request,
                       target,
                       options=(),
                       channel_credentials=None,
                       call_credentials=None,
                       insecure=False,
                       compression=None,
                       wait_for_ready=None,
                       timeout=None,
                       metadata=None):
        return grpc.experimental.unary_stream(request, target, '/cqsim.control.SimController/GetDataSysInfo',
                                              engine__pb2.CommonRequest.SerializeToString,
                                              engine__pb2.DataSysInfoResponse.FromString, options, channel_credentials,
                                              insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetAllNode(request,
                   target,
                   options=(),
                   channel_credentials=None,
                   call_credentials=None,
                   insecure=False,
                   compression=None,
                   wait_for_ready=None,
                   timeout=None,
                   metadata=None):
        return grpc.experimental.unary_unary(request, target, '/cqsim.control.SimController/GetAllNode',
                                             engine__pb2.CommonRequest.SerializeToString, engine__pb2.NodeResponse.FromString,
                                             options, channel_credentials, insecure, call_credentials, compression,
                                             wait_for_ready, timeout, metadata)

    @staticmethod
    def GetInitedPercent(request,
                         target,
                         options=(),
                         channel_credentials=None,
                         call_credentials=None,
                         insecure=False,
                         compression=None,
                         wait_for_ready=None,
                         timeout=None,
                         metadata=None):
        return grpc.experimental.unary_stream(request, target, '/cqsim.control.SimController/GetInitedPercent',
                                              engine__pb2.CommonRequest.SerializeToString,
                                              engine__pb2.InitedResponse.FromString, options, channel_credentials, insecure,
                                              call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetErrorMsg(request,
                    target,
                    options=(),
                    channel_credentials=None,
                    call_credentials=None,
                    insecure=False,
                    compression=None,
                    wait_for_ready=None,
                    timeout=None,
                    metadata=None):
        return grpc.experimental.unary_stream(request, target, '/cqsim.control.SimController/GetErrorMsg',
                                              engine__pb2.CommonRequest.SerializeToString,
                                              engine__pb2.ErrMsgResponse.FromString, options, channel_credentials, insecure,
                                              call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetNodeJoinExit(request,
                        target,
                        options=(),
                        channel_credentials=None,
                        call_credentials=None,
                        insecure=False,
                        compression=None,
                        wait_for_ready=None,
                        timeout=None,
                        metadata=None):
        return grpc.experimental.unary_stream(request, target, '/cqsim.control.SimController/GetNodeJoinExit',
                                              engine__pb2.CommonRequest.SerializeToString,
                                              engine__pb2.NodeJoinExitResponse.FromString, options, channel_credentials,
                                              insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def SetLogLevel(request,
                    target,
                    options=(),
                    channel_credentials=None,
                    call_credentials=None,
                    insecure=False,
                    compression=None,
                    wait_for_ready=None,
                    timeout=None,
                    metadata=None):
        return grpc.experimental.unary_unary(request, target, '/cqsim.control.SimController/SetLogLevel',
                                             engine__pb2.LogLevelRequest.SerializeToString,
                                             engine__pb2.CommonResponse.FromString, options, channel_credentials, insecure,
                                             call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def SetHttpInfo(request,
                    target,
                    options=(),
                    channel_credentials=None,
                    call_credentials=None,
                    insecure=False,
                    compression=None,
                    wait_for_ready=None,
                    timeout=None,
                    metadata=None):
        return grpc.experimental.unary_unary(request, target, '/cqsim.control.SimController/SetHttpInfo',
                                             engine__pb2.HttpInfo.SerializeToString, engine__pb2.CommonResponse.FromString,
                                             options, channel_credentials, insecure, call_credentials, compression,
                                             wait_for_ready, timeout, metadata)

    @staticmethod
    def GetScenarioInfo(request,
                        target,
                        options=(),
                        channel_credentials=None,
                        call_credentials=None,
                        insecure=False,
                        compression=None,
                        wait_for_ready=None,
                        timeout=None,
                        metadata=None):
        return grpc.experimental.unary_unary(request, target, '/cqsim.control.SimController/GetScenarioInfo',
                                             engine__pb2.CommonRequest.SerializeToString, engine__pb2.ScenarioInfo.FromString,
                                             options, channel_credentials, insecure, call_credentials, compression,
                                             wait_for_ready, timeout, metadata)

    @staticmethod
    def GetDataAddress(request,
                       target,
                       options=(),
                       channel_credentials=None,
                       call_credentials=None,
                       insecure=False,
                       compression=None,
                       wait_for_ready=None,
                       timeout=None,
                       metadata=None):
        return grpc.experimental.unary_unary(request, target, '/cqsim.control.SimController/GetDataAddress',
                                             engine__pb2.CommonRequest.SerializeToString, engine__pb2.CommonResponse.FromString,
                                             options, channel_credentials, insecure, call_credentials, compression,
                                             wait_for_ready, timeout, metadata)
