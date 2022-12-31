# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: protos/bff.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()

from protos import types_pb2 as protos_dot_types__pb2
from protos import agent_pb2 as protos_dot_agent__pb2
from protos import simenv_pb2 as protos_dot_simenv__pb2

DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(
    b'\n\x10protos/bff.proto\x12\x08game.bff\x1a\x12protos/types.proto\x1a\x12protos/agent.proto\x1a\x13protos/simenv.proto\"\x8c\x01\n\x0bServiceInfo\x12(\n\x04type\x18\x01 \x01(\x0e\x32\x1a.game.bff.ServiceInfo.Type\x12\x0c\n\x04name\x18\x02 \x01(\t\x12\n\n\x02ip\x18\x03 \x01(\t\x12\x0c\n\x04port\x18\x04 \x01(\r\x12\x0c\n\x04\x64\x65sc\x18\x05 \x01(\t\"\x1d\n\x04Type\x12\t\n\x05\x41GENT\x10\x00\x12\n\n\x06SIMENV\x10\x01\"\x1c\n\rServiceIdList\x12\x0b\n\x03ids\x18\x01 \x03(\t\":\n\x0fServiceInfoList\x12\'\n\x08services\x18\x01 \x03(\x0b\x32\x15.game.bff.ServiceInfo\"\x92\x01\n\x0eServiceInfoMap\x12\x38\n\x08services\x18\x01 \x03(\x0b\x32&.game.bff.ServiceInfoMap.ServicesEntry\x1a\x46\n\rServicesEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12$\n\x05value\x18\x02 \x01(\x0b\x32\x15.game.bff.ServiceInfo:\x02\x38\x01\"\xdf\x05\n\nDataConfig\x12.\n\x05types\x18\x01 \x03(\x0b\x32\x1f.game.bff.DataConfig.TypesEntry\x12,\n\x04\x64\x61ta\x18\x02 \x03(\x0b\x32\x1e.game.bff.DataConfig.DataEntry\x1al\n\x04Type\x12\x35\n\x06\x66ields\x18\x01 \x03(\x0b\x32%.game.bff.DataConfig.Type.FieldsEntry\x1a-\n\x0b\x46ieldsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\x1a\x32\n\x05Param\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x0c\n\x04type\x18\x02 \x01(\t\x12\r\n\x05value\x18\x03 \x01(\t\x1a\xbe\x02\n\x05Model\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x41\n\x0cinput_params\x18\x02 \x03(\x0b\x32+.game.bff.DataConfig.Model.InputParamsEntry\x12\x43\n\routput_params\x18\x03 \x03(\x0b\x32,.game.bff.DataConfig.Model.OutputParamsEntry\x1aN\n\x10InputParamsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12)\n\x05value\x18\x02 \x01(\x0b\x32\x1a.game.bff.DataConfig.Param:\x02\x38\x01\x1aO\n\x11OutputParamsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12)\n\x05value\x18\x02 \x01(\x0b\x32\x1a.game.bff.DataConfig.Param:\x02\x38\x01\x1aG\n\nTypesEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12(\n\x05value\x18\x02 \x01(\x0b\x32\x19.game.bff.DataConfig.Type:\x02\x38\x01\x1aG\n\tDataEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12)\n\x05value\x18\x02 \x01(\x0b\x32\x1a.game.bff.DataConfig.Model:\x02\x38\x01\"\x84\x03\n\x0bRouteConfig\x12\x31\n\x06routes\x18\x01 \x03(\x0b\x32!.game.bff.RouteConfig.RoutesEntry\x12\x15\n\rsim_done_func\x18\x02 \x01(\t\x12\x16\n\x0esim_step_ratio\x18\x03 \x01(\r\x1a&\n\x06\x43onfig\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x0e\n\x06models\x18\x02 \x03(\t\x1a\x9e\x01\n\x05Route\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x39\n\x07\x63onfigs\x18\x02 \x03(\x0b\x32(.game.bff.RouteConfig.Route.ConfigsEntry\x1aL\n\x0c\x43onfigsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12+\n\x05value\x18\x02 \x01(\x0b\x32\x1c.game.bff.RouteConfig.Config:\x02\x38\x01\x1aJ\n\x0bRoutesEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12*\n\x05value\x18\x02 \x01(\x0b\x32\x1b.game.bff.RouteConfig.Route:\x02\x38\x01\"\x91\x01\n\x0fServiceStateMap\x12\x35\n\x06states\x18\x01 \x03(\x0b\x32%.game.bff.ServiceStateMap.StatesEntry\x1aG\n\x0bStatesEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\'\n\x05value\x18\x02 \x01(\x0b\x32\x18.game.types.ServiceState:\x02\x38\x01\"\x91\x01\n\x0e\x41gentConfigMap\x12\x36\n\x07\x63onfigs\x18\x01 \x03(\x0b\x32%.game.bff.AgentConfigMap.ConfigsEntry\x1aG\n\x0c\x43onfigsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12&\n\x05value\x18\x02 \x01(\x0b\x32\x17.game.agent.AgentConfig:\x02\x38\x01\"\x85\x01\n\x0c\x41gentModeMap\x12\x30\n\x05modes\x18\x01 \x03(\x0b\x32!.game.bff.AgentModeMap.ModesEntry\x1a\x43\n\nModesEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12$\n\x05value\x18\x02 \x01(\x0b\x32\x15.game.agent.AgentMode:\x02\x38\x01\"\x94\x01\n\x0fModelWeightsMap\x12\x37\n\x07weights\x18\x01 \x03(\x0b\x32&.game.bff.ModelWeightsMap.WeightsEntry\x1aH\n\x0cWeightsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\'\n\x05value\x18\x02 \x01(\x0b\x32\x18.game.agent.ModelWeights:\x02\x38\x01\"\x91\x01\n\x0eModelBufferMap\x12\x36\n\x07\x62uffers\x18\x01 \x03(\x0b\x32%.game.bff.ModelBufferMap.BuffersEntry\x1aG\n\x0c\x42uffersEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12&\n\x05value\x18\x02 \x01(\x0b\x32\x17.game.agent.ModelBuffer:\x02\x38\x01\"\x8e\x01\n\x0eModelStatusMap\x12\x34\n\x06status\x18\x01 \x03(\x0b\x32$.game.bff.ModelStatusMap.StatusEntry\x1a\x46\n\x0bStatusEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12&\n\x05value\x18\x02 \x01(\x0b\x32\x17.game.agent.ModelStatus:\x02\x38\x01\"\x95\x01\n\x0fSimenvConfigMap\x12\x37\n\x07\x63onfigs\x18\x01 \x03(\x0b\x32&.game.bff.SimenvConfigMap.ConfigsEntry\x1aI\n\x0c\x43onfigsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12(\n\x05value\x18\x02 \x01(\x0b\x32\x19.game.simenv.SimenvConfig:\x02\x38\x01\"z\n\tSimCmdMap\x12+\n\x04\x63mds\x18\x01 \x03(\x0b\x32\x1d.game.bff.SimCmdMap.CmdsEntry\x1a@\n\tCmdsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\"\n\x05value\x18\x02 \x01(\x0b\x32\x13.game.simenv.SimCmd:\x02\x38\x01\"\x80\x01\n\nSimInfoMap\x12.\n\x05infos\x18\x01 \x03(\x0b\x32\x1f.game.bff.SimInfoMap.InfosEntry\x1a\x42\n\nInfosEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12#\n\x05value\x18\x02 \x01(\x0b\x32\x14.game.simenv.SimInfo:\x02\x38\x01\x32\x8b\x0e\n\x03\x42\x46\x46\x12\x44\n\x0bResetServer\x12\x19.game.types.CommonRequest\x1a\x1a.game.types.CommonResponse\x12\x45\n\x0fRegisterService\x12\x19.game.bff.ServiceInfoList\x1a\x17.game.bff.ServiceIdList\x12H\n\x11UnRegisterService\x12\x17.game.bff.ServiceIdList\x1a\x1a.game.types.CommonResponse\x12\x43\n\x0eGetServiceInfo\x12\x17.game.bff.ServiceIdList\x1a\x18.game.bff.ServiceInfoMap\x12\x46\n\x0eSetServiceInfo\x12\x18.game.bff.ServiceInfoMap\x1a\x1a.game.types.CommonResponse\x12@\n\rGetDataConfig\x12\x19.game.types.CommonRequest\x1a\x14.game.bff.DataConfig\x12\x41\n\rSetDataConfig\x12\x14.game.bff.DataConfig\x1a\x1a.game.types.CommonResponse\x12\x42\n\x0eGetRouteConfig\x12\x19.game.types.CommonRequest\x1a\x15.game.bff.RouteConfig\x12\x43\n\x0eSetRouteConfig\x12\x15.game.bff.RouteConfig\x1a\x1a.game.types.CommonResponse\x12?\n\tProxyChat\x12\x16.game.types.JsonString\x1a\x16.game.types.JsonString(\x01\x30\x01\x12\x43\n\x0cResetService\x12\x17.game.bff.ServiceIdList\x1a\x1a.game.types.CommonResponse\x12\x42\n\x0cQueryService\x12\x17.game.bff.ServiceIdList\x1a\x19.game.bff.ServiceStateMap\x12\x43\n\x0eGetAgentConfig\x12\x17.game.bff.ServiceIdList\x1a\x18.game.bff.AgentConfigMap\x12\x46\n\x0eSetAgentConfig\x12\x18.game.bff.AgentConfigMap\x1a\x1a.game.types.CommonResponse\x12?\n\x0cGetAgentMode\x12\x17.game.bff.ServiceIdList\x1a\x16.game.bff.AgentModeMap\x12\x42\n\x0cSetAgentMode\x12\x16.game.bff.AgentModeMap\x1a\x1a.game.types.CommonResponse\x12\x45\n\x0fGetModelWeights\x12\x17.game.bff.ServiceIdList\x1a\x19.game.bff.ModelWeightsMap\x12H\n\x0fSetModelWeights\x12\x19.game.bff.ModelWeightsMap\x1a\x1a.game.types.CommonResponse\x12\x43\n\x0eGetModelBuffer\x12\x17.game.bff.ServiceIdList\x1a\x18.game.bff.ModelBufferMap\x12\x46\n\x0eSetModelBuffer\x12\x18.game.bff.ModelBufferMap\x1a\x1a.game.types.CommonResponse\x12\x43\n\x0eGetModelStatus\x12\x17.game.bff.ServiceIdList\x1a\x18.game.bff.ModelStatusMap\x12\x46\n\x0eSetModelStatus\x12\x18.game.bff.ModelStatusMap\x1a\x1a.game.types.CommonResponse\x12\x45\n\x0fGetSimenvConfig\x12\x17.game.bff.ServiceIdList\x1a\x19.game.bff.SimenvConfigMap\x12H\n\x0fSetSimenvConfig\x12\x19.game.bff.SimenvConfigMap\x1a\x1a.game.types.CommonResponse\x12=\n\nSimControl\x12\x13.game.bff.SimCmdMap\x1a\x1a.game.types.CommonResponse\x12;\n\nSimMonitor\x12\x17.game.bff.ServiceIdList\x1a\x14.game.bff.SimInfoMapb\x06proto3'
)

_SERVICEINFO = DESCRIPTOR.message_types_by_name['ServiceInfo']
_SERVICEIDLIST = DESCRIPTOR.message_types_by_name['ServiceIdList']
_SERVICEINFOLIST = DESCRIPTOR.message_types_by_name['ServiceInfoList']
_SERVICEINFOMAP = DESCRIPTOR.message_types_by_name['ServiceInfoMap']
_SERVICEINFOMAP_SERVICESENTRY = _SERVICEINFOMAP.nested_types_by_name['ServicesEntry']
_DATACONFIG = DESCRIPTOR.message_types_by_name['DataConfig']
_DATACONFIG_TYPE = _DATACONFIG.nested_types_by_name['Type']
_DATACONFIG_TYPE_FIELDSENTRY = _DATACONFIG_TYPE.nested_types_by_name['FieldsEntry']
_DATACONFIG_PARAM = _DATACONFIG.nested_types_by_name['Param']
_DATACONFIG_MODEL = _DATACONFIG.nested_types_by_name['Model']
_DATACONFIG_MODEL_INPUTPARAMSENTRY = _DATACONFIG_MODEL.nested_types_by_name['InputParamsEntry']
_DATACONFIG_MODEL_OUTPUTPARAMSENTRY = _DATACONFIG_MODEL.nested_types_by_name['OutputParamsEntry']
_DATACONFIG_TYPESENTRY = _DATACONFIG.nested_types_by_name['TypesEntry']
_DATACONFIG_DATAENTRY = _DATACONFIG.nested_types_by_name['DataEntry']
_ROUTECONFIG = DESCRIPTOR.message_types_by_name['RouteConfig']
_ROUTECONFIG_CONFIG = _ROUTECONFIG.nested_types_by_name['Config']
_ROUTECONFIG_ROUTE = _ROUTECONFIG.nested_types_by_name['Route']
_ROUTECONFIG_ROUTE_CONFIGSENTRY = _ROUTECONFIG_ROUTE.nested_types_by_name['ConfigsEntry']
_ROUTECONFIG_ROUTESENTRY = _ROUTECONFIG.nested_types_by_name['RoutesEntry']
_SERVICESTATEMAP = DESCRIPTOR.message_types_by_name['ServiceStateMap']
_SERVICESTATEMAP_STATESENTRY = _SERVICESTATEMAP.nested_types_by_name['StatesEntry']
_AGENTCONFIGMAP = DESCRIPTOR.message_types_by_name['AgentConfigMap']
_AGENTCONFIGMAP_CONFIGSENTRY = _AGENTCONFIGMAP.nested_types_by_name['ConfigsEntry']
_AGENTMODEMAP = DESCRIPTOR.message_types_by_name['AgentModeMap']
_AGENTMODEMAP_MODESENTRY = _AGENTMODEMAP.nested_types_by_name['ModesEntry']
_MODELWEIGHTSMAP = DESCRIPTOR.message_types_by_name['ModelWeightsMap']
_MODELWEIGHTSMAP_WEIGHTSENTRY = _MODELWEIGHTSMAP.nested_types_by_name['WeightsEntry']
_MODELBUFFERMAP = DESCRIPTOR.message_types_by_name['ModelBufferMap']
_MODELBUFFERMAP_BUFFERSENTRY = _MODELBUFFERMAP.nested_types_by_name['BuffersEntry']
_MODELSTATUSMAP = DESCRIPTOR.message_types_by_name['ModelStatusMap']
_MODELSTATUSMAP_STATUSENTRY = _MODELSTATUSMAP.nested_types_by_name['StatusEntry']
_SIMENVCONFIGMAP = DESCRIPTOR.message_types_by_name['SimenvConfigMap']
_SIMENVCONFIGMAP_CONFIGSENTRY = _SIMENVCONFIGMAP.nested_types_by_name['ConfigsEntry']
_SIMCMDMAP = DESCRIPTOR.message_types_by_name['SimCmdMap']
_SIMCMDMAP_CMDSENTRY = _SIMCMDMAP.nested_types_by_name['CmdsEntry']
_SIMINFOMAP = DESCRIPTOR.message_types_by_name['SimInfoMap']
_SIMINFOMAP_INFOSENTRY = _SIMINFOMAP.nested_types_by_name['InfosEntry']
_SERVICEINFO_TYPE = _SERVICEINFO.enum_types_by_name['Type']
ServiceInfo = _reflection.GeneratedProtocolMessageType(
    'ServiceInfo',
    (_message.Message,),
    {
        'DESCRIPTOR': _SERVICEINFO,
        '__module__': 'protos.bff_pb2'
        # @@protoc_insertion_point(class_scope:game.bff.ServiceInfo)
    })
_sym_db.RegisterMessage(ServiceInfo)

ServiceIdList = _reflection.GeneratedProtocolMessageType(
    'ServiceIdList',
    (_message.Message,),
    {
        'DESCRIPTOR': _SERVICEIDLIST,
        '__module__': 'protos.bff_pb2'
        # @@protoc_insertion_point(class_scope:game.bff.ServiceIdList)
    })
_sym_db.RegisterMessage(ServiceIdList)

ServiceInfoList = _reflection.GeneratedProtocolMessageType(
    'ServiceInfoList',
    (_message.Message,),
    {
        'DESCRIPTOR': _SERVICEINFOLIST,
        '__module__': 'protos.bff_pb2'
        # @@protoc_insertion_point(class_scope:game.bff.ServiceInfoList)
    })
_sym_db.RegisterMessage(ServiceInfoList)

ServiceInfoMap = _reflection.GeneratedProtocolMessageType(
    'ServiceInfoMap',
    (_message.Message,),
    {
        'ServicesEntry':
            _reflection.GeneratedProtocolMessageType(
                'ServicesEntry',
                (_message.Message,),
                {
                    'DESCRIPTOR': _SERVICEINFOMAP_SERVICESENTRY,
                    '__module__': 'protos.bff_pb2'
                    # @@protoc_insertion_point(class_scope:game.bff.ServiceInfoMap.ServicesEntry)
                }),
        'DESCRIPTOR':
            _SERVICEINFOMAP,
        '__module__':
            'protos.bff_pb2'
        # @@protoc_insertion_point(class_scope:game.bff.ServiceInfoMap)
    })
_sym_db.RegisterMessage(ServiceInfoMap)
_sym_db.RegisterMessage(ServiceInfoMap.ServicesEntry)

DataConfig = _reflection.GeneratedProtocolMessageType(
    'DataConfig',
    (_message.Message,),
    {
        'Type':
            _reflection.GeneratedProtocolMessageType(
                'Type',
                (_message.Message,),
                {
                    'FieldsEntry':
                        _reflection.GeneratedProtocolMessageType(
                            'FieldsEntry',
                            (_message.Message,),
                            {
                                'DESCRIPTOR': _DATACONFIG_TYPE_FIELDSENTRY,
                                '__module__': 'protos.bff_pb2'
                                # @@protoc_insertion_point(class_scope:game.bff.DataConfig.Type.FieldsEntry)
                            }),
                    'DESCRIPTOR':
                        _DATACONFIG_TYPE,
                    '__module__':
                        'protos.bff_pb2'
                    # @@protoc_insertion_point(class_scope:game.bff.DataConfig.Type)
                }),
        'Param':
            _reflection.GeneratedProtocolMessageType(
                'Param',
                (_message.Message,),
                {
                    'DESCRIPTOR': _DATACONFIG_PARAM,
                    '__module__': 'protos.bff_pb2'
                    # @@protoc_insertion_point(class_scope:game.bff.DataConfig.Param)
                }),
        'Model':
            _reflection.GeneratedProtocolMessageType(
                'Model',
                (_message.Message,),
                {
                    'InputParamsEntry':
                        _reflection.GeneratedProtocolMessageType(
                            'InputParamsEntry',
                            (_message.Message,),
                            {
                                'DESCRIPTOR': _DATACONFIG_MODEL_INPUTPARAMSENTRY,
                                '__module__': 'protos.bff_pb2'
                                # @@protoc_insertion_point(class_scope:game.bff.DataConfig.Model.InputParamsEntry)
                            }),
                    'OutputParamsEntry':
                        _reflection.GeneratedProtocolMessageType(
                            'OutputParamsEntry',
                            (_message.Message,),
                            {
                                'DESCRIPTOR': _DATACONFIG_MODEL_OUTPUTPARAMSENTRY,
                                '__module__': 'protos.bff_pb2'
                                # @@protoc_insertion_point(class_scope:game.bff.DataConfig.Model.OutputParamsEntry)
                            }),
                    'DESCRIPTOR':
                        _DATACONFIG_MODEL,
                    '__module__':
                        'protos.bff_pb2'
                    # @@protoc_insertion_point(class_scope:game.bff.DataConfig.Model)
                }),
        'TypesEntry':
            _reflection.GeneratedProtocolMessageType(
                'TypesEntry',
                (_message.Message,),
                {
                    'DESCRIPTOR': _DATACONFIG_TYPESENTRY,
                    '__module__': 'protos.bff_pb2'
                    # @@protoc_insertion_point(class_scope:game.bff.DataConfig.TypesEntry)
                }),
        'DataEntry':
            _reflection.GeneratedProtocolMessageType(
                'DataEntry',
                (_message.Message,),
                {
                    'DESCRIPTOR': _DATACONFIG_DATAENTRY,
                    '__module__': 'protos.bff_pb2'
                    # @@protoc_insertion_point(class_scope:game.bff.DataConfig.DataEntry)
                }),
        'DESCRIPTOR':
            _DATACONFIG,
        '__module__':
            'protos.bff_pb2'
        # @@protoc_insertion_point(class_scope:game.bff.DataConfig)
    })
_sym_db.RegisterMessage(DataConfig)
_sym_db.RegisterMessage(DataConfig.Type)
_sym_db.RegisterMessage(DataConfig.Type.FieldsEntry)
_sym_db.RegisterMessage(DataConfig.Param)
_sym_db.RegisterMessage(DataConfig.Model)
_sym_db.RegisterMessage(DataConfig.Model.InputParamsEntry)
_sym_db.RegisterMessage(DataConfig.Model.OutputParamsEntry)
_sym_db.RegisterMessage(DataConfig.TypesEntry)
_sym_db.RegisterMessage(DataConfig.DataEntry)

RouteConfig = _reflection.GeneratedProtocolMessageType(
    'RouteConfig',
    (_message.Message,),
    {
        'Config':
            _reflection.GeneratedProtocolMessageType(
                'Config',
                (_message.Message,),
                {
                    'DESCRIPTOR': _ROUTECONFIG_CONFIG,
                    '__module__': 'protos.bff_pb2'
                    # @@protoc_insertion_point(class_scope:game.bff.RouteConfig.Config)
                }),
        'Route':
            _reflection.GeneratedProtocolMessageType(
                'Route',
                (_message.Message,),
                {
                    'ConfigsEntry':
                        _reflection.GeneratedProtocolMessageType(
                            'ConfigsEntry',
                            (_message.Message,),
                            {
                                'DESCRIPTOR': _ROUTECONFIG_ROUTE_CONFIGSENTRY,
                                '__module__': 'protos.bff_pb2'
                                # @@protoc_insertion_point(class_scope:game.bff.RouteConfig.Route.ConfigsEntry)
                            }),
                    'DESCRIPTOR':
                        _ROUTECONFIG_ROUTE,
                    '__module__':
                        'protos.bff_pb2'
                    # @@protoc_insertion_point(class_scope:game.bff.RouteConfig.Route)
                }),
        'RoutesEntry':
            _reflection.GeneratedProtocolMessageType(
                'RoutesEntry',
                (_message.Message,),
                {
                    'DESCRIPTOR': _ROUTECONFIG_ROUTESENTRY,
                    '__module__': 'protos.bff_pb2'
                    # @@protoc_insertion_point(class_scope:game.bff.RouteConfig.RoutesEntry)
                }),
        'DESCRIPTOR':
            _ROUTECONFIG,
        '__module__':
            'protos.bff_pb2'
        # @@protoc_insertion_point(class_scope:game.bff.RouteConfig)
    })
_sym_db.RegisterMessage(RouteConfig)
_sym_db.RegisterMessage(RouteConfig.Config)
_sym_db.RegisterMessage(RouteConfig.Route)
_sym_db.RegisterMessage(RouteConfig.Route.ConfigsEntry)
_sym_db.RegisterMessage(RouteConfig.RoutesEntry)

ServiceStateMap = _reflection.GeneratedProtocolMessageType(
    'ServiceStateMap',
    (_message.Message,),
    {
        'StatesEntry':
            _reflection.GeneratedProtocolMessageType(
                'StatesEntry',
                (_message.Message,),
                {
                    'DESCRIPTOR': _SERVICESTATEMAP_STATESENTRY,
                    '__module__': 'protos.bff_pb2'
                    # @@protoc_insertion_point(class_scope:game.bff.ServiceStateMap.StatesEntry)
                }),
        'DESCRIPTOR':
            _SERVICESTATEMAP,
        '__module__':
            'protos.bff_pb2'
        # @@protoc_insertion_point(class_scope:game.bff.ServiceStateMap)
    })
_sym_db.RegisterMessage(ServiceStateMap)
_sym_db.RegisterMessage(ServiceStateMap.StatesEntry)

AgentConfigMap = _reflection.GeneratedProtocolMessageType(
    'AgentConfigMap',
    (_message.Message,),
    {
        'ConfigsEntry':
            _reflection.GeneratedProtocolMessageType(
                'ConfigsEntry',
                (_message.Message,),
                {
                    'DESCRIPTOR': _AGENTCONFIGMAP_CONFIGSENTRY,
                    '__module__': 'protos.bff_pb2'
                    # @@protoc_insertion_point(class_scope:game.bff.AgentConfigMap.ConfigsEntry)
                }),
        'DESCRIPTOR':
            _AGENTCONFIGMAP,
        '__module__':
            'protos.bff_pb2'
        # @@protoc_insertion_point(class_scope:game.bff.AgentConfigMap)
    })
_sym_db.RegisterMessage(AgentConfigMap)
_sym_db.RegisterMessage(AgentConfigMap.ConfigsEntry)

AgentModeMap = _reflection.GeneratedProtocolMessageType(
    'AgentModeMap',
    (_message.Message,),
    {
        'ModesEntry':
            _reflection.GeneratedProtocolMessageType(
                'ModesEntry',
                (_message.Message,),
                {
                    'DESCRIPTOR': _AGENTMODEMAP_MODESENTRY,
                    '__module__': 'protos.bff_pb2'
                    # @@protoc_insertion_point(class_scope:game.bff.AgentModeMap.ModesEntry)
                }),
        'DESCRIPTOR':
            _AGENTMODEMAP,
        '__module__':
            'protos.bff_pb2'
        # @@protoc_insertion_point(class_scope:game.bff.AgentModeMap)
    })
_sym_db.RegisterMessage(AgentModeMap)
_sym_db.RegisterMessage(AgentModeMap.ModesEntry)

ModelWeightsMap = _reflection.GeneratedProtocolMessageType(
    'ModelWeightsMap',
    (_message.Message,),
    {
        'WeightsEntry':
            _reflection.GeneratedProtocolMessageType(
                'WeightsEntry',
                (_message.Message,),
                {
                    'DESCRIPTOR': _MODELWEIGHTSMAP_WEIGHTSENTRY,
                    '__module__': 'protos.bff_pb2'
                    # @@protoc_insertion_point(class_scope:game.bff.ModelWeightsMap.WeightsEntry)
                }),
        'DESCRIPTOR':
            _MODELWEIGHTSMAP,
        '__module__':
            'protos.bff_pb2'
        # @@protoc_insertion_point(class_scope:game.bff.ModelWeightsMap)
    })
_sym_db.RegisterMessage(ModelWeightsMap)
_sym_db.RegisterMessage(ModelWeightsMap.WeightsEntry)

ModelBufferMap = _reflection.GeneratedProtocolMessageType(
    'ModelBufferMap',
    (_message.Message,),
    {
        'BuffersEntry':
            _reflection.GeneratedProtocolMessageType(
                'BuffersEntry',
                (_message.Message,),
                {
                    'DESCRIPTOR': _MODELBUFFERMAP_BUFFERSENTRY,
                    '__module__': 'protos.bff_pb2'
                    # @@protoc_insertion_point(class_scope:game.bff.ModelBufferMap.BuffersEntry)
                }),
        'DESCRIPTOR':
            _MODELBUFFERMAP,
        '__module__':
            'protos.bff_pb2'
        # @@protoc_insertion_point(class_scope:game.bff.ModelBufferMap)
    })
_sym_db.RegisterMessage(ModelBufferMap)
_sym_db.RegisterMessage(ModelBufferMap.BuffersEntry)

ModelStatusMap = _reflection.GeneratedProtocolMessageType(
    'ModelStatusMap',
    (_message.Message,),
    {
        'StatusEntry':
            _reflection.GeneratedProtocolMessageType(
                'StatusEntry',
                (_message.Message,),
                {
                    'DESCRIPTOR': _MODELSTATUSMAP_STATUSENTRY,
                    '__module__': 'protos.bff_pb2'
                    # @@protoc_insertion_point(class_scope:game.bff.ModelStatusMap.StatusEntry)
                }),
        'DESCRIPTOR':
            _MODELSTATUSMAP,
        '__module__':
            'protos.bff_pb2'
        # @@protoc_insertion_point(class_scope:game.bff.ModelStatusMap)
    })
_sym_db.RegisterMessage(ModelStatusMap)
_sym_db.RegisterMessage(ModelStatusMap.StatusEntry)

SimenvConfigMap = _reflection.GeneratedProtocolMessageType(
    'SimenvConfigMap',
    (_message.Message,),
    {
        'ConfigsEntry':
            _reflection.GeneratedProtocolMessageType(
                'ConfigsEntry',
                (_message.Message,),
                {
                    'DESCRIPTOR': _SIMENVCONFIGMAP_CONFIGSENTRY,
                    '__module__': 'protos.bff_pb2'
                    # @@protoc_insertion_point(class_scope:game.bff.SimenvConfigMap.ConfigsEntry)
                }),
        'DESCRIPTOR':
            _SIMENVCONFIGMAP,
        '__module__':
            'protos.bff_pb2'
        # @@protoc_insertion_point(class_scope:game.bff.SimenvConfigMap)
    })
_sym_db.RegisterMessage(SimenvConfigMap)
_sym_db.RegisterMessage(SimenvConfigMap.ConfigsEntry)

SimCmdMap = _reflection.GeneratedProtocolMessageType(
    'SimCmdMap',
    (_message.Message,),
    {
        'CmdsEntry':
            _reflection.GeneratedProtocolMessageType(
                'CmdsEntry',
                (_message.Message,),
                {
                    'DESCRIPTOR': _SIMCMDMAP_CMDSENTRY,
                    '__module__': 'protos.bff_pb2'
                    # @@protoc_insertion_point(class_scope:game.bff.SimCmdMap.CmdsEntry)
                }),
        'DESCRIPTOR':
            _SIMCMDMAP,
        '__module__':
            'protos.bff_pb2'
        # @@protoc_insertion_point(class_scope:game.bff.SimCmdMap)
    })
_sym_db.RegisterMessage(SimCmdMap)
_sym_db.RegisterMessage(SimCmdMap.CmdsEntry)

SimInfoMap = _reflection.GeneratedProtocolMessageType(
    'SimInfoMap',
    (_message.Message,),
    {
        'InfosEntry':
            _reflection.GeneratedProtocolMessageType(
                'InfosEntry',
                (_message.Message,),
                {
                    'DESCRIPTOR': _SIMINFOMAP_INFOSENTRY,
                    '__module__': 'protos.bff_pb2'
                    # @@protoc_insertion_point(class_scope:game.bff.SimInfoMap.InfosEntry)
                }),
        'DESCRIPTOR':
            _SIMINFOMAP,
        '__module__':
            'protos.bff_pb2'
        # @@protoc_insertion_point(class_scope:game.bff.SimInfoMap)
    })
_sym_db.RegisterMessage(SimInfoMap)
_sym_db.RegisterMessage(SimInfoMap.InfosEntry)

_BFF = DESCRIPTOR.services_by_name['BFF']
if _descriptor._USE_C_DESCRIPTORS == False:

    DESCRIPTOR._options = None
    _SERVICEINFOMAP_SERVICESENTRY._options = None
    _SERVICEINFOMAP_SERVICESENTRY._serialized_options = b'8\001'
    _DATACONFIG_TYPE_FIELDSENTRY._options = None
    _DATACONFIG_TYPE_FIELDSENTRY._serialized_options = b'8\001'
    _DATACONFIG_MODEL_INPUTPARAMSENTRY._options = None
    _DATACONFIG_MODEL_INPUTPARAMSENTRY._serialized_options = b'8\001'
    _DATACONFIG_MODEL_OUTPUTPARAMSENTRY._options = None
    _DATACONFIG_MODEL_OUTPUTPARAMSENTRY._serialized_options = b'8\001'
    _DATACONFIG_TYPESENTRY._options = None
    _DATACONFIG_TYPESENTRY._serialized_options = b'8\001'
    _DATACONFIG_DATAENTRY._options = None
    _DATACONFIG_DATAENTRY._serialized_options = b'8\001'
    _ROUTECONFIG_ROUTE_CONFIGSENTRY._options = None
    _ROUTECONFIG_ROUTE_CONFIGSENTRY._serialized_options = b'8\001'
    _ROUTECONFIG_ROUTESENTRY._options = None
    _ROUTECONFIG_ROUTESENTRY._serialized_options = b'8\001'
    _SERVICESTATEMAP_STATESENTRY._options = None
    _SERVICESTATEMAP_STATESENTRY._serialized_options = b'8\001'
    _AGENTCONFIGMAP_CONFIGSENTRY._options = None
    _AGENTCONFIGMAP_CONFIGSENTRY._serialized_options = b'8\001'
    _AGENTMODEMAP_MODESENTRY._options = None
    _AGENTMODEMAP_MODESENTRY._serialized_options = b'8\001'
    _MODELWEIGHTSMAP_WEIGHTSENTRY._options = None
    _MODELWEIGHTSMAP_WEIGHTSENTRY._serialized_options = b'8\001'
    _MODELBUFFERMAP_BUFFERSENTRY._options = None
    _MODELBUFFERMAP_BUFFERSENTRY._serialized_options = b'8\001'
    _MODELSTATUSMAP_STATUSENTRY._options = None
    _MODELSTATUSMAP_STATUSENTRY._serialized_options = b'8\001'
    _SIMENVCONFIGMAP_CONFIGSENTRY._options = None
    _SIMENVCONFIGMAP_CONFIGSENTRY._serialized_options = b'8\001'
    _SIMCMDMAP_CMDSENTRY._options = None
    _SIMCMDMAP_CMDSENTRY._serialized_options = b'8\001'
    _SIMINFOMAP_INFOSENTRY._options = None
    _SIMINFOMAP_INFOSENTRY._serialized_options = b'8\001'
    _SERVICEINFO._serialized_start = 92
    _SERVICEINFO._serialized_end = 232
    _SERVICEINFO_TYPE._serialized_start = 203
    _SERVICEINFO_TYPE._serialized_end = 232
    _SERVICEIDLIST._serialized_start = 234
    _SERVICEIDLIST._serialized_end = 262
    _SERVICEINFOLIST._serialized_start = 264
    _SERVICEINFOLIST._serialized_end = 322
    _SERVICEINFOMAP._serialized_start = 325
    _SERVICEINFOMAP._serialized_end = 471
    _SERVICEINFOMAP_SERVICESENTRY._serialized_start = 401
    _SERVICEINFOMAP_SERVICESENTRY._serialized_end = 471
    _DATACONFIG._serialized_start = 474
    _DATACONFIG._serialized_end = 1209
    _DATACONFIG_TYPE._serialized_start = 582
    _DATACONFIG_TYPE._serialized_end = 690
    _DATACONFIG_TYPE_FIELDSENTRY._serialized_start = 645
    _DATACONFIG_TYPE_FIELDSENTRY._serialized_end = 690
    _DATACONFIG_PARAM._serialized_start = 692
    _DATACONFIG_PARAM._serialized_end = 742
    _DATACONFIG_MODEL._serialized_start = 745
    _DATACONFIG_MODEL._serialized_end = 1063
    _DATACONFIG_MODEL_INPUTPARAMSENTRY._serialized_start = 904
    _DATACONFIG_MODEL_INPUTPARAMSENTRY._serialized_end = 982
    _DATACONFIG_MODEL_OUTPUTPARAMSENTRY._serialized_start = 984
    _DATACONFIG_MODEL_OUTPUTPARAMSENTRY._serialized_end = 1063
    _DATACONFIG_TYPESENTRY._serialized_start = 1065
    _DATACONFIG_TYPESENTRY._serialized_end = 1136
    _DATACONFIG_DATAENTRY._serialized_start = 1138
    _DATACONFIG_DATAENTRY._serialized_end = 1209
    _ROUTECONFIG._serialized_start = 1212
    _ROUTECONFIG._serialized_end = 1600
    _ROUTECONFIG_CONFIG._serialized_start = 1325
    _ROUTECONFIG_CONFIG._serialized_end = 1363
    _ROUTECONFIG_ROUTE._serialized_start = 1366
    _ROUTECONFIG_ROUTE._serialized_end = 1524
    _ROUTECONFIG_ROUTE_CONFIGSENTRY._serialized_start = 1448
    _ROUTECONFIG_ROUTE_CONFIGSENTRY._serialized_end = 1524
    _ROUTECONFIG_ROUTESENTRY._serialized_start = 1526
    _ROUTECONFIG_ROUTESENTRY._serialized_end = 1600
    _SERVICESTATEMAP._serialized_start = 1603
    _SERVICESTATEMAP._serialized_end = 1748
    _SERVICESTATEMAP_STATESENTRY._serialized_start = 1677
    _SERVICESTATEMAP_STATESENTRY._serialized_end = 1748
    _AGENTCONFIGMAP._serialized_start = 1751
    _AGENTCONFIGMAP._serialized_end = 1896
    _AGENTCONFIGMAP_CONFIGSENTRY._serialized_start = 1825
    _AGENTCONFIGMAP_CONFIGSENTRY._serialized_end = 1896
    _AGENTMODEMAP._serialized_start = 1899
    _AGENTMODEMAP._serialized_end = 2032
    _AGENTMODEMAP_MODESENTRY._serialized_start = 1965
    _AGENTMODEMAP_MODESENTRY._serialized_end = 2032
    _MODELWEIGHTSMAP._serialized_start = 2035
    _MODELWEIGHTSMAP._serialized_end = 2183
    _MODELWEIGHTSMAP_WEIGHTSENTRY._serialized_start = 2111
    _MODELWEIGHTSMAP_WEIGHTSENTRY._serialized_end = 2183
    _MODELBUFFERMAP._serialized_start = 2186
    _MODELBUFFERMAP._serialized_end = 2331
    _MODELBUFFERMAP_BUFFERSENTRY._serialized_start = 2260
    _MODELBUFFERMAP_BUFFERSENTRY._serialized_end = 2331
    _MODELSTATUSMAP._serialized_start = 2334
    _MODELSTATUSMAP._serialized_end = 2476
    _MODELSTATUSMAP_STATUSENTRY._serialized_start = 2406
    _MODELSTATUSMAP_STATUSENTRY._serialized_end = 2476
    _SIMENVCONFIGMAP._serialized_start = 2479
    _SIMENVCONFIGMAP._serialized_end = 2628
    _SIMENVCONFIGMAP_CONFIGSENTRY._serialized_start = 2555
    _SIMENVCONFIGMAP_CONFIGSENTRY._serialized_end = 2628
    _SIMCMDMAP._serialized_start = 2630
    _SIMCMDMAP._serialized_end = 2752
    _SIMCMDMAP_CMDSENTRY._serialized_start = 2688
    _SIMCMDMAP_CMDSENTRY._serialized_end = 2752
    _SIMINFOMAP._serialized_start = 2755
    _SIMINFOMAP._serialized_end = 2883
    _SIMINFOMAP_INFOSENTRY._serialized_start = 2817
    _SIMINFOMAP_INFOSENTRY._serialized_end = 2883
    _BFF._serialized_start = 2886
    _BFF._serialized_end = 4689
# @@protoc_insertion_point(module_scope)
