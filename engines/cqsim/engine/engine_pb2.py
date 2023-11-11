"""Generated protocol buffer code."""
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database

_sym_db = _symbol_database.Default()
from google.protobuf import timestamp_pb2 as google_dot_protobuf_dot_timestamp__pb2
from google.protobuf import duration_pb2 as google_dot_protobuf_dot_duration__pb2

DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(
    b'\n\x0cengine.proto\x12\rcqsim.control\x1a\x1fgoogle/protobuf/timestamp.proto\x1a\x1egoogle/protobuf/duration.proto"C\n\x0eCommonResponse\x12\x0b\n\x03msg\x18\x01 \x01(\t\x12\x0c\n\x04code\x18\x02 \x01(\x04\x12\x16\n\x0eerror_location\x18\x03 \x01(\x04"\x1a\n\x0cScenarioInfo\x12\n\n\x02ID\x18\x01 \x01(\x04":\n\x0fLogLevelRequest\x12\'\n\x06levels\x18\x01 \x03(\x0e2\x17.cqsim.control.LogLevel"E\n\x0eErrMsgResponse\x12\x0b\n\x03msg\x18\x01 \x01(\t\x12&\n\x05level\x18\x02 \x01(\x0e2\x17.cqsim.control.LogLevel"\x0f\n\rCommonRequest"P\n\x14NodeJoinExitResponse\x12\x0f\n\x07address\x18\x01 \x01(\t\x12\x16\n\x0eis_master_node\x18\x02 \x01(\x08\x12\x0f\n\x07is_join\x18\x03 \x01(\x08"$\n\x11EntityListRequest\x12\x0f\n\x07task_id\x18\x01 \x01(\x04"&\n\x06Entity\x12\n\n\x02id\x18\x01 \x01(\t\x12\x10\n\x08model_id\x18\x02 \x01(\t"\xd2\x01\n\x12EntityListResponse\x12A\n\x0bentity_list\x18\x01 \x03(\x0b2,.cqsim.control.EntityListResponse.EntityInfo\x12\x18\n\x10entity_list_json\x18\x02 \x01(\t\x1a_\n\nEntityInfo\x12\n\n\x02id\x18\x01 \x01(\t\x12\x0c\n\x04name\x18\x02 \x01(\t\x12\x12\n\nmodel_name\x18\x03 \x01(\t\x12\x10\n\x08model_id\x18\x04 \x01(\t\x12\x11\n\tcamp_name\x18\x05 \x01(\t"F\n\nNodeDesign\x12\x0f\n\x07address\x18\x01 \x01(\t\x12\'\n\x08entities\x18\x02 \x03(\x0b2\x15.cqsim.control.Entity"\xc6\x02\n\x08InitInfo\x12>\n\x11one_sample_config\x18\x01 \x01(\x0b2!.cqsim.control.InitInfo.OneSampleH\x00\x12B\n\x13multi_sample_config\x18\x02 \x01(\x0b2#.cqsim.control.InitInfo.MultiSampleH\x00\x12\x11\n\x07data_id\x18\x03 \x01(\x04H\x00\x1aF\n\tOneSample\x12\x0f\n\x07task_id\x18\x01 \x01(\x04\x12(\n\x05nodes\x18\x02 \x03(\x0b2\x19.cqsim.control.NodeDesign\x1aN\n\x0bMultiSample\x12\x15\n\rexp_design_id\x18\x01 \x01(\x04\x12(\n\x05nodes\x18\x02 \x03(\x0b2\x19.cqsim.control.NodeDesignB\x0b\n\tinit_info"\x19\n\x08HttpInfo\x12\r\n\x05token\x18\x01 \x01(\t"\xc9\x03\n\nControlCmd\x124\n\x0esim_start_time\x18\x01 \x01(\x0b2\x1a.google.protobuf.TimestampH\x00\x121\n\x0csim_duration\x18\x02 \x01(\x0b2\x19.google.protobuf.DurationH\x00\x127\n\x07run_cmd\x18\x03 \x01(\x0e2$.cqsim.control.ControlCmd.RunCmdTypeH\x00\x12\x13\n\ttime_step\x18\x04 \x01(\rH\x00\x12\x15\n\x0bspeed_ratio\x18\x05 \x01(\x01H\x00\x12.\n\x04mode\x18\x06 \x01(\x0e2\x1e.cqsim.control.ControlCmd.ModeH\x00\x12\x16\n\x0cschedule_val\x18\x07 \x01(\x01H\x00\x12\x19\n\x0fback_track_time\x18\x08 \x01(\x03H\x00"U\n\nRunCmdType\x12\t\n\x05START\x10\x00\x12\x0b\n\x07SUSPEND\x10\x01\x12\x0c\n\x08CONTINUE\x10\x02\x12\x08\n\x04STOP\x10\x03\x12\x17\n\x13STOP_CURRENT_SAMPLE\x10\x04",\n\x04Mode\x12\n\n\x06RECORD\x10\x00\x12\n\n\x06REPLAY\x10\x01\x12\x0c\n\x08CONVERSE\x10\x02B\x05\n\x03cmd"\x90\x02\n\x0fEngineNodeState\x12\x0f\n\x07address\x18\x01 \x01(\t\x123\n\x05state\x18\x02 \x01(\x0e2$.cqsim.control.EngineNodeState.State\x12\x10\n\x08cpu_load\x18\x03 \x01(\x01\x12\x13\n\x0bmemory_load\x18\x04 \x01(\x01\x12\x14\n\x0cnetwork_load\x18\x05 \x01(\x01\x12\x16\n\x0eis_master_node\x18\x06 \x01(\x08"b\n\x05State\x12\x0c\n\x08UNINITED\x10\x00\x12\n\n\x06INITED\x10\x01\x12\x0b\n\x07RUNNING\x10\x02\x12\r\n\tSUSPENDED\x10\x03\x12\x0b\n\x07STOPPED\x10\x04\x12\t\n\x05ERROR\x10\x05\x12\x0b\n\x07OFFLINE\x10\x06"\xd4\x02\n\x0fSysInfoResponse\x124\n\x10sim_current_time\x18\x01 \x01(\x0b2\x1a.google.protobuf.Timestamp\x12/\n\x0csim_duration\x18\x02 \x01(\x0b2\x19.google.protobuf.Duration\x120\n\rreal_duration\x18\x03 \x01(\x0b2\x19.google.protobuf.Duration\x12\x15\n\rsim_time_step\x18\x04 \x01(\r\x12\x13\n\x0bspeed_ratio\x18\x05 \x01(\x01\x12\x18\n\x10real_speed_ratio\x18\x06 \x01(\x01\x122\n\nnode_state\x18\x07 \x03(\x0b2\x1e.cqsim.control.EngineNodeState\x12\x19\n\x11current_sample_id\x18\x08 \x01(\r\x12\x13\n\x0bsim_address\x18\t \x03(\t"\xf2\x02\n\x13DataSysInfoResponse\x12>\n\x05state\x18\x01 \x01(\x0e2/.cqsim.control.DataSysInfoResponse.ServiceState\x120\n\x0ccurrent_time\x18\x02 \x01(\x0b2\x1a.google.protobuf.Timestamp\x12+\n\x08duration\x18\x03 \x01(\x0b2\x19.google.protobuf.Duration\x12\x14\n\x0cschedule_val\x18\x04 \x01(\x01\x12\x10\n\x08cpu_load\x18\x05 \x01(\x01\x12\x13\n\x0bmemory_load\x18\x06 \x01(\x01\x12\x14\n\x0cnetwork_load\x18\x07 \x01(\x01"i\n\x0cServiceState\x12\x0c\n\x08UNINITED\x10\x00\x12\n\n\x06INITED\x10\x01\x12\x0b\n\x07RUNNING\x10\x02\x12\r\n\tSUSPENDED\x10\x03\x12\x0b\n\x07STOPPED\x10\x04\x12\t\n\x05ERROR\x10\x05\x12\x0b\n\x07OFFLINE\x10\x06"\x1d\n\x0cNodeResponse\x12\r\n\x05nodes\x18\x01 \x03(\t"&\n\x0eInitedResponse\x12\x14\n\x0cinit_percent\x18\x01 \x01(\r*S\n\x08LogLevel\x12\t\n\x05TRACE\x10\x00\x12\t\n\x05DEBUG\x10\x01\x12\x08\n\x04INFO\x10\x02\x12\x08\n\x04WARN\x10\x03\x12\t\n\x05ERROR\x10\x04\x12\x12\n\x0eCRITICAL_ERROR\x10\x052\x82\x08\n\rSimController\x12T\n\rGetEntityList\x12 .cqsim.control.EntityListRequest\x1a!.cqsim.control.EntityListResponse\x12>\n\x04Init\x12\x17.cqsim.control.InitInfo\x1a\x1d.cqsim.control.CommonResponse\x12C\n\x07Control\x12\x19.cqsim.control.ControlCmd\x1a\x1d.cqsim.control.CommonResponse\x12L\n\nGetSysInfo\x12\x1c.cqsim.control.CommonRequest\x1a\x1e.cqsim.control.SysInfoResponse0\x01\x12T\n\x0eGetDataSysInfo\x12\x1c.cqsim.control.CommonRequest\x1a".cqsim.control.DataSysInfoResponse0\x01\x12G\n\nGetAllNode\x12\x1c.cqsim.control.CommonRequest\x1a\x1b.cqsim.control.NodeResponse\x12Q\n\x10GetInitedPercent\x12\x1c.cqsim.control.CommonRequest\x1a\x1d.cqsim.control.InitedResponse0\x01\x12L\n\x0bGetErrorMsg\x12\x1c.cqsim.control.CommonRequest\x1a\x1d.cqsim.control.ErrMsgResponse0\x01\x12V\n\x0fGetNodeJoinExit\x12\x1c.cqsim.control.CommonRequest\x1a#.cqsim.control.NodeJoinExitResponse0\x01\x12L\n\x0bSetLogLevel\x12\x1e.cqsim.control.LogLevelRequest\x1a\x1d.cqsim.control.CommonResponse\x12E\n\x0bSetHttpInfo\x12\x17.cqsim.control.HttpInfo\x1a\x1d.cqsim.control.CommonResponse\x12L\n\x0fGetScenarioInfo\x12\x1c.cqsim.control.CommonRequest\x1a\x1b.cqsim.control.ScenarioInfo\x12M\n\x0eGetDataAddress\x12\x1c.cqsim.control.CommonRequest\x1a\x1d.cqsim.control.CommonResponseb\x06proto3'
)
_LOGLEVEL = DESCRIPTOR.enum_types_by_name['LogLevel']
LogLevel = enum_type_wrapper.EnumTypeWrapper(_LOGLEVEL)
TRACE = 0
DEBUG = 1
INFO = 2
WARN = 3
ERROR = 4
CRITICAL_ERROR = 5
_COMMONRESPONSE = DESCRIPTOR.message_types_by_name['CommonResponse']
_SCENARIOINFO = DESCRIPTOR.message_types_by_name['ScenarioInfo']
_LOGLEVELREQUEST = DESCRIPTOR.message_types_by_name['LogLevelRequest']
_ERRMSGRESPONSE = DESCRIPTOR.message_types_by_name['ErrMsgResponse']
_COMMONREQUEST = DESCRIPTOR.message_types_by_name['CommonRequest']
_NODEJOINEXITRESPONSE = DESCRIPTOR.message_types_by_name['NodeJoinExitResponse']
_ENTITYLISTREQUEST = DESCRIPTOR.message_types_by_name['EntityListRequest']
_ENTITY = DESCRIPTOR.message_types_by_name['Entity']
_ENTITYLISTRESPONSE = DESCRIPTOR.message_types_by_name['EntityListResponse']
_ENTITYLISTRESPONSE_ENTITYINFO = _ENTITYLISTRESPONSE.nested_types_by_name['EntityInfo']
_NODEDESIGN = DESCRIPTOR.message_types_by_name['NodeDesign']
_INITINFO = DESCRIPTOR.message_types_by_name['InitInfo']
_INITINFO_ONESAMPLE = _INITINFO.nested_types_by_name['OneSample']
_INITINFO_MULTISAMPLE = _INITINFO.nested_types_by_name['MultiSample']
_HTTPINFO = DESCRIPTOR.message_types_by_name['HttpInfo']
_CONTROLCMD = DESCRIPTOR.message_types_by_name['ControlCmd']
_ENGINENODESTATE = DESCRIPTOR.message_types_by_name['EngineNodeState']
_SYSINFORESPONSE = DESCRIPTOR.message_types_by_name['SysInfoResponse']
_DATASYSINFORESPONSE = DESCRIPTOR.message_types_by_name['DataSysInfoResponse']
_NODERESPONSE = DESCRIPTOR.message_types_by_name['NodeResponse']
_INITEDRESPONSE = DESCRIPTOR.message_types_by_name['InitedResponse']
_CONTROLCMD_RUNCMDTYPE = _CONTROLCMD.enum_types_by_name['RunCmdType']
_CONTROLCMD_MODE = _CONTROLCMD.enum_types_by_name['Mode']
_ENGINENODESTATE_STATE = _ENGINENODESTATE.enum_types_by_name['State']
_DATASYSINFORESPONSE_SERVICESTATE = _DATASYSINFORESPONSE.enum_types_by_name['ServiceState']
CommonResponse = _reflection.GeneratedProtocolMessageType('CommonResponse', (_message.Message,), {
    'DESCRIPTOR': _COMMONRESPONSE,
    '__module__': 'engine_pb2'
})
_sym_db.RegisterMessage(CommonResponse)
ScenarioInfo = _reflection.GeneratedProtocolMessageType('ScenarioInfo', (_message.Message,), {
    'DESCRIPTOR': _SCENARIOINFO,
    '__module__': 'engine_pb2'
})
_sym_db.RegisterMessage(ScenarioInfo)
LogLevelRequest = _reflection.GeneratedProtocolMessageType('LogLevelRequest', (_message.Message,), {
    'DESCRIPTOR': _LOGLEVELREQUEST,
    '__module__': 'engine_pb2'
})
_sym_db.RegisterMessage(LogLevelRequest)
ErrMsgResponse = _reflection.GeneratedProtocolMessageType('ErrMsgResponse', (_message.Message,), {
    'DESCRIPTOR': _ERRMSGRESPONSE,
    '__module__': 'engine_pb2'
})
_sym_db.RegisterMessage(ErrMsgResponse)
CommonRequest = _reflection.GeneratedProtocolMessageType('CommonRequest', (_message.Message,), {
    'DESCRIPTOR': _COMMONREQUEST,
    '__module__': 'engine_pb2'
})
_sym_db.RegisterMessage(CommonRequest)
NodeJoinExitResponse = _reflection.GeneratedProtocolMessageType('NodeJoinExitResponse', (_message.Message,), {
    'DESCRIPTOR': _NODEJOINEXITRESPONSE,
    '__module__': 'engine_pb2'
})
_sym_db.RegisterMessage(NodeJoinExitResponse)
EntityListRequest = _reflection.GeneratedProtocolMessageType('EntityListRequest', (_message.Message,), {
    'DESCRIPTOR': _ENTITYLISTREQUEST,
    '__module__': 'engine_pb2'
})
_sym_db.RegisterMessage(EntityListRequest)
Entity = _reflection.GeneratedProtocolMessageType('Entity', (_message.Message,), {
    'DESCRIPTOR': _ENTITY,
    '__module__': 'engine_pb2'
})
_sym_db.RegisterMessage(Entity)
EntityListResponse = _reflection.GeneratedProtocolMessageType(
    'EntityListResponse', (_message.Message,), {
        'EntityInfo':
            _reflection.GeneratedProtocolMessageType('EntityInfo', (_message.Message,), {
                'DESCRIPTOR': _ENTITYLISTRESPONSE_ENTITYINFO,
                '__module__': 'engine_pb2'
            }),
        'DESCRIPTOR':
            _ENTITYLISTRESPONSE,
        '__module__':
            'engine_pb2'
    })
_sym_db.RegisterMessage(EntityListResponse)
_sym_db.RegisterMessage(EntityListResponse.EntityInfo)
NodeDesign = _reflection.GeneratedProtocolMessageType('NodeDesign', (_message.Message,), {
    'DESCRIPTOR': _NODEDESIGN,
    '__module__': 'engine_pb2'
})
_sym_db.RegisterMessage(NodeDesign)
InitInfo = _reflection.GeneratedProtocolMessageType(
    'InitInfo', (_message.Message,), {
        'OneSample':
            _reflection.GeneratedProtocolMessageType('OneSample', (_message.Message,), {
                'DESCRIPTOR': _INITINFO_ONESAMPLE,
                '__module__': 'engine_pb2'
            }),
        'MultiSample':
            _reflection.GeneratedProtocolMessageType('MultiSample', (_message.Message,), {
                'DESCRIPTOR': _INITINFO_MULTISAMPLE,
                '__module__': 'engine_pb2'
            }),
        'DESCRIPTOR':
            _INITINFO,
        '__module__':
            'engine_pb2'
    })
_sym_db.RegisterMessage(InitInfo)
_sym_db.RegisterMessage(InitInfo.OneSample)
_sym_db.RegisterMessage(InitInfo.MultiSample)
HttpInfo = _reflection.GeneratedProtocolMessageType('HttpInfo', (_message.Message,), {
    'DESCRIPTOR': _HTTPINFO,
    '__module__': 'engine_pb2'
})
_sym_db.RegisterMessage(HttpInfo)
ControlCmd = _reflection.GeneratedProtocolMessageType('ControlCmd', (_message.Message,), {
    'DESCRIPTOR': _CONTROLCMD,
    '__module__': 'engine_pb2'
})
_sym_db.RegisterMessage(ControlCmd)
EngineNodeState = _reflection.GeneratedProtocolMessageType('EngineNodeState', (_message.Message,), {
    'DESCRIPTOR': _ENGINENODESTATE,
    '__module__': 'engine_pb2'
})
_sym_db.RegisterMessage(EngineNodeState)
SysInfoResponse = _reflection.GeneratedProtocolMessageType('SysInfoResponse', (_message.Message,), {
    'DESCRIPTOR': _SYSINFORESPONSE,
    '__module__': 'engine_pb2'
})
_sym_db.RegisterMessage(SysInfoResponse)
DataSysInfoResponse = _reflection.GeneratedProtocolMessageType('DataSysInfoResponse', (_message.Message,), {
    'DESCRIPTOR': _DATASYSINFORESPONSE,
    '__module__': 'engine_pb2'
})
_sym_db.RegisterMessage(DataSysInfoResponse)
NodeResponse = _reflection.GeneratedProtocolMessageType('NodeResponse', (_message.Message,), {
    'DESCRIPTOR': _NODERESPONSE,
    '__module__': 'engine_pb2'
})
_sym_db.RegisterMessage(NodeResponse)
InitedResponse = _reflection.GeneratedProtocolMessageType('InitedResponse', (_message.Message,), {
    'DESCRIPTOR': _INITEDRESPONSE,
    '__module__': 'engine_pb2'
})
_sym_db.RegisterMessage(InitedResponse)
_SIMCONTROLLER = DESCRIPTOR.services_by_name['SimController']
if _descriptor._USE_C_DESCRIPTORS == False:
    DESCRIPTOR._options = None
    _LOGLEVEL._serialized_start = 2664
    _LOGLEVEL._serialized_end = 2747
    _COMMONRESPONSE._serialized_start = 96
    _COMMONRESPONSE._serialized_end = 163
    _SCENARIOINFO._serialized_start = 165
    _SCENARIOINFO._serialized_end = 191
    _LOGLEVELREQUEST._serialized_start = 193
    _LOGLEVELREQUEST._serialized_end = 251
    _ERRMSGRESPONSE._serialized_start = 253
    _ERRMSGRESPONSE._serialized_end = 322
    _COMMONREQUEST._serialized_start = 324
    _COMMONREQUEST._serialized_end = 339
    _NODEJOINEXITRESPONSE._serialized_start = 341
    _NODEJOINEXITRESPONSE._serialized_end = 421
    _ENTITYLISTREQUEST._serialized_start = 423
    _ENTITYLISTREQUEST._serialized_end = 459
    _ENTITY._serialized_start = 461
    _ENTITY._serialized_end = 499
    _ENTITYLISTRESPONSE._serialized_start = 502
    _ENTITYLISTRESPONSE._serialized_end = 712
    _ENTITYLISTRESPONSE_ENTITYINFO._serialized_start = 617
    _ENTITYLISTRESPONSE_ENTITYINFO._serialized_end = 712
    _NODEDESIGN._serialized_start = 714
    _NODEDESIGN._serialized_end = 784
    _INITINFO._serialized_start = 787
    _INITINFO._serialized_end = 1113
    _INITINFO_ONESAMPLE._serialized_start = 950
    _INITINFO_ONESAMPLE._serialized_end = 1020
    _INITINFO_MULTISAMPLE._serialized_start = 1022
    _INITINFO_MULTISAMPLE._serialized_end = 1100
    _HTTPINFO._serialized_start = 1115
    _HTTPINFO._serialized_end = 1140
    _CONTROLCMD._serialized_start = 1143
    _CONTROLCMD._serialized_end = 1600
    _CONTROLCMD_RUNCMDTYPE._serialized_start = 1462
    _CONTROLCMD_RUNCMDTYPE._serialized_end = 1547
    _CONTROLCMD_MODE._serialized_start = 1549
    _CONTROLCMD_MODE._serialized_end = 1593
    _ENGINENODESTATE._serialized_start = 1603
    _ENGINENODESTATE._serialized_end = 1875
    _ENGINENODESTATE_STATE._serialized_start = 1777
    _ENGINENODESTATE_STATE._serialized_end = 1875
    _SYSINFORESPONSE._serialized_start = 1878
    _SYSINFORESPONSE._serialized_end = 2218
    _DATASYSINFORESPONSE._serialized_start = 2221
    _DATASYSINFORESPONSE._serialized_end = 2591
    _DATASYSINFORESPONSE_SERVICESTATE._serialized_start = 2486
    _DATASYSINFORESPONSE_SERVICESTATE._serialized_end = 2591
    _NODERESPONSE._serialized_start = 2593
    _NODERESPONSE._serialized_end = 2622
    _INITEDRESPONSE._serialized_start = 2624
    _INITEDRESPONSE._serialized_end = 2662
    _SIMCONTROLLER._serialized_start = 2750
    _SIMCONTROLLER._serialized_end = 3776
