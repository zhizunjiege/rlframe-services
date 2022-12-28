# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: protos/simenv.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()

from protos import types_pb2 as protos_dot_types__pb2

DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(
    b'\n\x13protos/simenv.proto\x12\x0bgame.simenv\x1a\x12protos/types.proto\"*\n\x0cSimenvConfig\x12\x0c\n\x04type\x18\x01 \x01(\t\x12\x0c\n\x04\x61rgs\x18\x02 \x01(\t\"\xb4\x01\n\x06SimCmd\x12%\n\x03\x63md\x18\x01 \x01(\x0e\x32\x18.game.simenv.SimCmd.Type\x12&\n\x06params\x18\x02 \x01(\x0b\x32\x16.game.types.JsonString\"[\n\x04Type\x12\x08\n\x04INIT\x10\x00\x12\t\n\x05START\x10\x01\x12\t\n\x05PAUSE\x10\x02\x12\x08\n\x04STEP\x10\x03\x12\n\n\x06RESUME\x10\x04\x12\x08\n\x04STOP\x10\x05\x12\x08\n\x04\x44ONE\x10\x06\x12\t\n\x05PARAM\x10\x07\"\xc0\x01\n\x07SimInfo\x12)\n\x05state\x18\x01 \x01(\x0e\x32\x1a.game.simenv.SimInfo.State\x12$\n\x04\x64\x61ta\x18\x02 \x01(\x0b\x32\x16.game.types.JsonString\x12$\n\x04logs\x18\x03 \x01(\x0b\x32\x16.game.types.JsonString\">\n\x05State\x12\x0c\n\x08UNINITED\x10\x00\x12\x0b\n\x07STOPPED\x10\x01\x12\x0b\n\x07RUNNING\x10\x02\x12\r\n\tSUSPENDED\x10\x03\x32\xa5\x03\n\x06Simenv\x12\x45\n\x0cResetService\x12\x19.game.types.CommonRequest\x1a\x1a.game.types.CommonResponse\x12\x43\n\x0cQueryService\x12\x19.game.types.CommonRequest\x1a\x18.game.types.ServiceState\x12G\n\x0fGetSimenvConfig\x12\x19.game.types.CommonRequest\x1a\x19.game.simenv.SimenvConfig\x12H\n\x0fSetSimenvConfig\x12\x19.game.simenv.SimenvConfig\x1a\x1a.game.types.CommonResponse\x12=\n\nSimControl\x12\x13.game.simenv.SimCmd\x1a\x1a.game.types.CommonResponse\x12=\n\nSimMonitor\x12\x19.game.types.CommonRequest\x1a\x14.game.simenv.SimInfob\x06proto3'
)

_SIMENVCONFIG = DESCRIPTOR.message_types_by_name['SimenvConfig']
_SIMCMD = DESCRIPTOR.message_types_by_name['SimCmd']
_SIMINFO = DESCRIPTOR.message_types_by_name['SimInfo']
_SIMCMD_TYPE = _SIMCMD.enum_types_by_name['Type']
_SIMINFO_STATE = _SIMINFO.enum_types_by_name['State']
SimenvConfig = _reflection.GeneratedProtocolMessageType(
    'SimenvConfig',
    (_message.Message,),
    {
        'DESCRIPTOR': _SIMENVCONFIG,
        '__module__': 'protos.simenv_pb2'
        # @@protoc_insertion_point(class_scope:game.simenv.SimenvConfig)
    })
_sym_db.RegisterMessage(SimenvConfig)

SimCmd = _reflection.GeneratedProtocolMessageType(
    'SimCmd',
    (_message.Message,),
    {
        'DESCRIPTOR': _SIMCMD,
        '__module__': 'protos.simenv_pb2'
        # @@protoc_insertion_point(class_scope:game.simenv.SimCmd)
    })
_sym_db.RegisterMessage(SimCmd)

SimInfo = _reflection.GeneratedProtocolMessageType(
    'SimInfo',
    (_message.Message,),
    {
        'DESCRIPTOR': _SIMINFO,
        '__module__': 'protos.simenv_pb2'
        # @@protoc_insertion_point(class_scope:game.simenv.SimInfo)
    })
_sym_db.RegisterMessage(SimInfo)

_SIMENV = DESCRIPTOR.services_by_name['Simenv']
if _descriptor._USE_C_DESCRIPTORS == False:

    DESCRIPTOR._options = None
    _SIMENVCONFIG._serialized_start = 56
    _SIMENVCONFIG._serialized_end = 98
    _SIMCMD._serialized_start = 101
    _SIMCMD._serialized_end = 281
    _SIMCMD_TYPE._serialized_start = 190
    _SIMCMD_TYPE._serialized_end = 281
    _SIMINFO._serialized_start = 284
    _SIMINFO._serialized_end = 476
    _SIMINFO_STATE._serialized_start = 414
    _SIMINFO_STATE._serialized_end = 476
    _SIMENV._serialized_start = 479
    _SIMENV._serialized_end = 900
# @@protoc_insertion_point(module_scope)
