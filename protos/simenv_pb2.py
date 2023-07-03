"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database

_sym_db = _symbol_database.Default()
from . import types_pb2 as types__pb2

DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(
    b'\n\x0csimenv.proto\x12\x0bgame.simenv\x1a\x0btypes.proto"\xca\x01\n\x0cSimenvConfig\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x0c\n\x04args\x18\x02 \x01(\t\x125\n\x06routes\x18\x03 \x03(\x0b2%.game.simenv.SimenvConfig.RoutesEntry\x1a\x17\n\x05Route\x12\x0e\n\x06models\x18\x01 \x03(\t\x1aN\n\x0bRoutesEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12.\n\x05value\x18\x02 \x01(\x0b2\x1f.game.simenv.SimenvConfig.Route:\x028\x01"&\n\x06SimCmd\x12\x0c\n\x04type\x18\x01 \x01(\t\x12\x0e\n\x06params\x18\x02 \x01(\t"4\n\x07SimInfo\x12\r\n\x05state\x18\x01 \x01(\t\x12\x0c\n\x04data\x18\x02 \x01(\t\x12\x0c\n\x04logs\x18\x03 \x01(\t2\xd9\x03\n\x06Simenv\x12E\n\x0cResetService\x12\x19.game.types.CommonRequest\x1a\x1a.game.types.CommonResponse\x12C\n\x0cQueryService\x12\x19.game.types.CommonRequest\x1a\x18.game.types.ServiceState\x12G\n\x0fGetSimenvConfig\x12\x19.game.types.CommonRequest\x1a\x19.game.simenv.SimenvConfig\x12H\n\x0fSetSimenvConfig\x12\x19.game.simenv.SimenvConfig\x1a\x1a.game.types.CommonResponse\x12=\n\nSimControl\x12\x13.game.simenv.SimCmd\x1a\x1a.game.types.CommonResponse\x12=\n\nSimMonitor\x12\x19.game.types.CommonRequest\x1a\x14.game.simenv.SimInfo\x122\n\x04Call\x12\x14.game.types.CallData\x1a\x14.game.types.CallDatab\x06proto3'
)
_SIMENVCONFIG = DESCRIPTOR.message_types_by_name['SimenvConfig']
_SIMENVCONFIG_ROUTE = _SIMENVCONFIG.nested_types_by_name['Route']
_SIMENVCONFIG_ROUTESENTRY = _SIMENVCONFIG.nested_types_by_name['RoutesEntry']
_SIMCMD = DESCRIPTOR.message_types_by_name['SimCmd']
_SIMINFO = DESCRIPTOR.message_types_by_name['SimInfo']
SimenvConfig = _reflection.GeneratedProtocolMessageType(
    'SimenvConfig', (_message.Message,), {
        'Route':
            _reflection.GeneratedProtocolMessageType('Route', (_message.Message,), {
                'DESCRIPTOR': _SIMENVCONFIG_ROUTE,
                '__module__': 'simenv_pb2'
            }),
        'RoutesEntry':
            _reflection.GeneratedProtocolMessageType('RoutesEntry', (_message.Message,), {
                'DESCRIPTOR': _SIMENVCONFIG_ROUTESENTRY,
                '__module__': 'simenv_pb2'
            }),
        'DESCRIPTOR':
            _SIMENVCONFIG,
        '__module__':
            'simenv_pb2'
    })
_sym_db.RegisterMessage(SimenvConfig)
_sym_db.RegisterMessage(SimenvConfig.Route)
_sym_db.RegisterMessage(SimenvConfig.RoutesEntry)
SimCmd = _reflection.GeneratedProtocolMessageType('SimCmd', (_message.Message,), {
    'DESCRIPTOR': _SIMCMD,
    '__module__': 'simenv_pb2'
})
_sym_db.RegisterMessage(SimCmd)
SimInfo = _reflection.GeneratedProtocolMessageType('SimInfo', (_message.Message,), {
    'DESCRIPTOR': _SIMINFO,
    '__module__': 'simenv_pb2'
})
_sym_db.RegisterMessage(SimInfo)
_SIMENV = DESCRIPTOR.services_by_name['Simenv']
if _descriptor._USE_C_DESCRIPTORS == False:
    DESCRIPTOR._options = None
    _SIMENVCONFIG_ROUTESENTRY._options = None
    _SIMENVCONFIG_ROUTESENTRY._serialized_options = b'8\x01'
    _SIMENVCONFIG._serialized_start = 43
    _SIMENVCONFIG._serialized_end = 245
    _SIMENVCONFIG_ROUTE._serialized_start = 142
    _SIMENVCONFIG_ROUTE._serialized_end = 165
    _SIMENVCONFIG_ROUTESENTRY._serialized_start = 167
    _SIMENVCONFIG_ROUTESENTRY._serialized_end = 245
    _SIMCMD._serialized_start = 247
    _SIMCMD._serialized_end = 285
    _SIMINFO._serialized_start = 287
    _SIMINFO._serialized_end = 339
    _SIMENV._serialized_start = 342
    _SIMENV._serialized_end = 815
