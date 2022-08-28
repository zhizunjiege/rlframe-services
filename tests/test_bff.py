import json
import unittest

import grpc

from protos import agent_pb2
from protos import agent_pb2_grpc
from protos import engine_pb2
from protos import engine_pb2_grpc
from protos import bff_pb2
from protos import bff_pb2_grpc
from protos import types_pb2


class BFFServicerTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.stub = bff_pb2_grpc.BFFStub(grpc.insecure_channel("localhost:50050"))

    @classmethod
    def tearDownClass(cls):
        cls.stub = None

    def test_00_set_proxy_config(self):
        ...
