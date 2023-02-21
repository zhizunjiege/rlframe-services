@echo off

python -m grpc_tools.protoc --proto_path=. --python_out=. --grpc_python_out=. ./protos/*.proto
yapf protos -i -p -r
