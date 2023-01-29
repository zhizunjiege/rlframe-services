# RLFrame

## 生成

运行以下命令生成 Python 接口文件：

```bash
python -m grpc_tools.protoc --proto_path=. --python_out=. --grpc_python_out=. ./protos/*.proto
```

运行以下命令生成 C++ 接口文件：

```bash
protoc -I . --grpc_out=. --plugin=protoc-gen-grpc="grpc_cpp_plugin.exe" ./protos/*.proto
protoc -I . --cpp_out=. ./protos/*.proto
```

或者直接在 VSCode 里运行构建任务.

## 测试

运行以下命令执行单元测试：

```bash
python -m unittest discover -s tests
```

或者在 VSCode 里运行测试任务.
