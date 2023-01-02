import threading
from typing import Any, Dict, List, Literal, Tuple

import grpc
from google.protobuf import duration_pb2
from google.protobuf import timestamp_pb2

from .base import SimEngineBase
from .cqsim_pb import engine_pb2
from .cqsim_pb import engine_pb2_grpc


class CQSim(SimEngineBase):

    def __init__(self, id: str, engine_addr: str):
        """Init CQSim engine.

        Args:
            id: Id of simulation engine.
            engine_addr: Address of CQSim engine.
        """
        super().__init__(id=id)

        self.engine_addr = engine_addr
        self.channel = grpc.insecure_channel(engine_addr)
        self.engine = engine_pb2_grpc.SimControllerStub(channel=self.channel)

        self.data_thread = None
        self.data_responses = None
        self.data_lock = threading.Lock()
        self.data_cache = {}

        self.logs_thread = None
        self.logs_responses = None
        self.logs_lock = threading.Lock()
        self.logs_cache = []

    def control(
        self,
        cmd: Literal['init', 'start', 'pause', 'step', 'resume', 'stop', 'done', 'param'],
        params: Dict[str, Any],
    ) -> bool:
        """Control CQSim engine.

        Args:
            cmd: Control command. `done` means ending current episode, `param` means setting simulation parameters.
            params: Control parameters.

        Returns:
            True if success.
        """
        if cmd == 'init':
            sample = engine_pb2.InitInfo.MultiSample(exp_design_id=params['exp_design_id'])
            self.engine.Init(engine_pb2.InitInfo(multi_sample_config=sample))
            self.control('param', params)
            self.data_thread = threading.Thread(name='data_thread', target=self.data_callback)
            self.data_thread.daemon = True
            self.data_thread.start()
            self.logs_thread = threading.Thread(name='logs_thread', target=self.logs_callback)
            self.logs_thread.daemon = True
            self.logs_thread.start()
            self._state = 'stopped'
            return True
        elif cmd == 'start':
            self.engine.Control(engine_pb2.ControlCmd(run_cmd=engine_pb2.ControlCmd.RunCmdType.START))
            self._state = 'running'
            return True
        elif cmd == 'pause':
            self.engine.Control(engine_pb2.ControlCmd(run_cmd=engine_pb2.ControlCmd.RunCmdType.SUSPEND))
            self._state = 'suspended'
            return True
        elif cmd == 'step':
            return False
        elif cmd == 'resume':
            self.engine.Control(engine_pb2.ControlCmd(run_cmd=engine_pb2.ControlCmd.RunCmdType.CONTINUE))
            self._state = 'running'
            return True
        elif cmd == 'stop':
            self.engine.Control(engine_pb2.ControlCmd(run_cmd=engine_pb2.ControlCmd.RunCmdType.STOP))
            self._state = 'stopped'
            return True
        elif cmd == 'done':
            self.engine.Control(engine_pb2.ControlCmd(run_cmd=engine_pb2.ControlCmd.RunCmdType.STOP_CURRENT_SAMPLE))
            self._state = 'running'
            return True
        elif cmd == 'param':
            if 'sim_start_time' in params:
                sim_start_time = timestamp_pb2.Timestamp()
                sim_start_time.FromSeconds(params['sim_start_time'])
                self.engine.Control(engine_pb2.ControlCmd(sim_start_time=sim_start_time))
            if 'sim_duration' in params:
                sim_duration = duration_pb2.Duration()
                sim_duration.FromSeconds(params['sim_duration'])
                self.engine.Control(engine_pb2.ControlCmd(sim_duration=sim_duration))
            if 'time_step' in params:
                self.engine.Control(engine_pb2.ControlCmd(time_step=params['time_step']))
            if 'speed_ratio' in params:
                self.engine.Control(engine_pb2.ControlCmd(speed_ratio=params['speed_ratio']))
            return True
        else:
            return False

    def monitor(self) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Monitor CQSim engine.

        Returns:
            Data of simulation process.
            Logs of CQSim engine.
        """
        with self.data_lock:
            data = self.data_cache.copy()
        with self.logs_lock:
            logs = self.logs_cache.copy()
            self.logs_cache.clear()
        return data, logs

    def data_callback(self):
        self.data_responses = self.engine.GetSysInfo(engine_pb2.CommonRequest())
        try:
            for response in self.data_responses:
                with self.data_lock:
                    self.data_cache['sim_current_time'] = response.sim_current_time.ToSeconds()
                    self.data_cache['sim_duration'] = response.sim_duration.ToSeconds()
                    self.data_cache['real_duration'] = response.real_duration.ToSeconds()
                    self.data_cache['sim_time_step'] = response.sim_time_step
                    self.data_cache['speed_ratio'] = response.speed_ratio
                    self.data_cache['real_speed_ratio'] = response.real_speed_ratio
                    self.data_cache['current_sample_id'] = response.current_sample_id
        except grpc.RpcError:
            ...

    def logs_callback(self):
        self.logs_responses = self.engine.GetErrorMsg(engine_pb2.CommonRequest())
        try:
            for response in self.logs_responses:
                with self.logs_lock:
                    self.logs_cache.append(response.msg)
        except grpc.RpcError:
            ...

    def close(self):
        """Close CQSim engine.

        Returns:
            True if success.
        """
        if self.data_thread is not None:
            self.data_responses.cancel()
            self.data_thread.join(1)
        if self.logs_thread is not None:
            self.logs_responses.cancel()
            self.logs_thread.join(1)
        self.channel.close()
        return True
