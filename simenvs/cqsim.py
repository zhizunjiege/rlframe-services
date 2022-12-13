import threading
from typing import Any, Dict, List, Literal, Tuple

import grpc

from .base import SimEnvBase
from .cqsim_pb import engine_pb2
from .cqsim_pb import engine_pb2_grpc


class CQSim(SimEnvBase):

    def __init__(self, id: str, *, engine_url: str):
        """Init CQSim env.

        Args:
            id: Id of simulation enviroment.
            engine_url: Url of simulation engine.
        """
        super().__init__(id=id)

        self.engine_url = engine_url

        self.channel = grpc.insecure_channel(engine_url)
        self.engine = engine_pb2_grpc.SimControllerStub(channel=self.channel)

        self.data_cache = None
        self.data_lock = threading.Lock()
        self.logs_cache = []
        self.logs_lock = threading.Lock()

    def _data_thread(self):
        for response in self.engine.GetSysInfo(engine_pb2.CommonRequest()):
            with self.data_lock:
                self.data_cache = {
                    'sim_current_time': response.sim_current_time.ToMilliseconds(),
                    'sim_duration': response.sim_duration.ToMilliseconds(),
                    'real_duration': response.real_duration.ToMilliseconds(),
                    'sim_time_step': response.sim_time_step,
                    'speed_ratio': response.speed_ratio,
                    'real_speed_ratio': response.real_speed_ratio,
                    'current_sample_id': response.current_sample_id,
                }
        with self.data_lock:
            self.data_cache = None

    def _logs_thread(self):
        for response in self.engine.GetErrorMsg(engine_pb2.CommonRequest()):
            with self.logs_lock:
                self.logs_cache.append(response.msg)
        with self.logs_lock:
            self.logs_cache.clear()

    def control(
        self,
        cmd: Literal['init', 'start', 'pause', 'step', 'resume', 'stop', 'done', 'param'],
        params: Dict[str, Any],
    ) -> bool:
        """Control CQSim env.

        Args:
            cmd: Control command. `done` means ending current episode, `param` means setting simulation parameters.
            params: Control parameters.

        Returns:
            True if supported, False otherwise.
        """
        if cmd == 'init':
            sample = engine_pb2.InitInfo.MultiSample(exp_design_id=params['exp_design_id'])
            self.engine.Init(engine_pb2.InitInfo(multi_sample_config=sample))
            return True
        elif cmd == 'start':
            self.control('param', params)
            self.engine.Control(engine_pb2.ControlCmd(run_cmd=engine_pb2.ControlCmd.RunCmdType.START))
            self.data_thread = threading.Thread(name='data_thread', target=self._data_thread)
            self.logs_thread = threading.Thread(name='logs_thread', target=self._logs_thread)
            self.data_thread.start()
            self.logs_thread.start()
            return True
        elif cmd == 'pause':
            self.engine.Control(engine_pb2.ControlCmd(run_cmd=engine_pb2.ControlCmd.RunCmdType.SUSPEND))
            return True
        elif cmd == 'step':
            return False
        elif cmd == 'resume':
            self.engine.Control(engine_pb2.ControlCmd(run_cmd=engine_pb2.ControlCmd.RunCmdType.CONTINUE))
            return True
        elif cmd == 'stop':
            self.engine.Control(engine_pb2.ControlCmd(run_cmd=engine_pb2.ControlCmd.RunCmdType.STOP))
            self.data_thread.join(1)
            self.logs_thread.join(1)
            return True
        elif cmd == 'done':
            self.engine.Control(engine_pb2.ControlCmd(run_cmd=engine_pb2.ControlCmd.RunCmdType.STOP_CURRENT_SAMPLE))
            return True
        elif cmd == 'param':
            if 'sim_start_time' in params:
                self.engine.Control(engine_pb2.ControlCmd(sim_start_time=params['sim_start_time']))
            if 'sim_duration' in params:
                self.engine.Control(engine_pb2.ControlCmd(sim_duration=params['sim_duration']))
            if 'time_step' in params:
                self.engine.Control(engine_pb2.ControlCmd(time_step=params['time_step']))
            if 'speed_ratio' in params:
                self.engine.Control(engine_pb2.ControlCmd(speed_ratio=params['speed_ratio']))
            return True
        else:
            return False

    def monitor(self) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Monitor CQSim env.

        Returns:
            Data of simulation.
            Logs of simulation enviroment.
        """
        with self.data_lock:
            data = self.data_cache
        with self.logs_lock:
            logs = self.logs_cache.copy()
            self.logs_cache.clear()
        return data, logs

    def close(self) -> bool:
        """Close CQSim env.

        Returns:
            True if success, False otherwise.
        """
        self.channel.close()
        return True
