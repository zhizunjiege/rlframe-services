import threading
import time
import unittest

from engines.cqsim import CQSim


class RepeatTimer(threading.Timer):

    def run(self):
        while not self.finished.is_set():
            self.function(*self.args, **self.kwargs)
            self.finished.wait(self.interval)


class CQSimTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.engine = CQSim(engine_addr='localhost:50041')
        cls.timer = RepeatTimer(1, cls.print_monitor)
        cls.timer.start()

    @classmethod
    def print_monitor(cls):
        print(cls.engine.monitor())

    @classmethod
    def tearDownClass(cls):
        cls.timer.cancel()
        cls.engine.close()
        cls.engine = None

    def test_00_onesample(self):
        self.engine.control(
            'init',
            {
                'task_id': 18,
                'repeat_times': 4,
                'sim_start_time': int(time.time()),
                'sim_duration': 30,
                'time_step': 1000,
                'speed_ratio': 10,
            },
        )
        self.engine.control('start', {})
        time.sleep(3)
        self.engine.control('pause', {})
        time.sleep(1)
        self.engine.control('param', {'speed_ratio': 100})
        self.engine.control('step', {})
        self.engine.control('resume', {})
        time.sleep(3)
        self.engine.control('episode', {})
        time.sleep(10)

    def test_01_multisample(self):
        self.engine.control(
            'init',
            {
                'exp_design_id': 28,
                'exp_sample_num': 4,
                'repeat_times': 2,
                'sim_start_time': int(time.time()),
                'sim_duration': 30,
                'time_step': 1000,
                'speed_ratio': 10,
            },
        )
        self.engine.control('start', {})
        time.sleep(3)
        self.engine.control('pause', {})
        time.sleep(1)
        self.engine.control('param', {'speed_ratio': 100})
        self.engine.control('step', {})
        self.engine.control('resume', {})
        time.sleep(3)
        self.engine.control('episode', {})
        time.sleep(10)
