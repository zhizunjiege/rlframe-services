import time
import unittest

from engines.cqsim import CQSim


class CQSimTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.engine = CQSim(id='cqsim', engine_addr='localhost:50041')

    @classmethod
    def tearDownClass(cls):
        cls.engine.close()
        cls.engine = None

    def test_00_control(self):
        self.engine.control(
            'init',
            {
                'exp_design_id': 28,
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
        print(self.engine.monitor())
        self.engine.control('done', {})
        time.sleep(10)
        print(self.engine.monitor())
        self.engine.control('stop', {})

    def test_01_monitor(self):
        print(self.engine.monitor())
