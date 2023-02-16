import json
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
        with open('examples/simenv/args.json', 'r') as f1, \
                open('examples/simenv/configs.json', 'r') as f2, \
                open('examples/simenv/sim_term_func.cc', 'r') as f3:
            cls.engine = CQSim(**json.load(f1)['args'])
            cls.sim_params = json.load(f2)
            cls.sim_params['proxy']['sim_term_func'] = f3.read()
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
        self.engine.control('init', self.sim_params)
        self.engine.control('start')
        time.sleep(3)
        self.engine.control('pause')
        self.engine.control('param', {'speed_ratio': 100})
        self.engine.control('step')
        self.engine.control('resume')
        time.sleep(3)
        self.engine.control('episode')
        time.sleep(10)
        self.engine.control('stop')

    def test_01_multisample(self):
        self.sim_params['task']['exp_design_id'] = 28
        self.sim_params['task']['exp_sample_num'] = 3
        self.sim_params['task']['repeat_times'] = 2

        self.engine.control('init', self.sim_params)
        self.engine.control('start')
        time.sleep(3)
        self.engine.control('pause')
        self.engine.control('param', {'speed_ratio': 100})
        self.engine.control('step')
        self.engine.control('resume')
        time.sleep(3)
        self.engine.control('episode')
        time.sleep(10)
        self.engine.control('stop')
