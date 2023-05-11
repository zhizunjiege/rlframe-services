import json
import threading
import time
import unittest

from engines.cqsim import CQSIM


class RepeatTimer(threading.Timer):

    def run(self):
        while not self.finished.is_set():
            self.function(*self.args, **self.kwargs)
            self.finished.wait(self.interval)


class CQSIMTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with open('tests/engines/test_cqsim_src/args.json', 'r') as f1, \
             open('tests/engines/test_cqsim_src/sim_term_func.cpp', 'r') as f2:
            cls.args = json.load(f1)
            cls.args['proxy']['sim_term_func'] = f2.read()
        cls.engine = CQSIM(**cls.args)
        cls.timer = RepeatTimer(1, cls.print_monitor)
        cls.timer.start()

    @classmethod
    def print_monitor(cls):
        print(cls.engine.monitor())

    @classmethod
    def tearDownClass(cls):
        cls.timer.cancel()
        cls.engine = None

    def test_00_onesample(self):
        self.engine.control('init')
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
        self.args['task']['exp_design_id'] = 28
        self.args['task']['exp_sample_num'] = 3
        self.args['task']['repeat_times'] = 2

        self.engine.control('init')
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
