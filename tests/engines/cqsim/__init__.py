import json
import threading
import time
import unittest

from engines.base import CommandType
from engines.cqsim import CQSIM


class RepeatTimer(threading.Timer):

    def run(self):
        while not self.finished.is_set():
            self.function(*self.args, **self.kwargs)
            self.finished.wait(self.interval)


class CQSIMTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        ...

    @classmethod
    def tearDownClass(cls):
        ...

    def test_00_single_sample(self):
        with open('tests/engines/cqsim/args-single-sample.json', 'r') as f1, \
             open('tests/engines/cqsim/sim_term_func.cpp', 'r') as f2:
            args = json.load(f1)
            args['sim_term_func'] = f2.read()
        engine = CQSIM(**args)

        timer = RepeatTimer(1, lambda: print(engine.monitor()))
        timer.start()

        engine.control(CommandType.INIT)
        engine.control(CommandType.START)
        time.sleep(3)
        engine.control(CommandType.PAUSE)
        engine.control(CommandType.PARAM, {'speed_ratio': 100})
        engine.control(CommandType.STEP)
        engine.control(CommandType.RESUME)
        time.sleep(3)
        engine.control(CommandType.EPISODE)
        time.sleep(30)
        engine.control(CommandType.STOP)

        timer.cancel()

    def test_01_multi_sample(self):
        with open('tests/engines/cqsim/args-multi-sample.json', 'r') as f1, \
             open('tests/engines/cqsim/sim_term_func.cpp', 'r') as f2:
            args = json.load(f1)
            args['sim_term_func'] = f2.read()
        engine = CQSIM(**args)

        timer = RepeatTimer(1, lambda: print(engine.monitor()))
        timer.start()

        engine.control(CommandType.INIT)
        engine.control(CommandType.START)
        time.sleep(3)
        engine.control(CommandType.PAUSE)
        engine.control(CommandType.PARAM, {'speed_ratio': 100})
        engine.control(CommandType.STEP)
        engine.control(CommandType.RESUME)
        time.sleep(3)
        engine.control(CommandType.EPISODE)
        time.sleep(30)
        engine.control(CommandType.STOP)

        timer.cancel()
