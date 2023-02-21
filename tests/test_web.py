import base64
import json
import unittest

import requests


class WebServerTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.addr = 'http://localhost:5000/api/db'

    @classmethod
    def tearDownClass(cls):
        ...

    def test_00_static(self):
        addr = f'{self.addr}/index.html'
        res = requests.get(addr)
        self.assertTrue(res.ok)

    def test_01_select(self):
        addr = f'{self.addr}/simenv'
        res = requests.get(
            addr,
            params={
                'columns': ['time'],
                'id': 1,
            },
        )
        self.assertTrue(res.ok)
        self.assertEqual(res.json()['code'], 1)
        print(res.json())

        addr = f'{self.addr}/agent'
        res = requests.get(
            addr,
            params={
                'columns': ['id', 'create_time', 'update_time'],
                'limit': 10,
                'offset': 0,
            },
        )
        self.assertTrue(res.ok)
        self.assertEqual(res.json()['code'], 0)
        print(res.json())

        addr = f'{self.addr}/task'
        res = requests.get(
            addr,
            params={
                'columns': [],
            },
        )
        self.assertTrue(res.ok)
        self.assertEqual(res.json()['code'], 0)
        print(res.json())

    def test_02_insert(self):
        addr = f'{self.addr}/simenv'
        res = requests.post(
            addr,
            json={
                'name': 'test',
                'description': 'test',
                'time': 0,
            },
        )
        self.assertTrue(res.ok)
        self.assertEqual(res.json()['code'], 1)
        print(res.json())

        addr = f'{self.addr}/agent'
        with open('examples/agent/states_inputs_func.py', 'r') as f1, \
             open('examples/agent/outputs_actions_func.py', 'r') as f2, \
             open('examples/agent/reward_func.py', 'r') as f3, \
             open('examples/agent/hypers.json', 'r') as f4, \
             open('examples/agent/structs.json', 'r') as f5, \
             open('examples/agent/builder.py', 'r') as f6:
            states_inputs_func = f1.read()
            outputs_actions_func = f2.read()
            reward_func = f3.read()
            hypers = json.load(f4)
            structs = f5.read()
            builder = f6.read()
        res = requests.post(
            addr,
            json={
                'name': 'test',
                'description': 'test',
                'type': hypers['type'],
                'hypers': json.dumps(hypers['hypers']),
                'structs': structs,
                'builder': builder,
                'states_inputs_func': states_inputs_func,
                'outputs_actions_func': outputs_actions_func,
                'reward_func': reward_func,
                'weights': base64.b64encode(b'Hello World!').decode('utf-8'),
            },
        )
        self.assertTrue(res.ok)
        self.assertEqual(res.json()['code'], 0)
        print(res.json())

        addr = f'{self.addr}/task'
        res = requests.post(
            addr,
            json={
                'name': 'test',
                'description': 'test',
            },
        )
        self.assertTrue(res.ok)
        self.assertEqual(res.json()['code'], 2)
        print(res.json())

    def test_03_update(self):
        addr = f'{self.addr}/simenv/1'
        res = requests.put(
            addr,
            json={
                'time': 0,
            },
        )
        self.assertTrue(res.ok)
        self.assertEqual(res.json()['code'], 1)
        print(res.json())

        addr = f'{self.addr}/agent/1'
        res = requests.put(
            addr,
            json={
                'weights': base64.b64encode(b'Hello World!').decode('utf-8'),
            },
        )
        self.assertTrue(res.ok)
        self.assertEqual(res.json()['code'], 0)
        print(res.json())

        addr = f'{self.addr}/task/1'
        res = requests.put(
            addr,
            json={
                'services': 0,
            },
        )
        self.assertTrue(res.ok)
        self.assertEqual(res.json()['code'], 2)
        print(res.json())

    def test_04_delete(self):
        addr = f'{self.addr}/agent/1'
        res = requests.delete(addr)
        self.assertTrue(res.ok)
        self.assertEqual(res.json()['code'], 0)
        print(res.json())
