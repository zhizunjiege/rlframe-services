import base64
import json
import unittest

import requests


class WebServerTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.addr = 'http://localhost:5000'

    @classmethod
    def tearDownClass(cls):
        ...

    def test_00_meta(self):
        addr = f'{self.addr}/api/db'
        res = requests.get(addr)
        self.assertTrue(res.ok)

    def test_01_select(self):
        addr = f'{self.addr}/api/db/simenv'
        res = requests.get(addr, params={
            'columns': ['time'],
        })
        self.assertEqual(res.status_code, 404)

        addr = f'{self.addr}/api/db/agent'
        res = requests.get(addr, params={
            'columns': ['id', 'hypers', 'weights'],
            'limit': 10,
            'offset': 0,
        })
        self.assertTrue(res.ok)

        addr = f'{self.addr}/api/db/task'
        res = requests.get(addr, params={
            'columns': [],
            'id': 0,
        })
        self.assertTrue(res.ok)

    def test_02_insert(self):
        addr = f'{self.addr}/api/db/simenv'
        res = requests.post(addr, json={
            'name': 'test',
            'description': 'test',
            'type': 0,
        })
        self.assertEqual(res.status_code, 400)

        addr = f'{self.addr}/api/db/agent'
        with open('examples/agent/hypers.json', 'r') as f1, \
             open('examples/agent/states_inputs_func.py', 'r') as f2, \
             open('examples/agent/outputs_actions_func.py', 'r') as f3, \
             open('examples/agent/reward_func.py', 'r') as f4:
            hypers = json.load(f1)
            states_inputs_func = f2.read()
            outputs_actions_func = f3.read()
            reward_func = f4.read()
        res = requests.post(
            addr,
            json={
                'id': -1,
                'name': 'test',
                'description': 'test',
                'training': 1,
                'type': hypers['type'],
                'hypers': json.dumps(hypers['hypers']),
                'sifunc': states_inputs_func,
                'oafunc': outputs_actions_func,
                'rewfunc': reward_func,
                'weights': base64.b64encode(b'Helloworld!').decode('utf-8'),
            },
        )
        self.assertTrue(res.ok)

        addr = f'{self.addr}/api/db/task'
        res = requests.post(addr, json={
            'name': 'test',
            'description': 'test',
        })
        self.assertEqual(res.status_code, 400)

    def test_03_update(self):
        addr = f'{self.addr}/api/db/simenvs'
        res = requests.put(addr, json={
            'id': 1,
            'params': '{}',
        })
        self.assertEqual(res.status_code, 404)

        addr = f'{self.addr}/api/db/agent'
        res = requests.put(addr, json={
            'id': 1,
            'buffer': base64.b64encode(b'Hello World!').decode('utf-8'),
        })
        self.assertTrue(res.ok)

        addr = f'{self.addr}/api/db/task'
        res = requests.put(addr, json={
            'services': '',
        })
        self.assertEqual(res.status_code, 400)

    def test_04_delete(self):
        addr = f'{self.addr}/api/db/simenv'
        res = requests.delete(addr)
        self.assertEqual(res.status_code, 400)

        addr = f'{self.addr}/api/db/agent'
        res = requests.delete(addr, json={'ids': [1]})
        self.assertTrue(res.ok)

        addr = f'{self.addr}/api/db/task'
        res = requests.delete(addr, json={'ids': 1})
        self.assertEqual(res.status_code, 400)
