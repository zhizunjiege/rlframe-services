import unittest

import requests


class WebServerTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.addr = 'http://localhost:5000'

    @classmethod
    def tearDownClass(cls):
        ...

    def test_00_index(self):
        res = requests.get(self.addr)
        self.assertTrue(res.ok)

    def test_01_select(self):
        addr = f'{self.addr}/api/db/task'
        res = requests.get(addr, params={
            'columns': ['description'],
            'id': 1,
        })
        self.assertEqual(res.status_code, 500)

        addr = f'{self.addr}/api/db/agent'
        res = requests.get(addr, params={})
        self.assertTrue(res.ok)

        addr = f'{self.addr}/api/db/simenv'
        res = requests.get(addr, params={
            'columns': ['id', 'args'],
            'id': 1,
        })
        self.assertTrue(res.ok)

        addr = f'{self.addr}/api/db/service'
        res = requests.get(addr, params={
            'columns': ['id', 'desc'],
            'conjunc': 'OR',
            'agent_id': 1,
            'simenv_id': 1,
        })
        self.assertTrue(res.ok)

    def test_02_insert(self):
        addr = f'{self.addr}/api/db/task'
        res = requests.post(addr, json={
            'name': 'test',
            'description': 'test',
        })
        self.assertEqual(res.status_code, 500)
        res = requests.post(addr, json={
            'name': 'test',
            'desc': 'test',
        })
        self.assertTrue(res.ok)

        addr = f'{self.addr}/api/db/agent'
        res = requests.post(
            addr,
            json={
                'desc': 'test',
                'training': 1,
                'name': 'DQN',
                'hypers': '{}',
                'sifunc': '',
                'oafunc': '',
                'rewfunc': '',
                'hooks': '[]',
            },
        )
        self.assertTrue(res.ok)

        addr = f'{self.addr}/api/db/simenv'
        res = requests.post(addr, json={
            'desc': 'test',
            'name': 'CQSIM',
        })
        self.assertEqual(res.status_code, 500)
        res = requests.post(addr, json={
            'desc': 'test',
            'name': 'CQSIM',
            'args': '{}',
        })
        self.assertTrue(res.ok)

        addr = f'{self.addr}/api/db/service'
        res = requests.post(addr, json={
            'desc': 'test',
            'task_id': 0,
            'server_id': 'test',
        })
        self.assertEqual(res.status_code, 500)
        res = requests.post(addr, json={
            'desc': 'test',
            'task_id': 1,
            'server_id': 'test',
            'agent_id': 1,
        })
        self.assertTrue(res.ok)
        res = requests.post(addr, json={
            'desc': 'test',
            'task_id': 1,
            'server_id': 'test',
            'simenv_id': 1,
        })
        self.assertTrue(res.ok)

    def test_03_update(self):
        addr = f'{self.addr}/api/db/task'
        res = requests.put(addr, json={
            'name': '',
        })
        self.assertEqual(res.status_code, 400)

        addr = f'{self.addr}/api/db/agent'
        res = requests.put(addr, json={
            'id': 1,
            'training': False,
        })
        self.assertTrue(res.ok)

        addr = f'{self.addr}/api/db/simenv'
        res = requests.put(addr, json={
            'id': 1,
            'params': '{}',
        })
        self.assertEqual(res.status_code, 500)

        addr = f'{self.addr}/api/db/service'
        res = requests.put(addr, json={
            'id': 1,
            'task_id': 0,
        })
        self.assertEqual(res.status_code, 500)
        res = requests.put(addr, json={
            'id': 1,
            'server_id': 'hsd2kw65',
        })
        self.assertTrue(res.ok)
        res = requests.put(addr, json={
            'id': 2,
            'server_id': '55dxfsd3',
        })
        self.assertTrue(res.ok)

    def test_04_delete(self):
        addr = f'{self.addr}/api/db/task'
        res = requests.delete(addr)
        self.assertEqual(res.status_code, 400)

        addr = f'{self.addr}/api/db/agent'
        res = requests.delete(addr, params={'ids': 1})
        self.assertTrue(res.ok)

        addr = f'{self.addr}/api/db/simenv'
        res = requests.delete(addr, params={'ids': [1]})
        self.assertTrue(res.ok)

        addr = f'{self.addr}/api/db/service'
        res = requests.delete(addr, params={'ids': [1, 2]})
        self.assertTrue(res.ok)

        addr = f'{self.addr}/api/db/task'
        res = requests.delete(addr, params={'ids': 1})
        self.assertTrue(res.ok)
