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
        data = res.json()
        self.assertListEqual(data, [])

        addr = f'{self.addr}/api/db/simenv'
        res = requests.get(addr, params={
            'columns': ['id', 'args'],
            'id': 1,
        })
        data = res.json()
        self.assertListEqual(data, [])

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
        data = res.json()
        self.assertIn('lastrowid', data)
        self.assertEqual(data['lastrowid'], 1)

        addr = f'{self.addr}/api/db/agent'
        res = requests.post(
            addr,
            json={
                'desc': 'test',
                'task': 1,
                'server': 'test',
                'training': 1,
                'name': 'DQN',
                'hypers': '{}',
                'sifunc': '',
                'oafunc': '',
                'rewfunc': '',
                'hooks': '[]',
            },
        )
        data = res.json()
        self.assertIn('lastrowid', data)
        self.assertEqual(data['lastrowid'], 1)

        addr = f'{self.addr}/api/db/simenv'
        res = requests.post(addr, json={
            'desc': 'test',
            'task': 1,
            'server': 'test',
            'name': 'CQSIM',
            'args': '{}',
        })
        data = res.json()
        self.assertIn('lastrowid', data)
        self.assertEqual(data['lastrowid'], 1)

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
        data = res.json()
        self.assertIn('rowcount', data)
        self.assertEqual(data['rowcount'], 1)

        addr = f'{self.addr}/api/db/simenv'
        res = requests.put(addr, json={
            'id': 1,
            'args': '{}',
        })
        data = res.json()
        self.assertIn('rowcount', data)
        self.assertEqual(data['rowcount'], 1)

    def test_04_delete(self):
        addr = f'{self.addr}/api/db/task'
        res = requests.delete(addr)
        self.assertEqual(res.status_code, 400)

        addr = f'{self.addr}/api/db/agent'
        res = requests.delete(addr, params={'ids': 1})
        data = res.json()
        self.assertIn('rowcount', data)
        self.assertEqual(data['rowcount'], 1)

        addr = f'{self.addr}/api/db/simenv'
        res = requests.delete(addr, params={'ids': [1]})
        data = res.json()
        self.assertIn('rowcount', data)
        self.assertEqual(data['rowcount'], 1)

        addr = f'{self.addr}/api/db/task'
        res = requests.delete(addr, params={'ids': 1})
        data = res.json()
        self.assertIn('rowcount', data)
        self.assertEqual(data['rowcount'], 1)

    def test_05_set_task(self):
        addr = f'{self.addr}/api/db/task/set'
        res = requests.post(
            addr,
            json={
                'task': {
                    'id': -1,
                    'name': 'test',
                },
                'agents': [{
                    'id': -1,
                    'task': -1,
                    'server': 'test',
                    'training': 1,
                    'name': 'DQN',
                    'hypers': '{}',
                    'sifunc': '',
                    'oafunc': '',
                    'rewfunc': '',
                    'hooks': '[]',
                }] * 2,
                'simenvs': [{
                    'id': -1,
                    'task': -1,
                    'server': 'test',
                    'name': 'CQSIM',
                    'args': '{}',
                }] * 2,
            },
        )
        data = res.json()
        self.assertIn('task', data)
        self.assertEqual(data['task'], 2)
        self.assertIn('agents', data)
        self.assertListEqual(data['agents'], [2, 3])
        self.assertIn('simenvs', data)
        self.assertListEqual(data['simenvs'], [2, 3])

        res = requests.post(
            addr,
            json={
                'task': {
                    'id': 2,
                    'name': 'test',
                },
                'agents': [{
                    'id': -1,
                    'task': -1,
                    'server': 'test',
                    'training': 1,
                    'name': 'DQN',
                    'hypers': '{}',
                    'sifunc': '',
                    'oafunc': '',
                    'rewfunc': '',
                    'hooks': '[]',
                }, {
                    'id': 2,
                    'server': 'test',
                }],
                'simenvs': [{
                    'id': -1,
                    'task': -1,
                    'server': 'test',
                    'name': 'CQSIM',
                    'args': '{}',
                }, {
                    'id': 3,
                    'server': 'test'
                }],
            },
        )
        data = res.json()
        self.assertIn('task', data)
        self.assertEqual(data['task'], 2)
        self.assertIn('agents', data)
        self.assertListEqual(data['agents'], [4, 2])
        self.assertIn('simenvs', data)
        self.assertListEqual(data['simenvs'], [4, 3])

    @unittest.skip
    def test_06_get_task(self):
        addr = f'{self.addr}/api/db/task/2'
        res = requests.get(addr)
        data = res.json()
        self.assertIn('task', data)
        self.assertEqual(data['task']['id'], 2)
        self.assertIn('agents', data)
        self.assertEqual(len(data['agents']), 2)
        self.assertIn('simenvs', data)
        self.assertEqual(len(data['simenvs']), 2)

    @unittest.skip
    def test_07_del_task(self):
        addr = f'{self.addr}/api/db/task/del'
        res = requests.delete(addr, params={'ids': 2})
        self.assertTrue(res.ok)
