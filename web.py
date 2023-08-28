import logging
import mimetypes
import pathlib
import sqlite3
import sys
import traceback

from flask import Flask, request, redirect
from werkzeug.datastructures import MultiDict

DB_PATH = 'data/db.sqlite3'
SQL_PATH = 'store/schema.sql'

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter('%(asctime)s - %(lineno)d - %(levelname)s - %(message)s'))
logger = logging.getLogger('web')
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def dict_factory(cursor, row):
    fields = [column[0] for column in cursor.description]
    return {key: value for key, value in zip(fields, row)}


def get_db():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = dict_factory
    cur = con.cursor()
    cur.execute('PRAGMA foreign_keys = true')
    return con, cur


def db_init():
    file = pathlib.Path(DB_PATH)
    inited = file.exists()
    if not inited:
        logger.info('Initializing database...')
        file.parent.mkdir(parents=True, exist_ok=True)
        con, cur = get_db()
        with open(SQL_PATH, 'r') as f:
            script = f.read()
        cur.executescript(script)
        con.commit()
        con.close()
        logger.info('Database initialized.')
    else:
        logger.info('Database already initialized.')


db_init()

mimetypes.add_type('application/javascript', '.js')

app = Flask(__name__, static_url_path='', static_folder='static')


@app.route('/')
def index():
    return redirect('/index.html')


@app.get('/api/db/<string:table>')
def select(table: str):
    args = MultiDict(request.args)
    try:
        columns = args.poplist('columns')
        targets = ', '.join(columns) if len(columns) > 0 else '*'
        conjunc = args.pop('conjunc', default='')
        options = [f'{key} = ?' for key in args.keys()]
        filters = f' {conjunc} '.join(options) if len(options) > 0 else '1'
    except Exception as e:
        logger.error(e)
        return 'Invalid arguments', 400

    query = f'SELECT {targets} FROM {table} WHERE {filters}'
    params = tuple(args.values())

    try:
        con, cur = get_db()
        cur.execute(query, params)
        res = cur.fetchall()
    except sqlite3.Error as e:
        traceback.print_exc()
        res = f'Unknown error: {e.args}', 500
    finally:
        con.close()

    return res


@app.post('/api/db/<string:table>')
def insert(table: str):
    data = request.json
    try:
        if 'id' in data:
            del data['id']
        keys = ', '.join(data.keys())
        values = ', '.join(['?' for _ in data])
    except Exception as e:
        logger.error(e)
        return 'Invalid arguments', 400

    query = f'INSERT INTO {table} ({keys}) VALUES ({values})'
    params = tuple(data.values())

    try:
        con, cur = get_db()
        cur.execute(query, params)
        con.commit()
        res = {'lastrowid': cur.lastrowid}
    except sqlite3.Error as e:
        traceback.print_exc()
        res = f'Unknown error: {e.args}', 500
    finally:
        con.close()

    return res


@app.put('/api/db/<string:table>')
def update(table: str):
    data = request.json
    try:
        if 'id' not in data:
            return 'id not found', 400
        else:
            id = data['id']
            del data['id']
        pairs = ', '.join([f'{key} = ?' for key in data.keys()])
    except Exception as e:
        logger.error(e)
        return 'Invalid arguments', 400

    query = f'UPDATE {table} SET {pairs} WHERE id = ?'
    params = tuple(data.values()) + (id,)

    try:
        con, cur = get_db()
        cur.execute(query, params)
        con.commit()
        res = {'rowcount': cur.rowcount}
    except sqlite3.Error as e:
        traceback.print_exc()
        res = f'Unknown error: {e.args}', 500
    finally:
        con.close()

    return res


@app.delete('/api/db/<string:table>')
def delete(table: str):
    args = request.args
    try:
        ids = args.getlist('ids', type=int)
        if len(ids) == 0:
            return 'ids not found', 400
        conds = ', '.join([str(id) for id in ids])
    except Exception as e:
        logger.error(e)
        return 'Invalid arguments', 400

    query = f'DELETE FROM {table} WHERE id in ({conds})'
    params = ()

    try:
        con, cur = get_db()
        cur.execute(query, params)
        con.commit()
        res = {'rowcount': cur.rowcount}
    except sqlite3.Error as e:
        traceback.print_exc()
        res = f'Unknown error: {e.args}', 500
    finally:
        con.close()

    return res


def placeholder(n):
    return ', '.join(['?' for _ in range(n)])


@app.post('/api/db/task/set')
def set_task():
    data = request.json

    try:
        res = {'task': -1, 'agents': [], 'simenvs': []}

        con, cur = get_db()

        task_id = data['task']['id']
        del data['task']['id']
        if task_id < 0:
            query = f'INSERT INTO task ({", ".join(data["task"].keys())}) VALUES ({placeholder(len(data["task"]))})'
            cur.execute(query, tuple(data['task'].values()))
        else:
            query = f'UPDATE task SET {", ".join([f"{key} = ?" for key in data["task"].keys()])} WHERE id = ?'
            cur.execute(query, tuple(data['task'].values()) + (task_id,))

        res['task'] = cur.lastrowid if task_id < 0 else task_id

        insert_agents, update_agents = [], []
        insert_simenvs, update_simenvs = [], []
        if task_id < 0:
            insert_agents = data['agents']
            insert_simenvs = data['simenvs']
        else:
            for agent in data['agents']:
                if agent['id'] < 0:
                    insert_agents.append(agent)
                else:
                    update_agents.append(agent)
            query1 = f'DELETE FROM agent WHERE task = ? AND id NOT IN ({", ".join([str(a["id"]) for a in update_agents])})'
            cur.execute(query1, (task_id,))

            for simenv in data['simenvs']:
                if simenv['id'] < 0:
                    insert_simenvs.append(simenv)
                else:
                    update_simenvs.append(simenv)
            query2 = f'DELETE FROM simenv WHERE task = ? AND id NOT IN ({", ".join([str(s["id"]) for s in update_simenvs])})'
            cur.execute(query2, (task_id,))

        row_ids = []
        if len(insert_agents) > 0:
            for agent in insert_agents:
                del agent['id']
                agent['task'] = res['task']
            query = f'INSERT INTO agent ({", ".join(insert_agents[0].keys())}) VALUES ({placeholder(len(insert_agents[0]))})'
            cur.executemany(query, [tuple(agent.values()) for agent in insert_agents])
            query = 'SELECT MAX(id) FROM agent'
            cur.execute(query)
            max_id = cur.fetchone()['MAX(id)']
            row_ids = list(range(max_id - len(insert_agents) + 1, max_id + 1))
        if len(update_agents) > 0:
            query = f'UPDATE agent SET {", ".join([f"{key} = ?" for key in update_agents[0].keys()])} WHERE id = ?'
            cur.executemany(query, [tuple(agent.values()) + (agent['id'],) for agent in update_agents])
        for agent in data['agents']:
            if 'id' not in agent:
                res['agents'].append(row_ids.pop(0))
            else:
                res['agents'].append(agent['id'])

        row_ids = []
        if len(insert_simenvs) > 0:
            for simenv in insert_simenvs:
                del simenv['id']
                simenv['task'] = res['task']
            query = f'INSERT INTO simenv ({", ".join(insert_simenvs[0].keys())}) VALUES ({placeholder(len(insert_simenvs[0]))})'
            cur.executemany(query, [tuple(simenv.values()) for simenv in insert_simenvs])
            query = 'SELECT MAX(id) FROM simenv'
            cur.execute(query)
            max_id = cur.fetchone()['MAX(id)']
            row_ids = list(range(max_id - len(insert_simenvs) + 1, max_id + 1))
        if len(update_simenvs) > 0:
            query = f'UPDATE simenv SET {", ".join([f"{key} = ?" for key in update_simenvs[0].keys()])} WHERE id = ?'
            cur.executemany(query, [tuple(simenv.values()) + (simenv['id'],) for simenv in update_simenvs])
        for simenv in data['simenvs']:
            if 'id' not in simenv:
                res['simenvs'].append(row_ids.pop(0))
            else:
                res['simenvs'].append(simenv['id'])

        con.commit()

        return res
    except sqlite3.Error as e:
        traceback.print_exc()
        return f'Unknown error: {e.args}', 500
    finally:
        con.close()


@app.get('/api/db/task/<int:id>')
def get_task(id: int):
    try:
        con, cur = get_db()

        query0 = 'SELECT * FROM task WHERE id = ?'
        cur.execute(query0, (id,))
        task = cur.fetchone()
        if task is None:
            return 'Task not found', 404

        query1 = 'SELECT * FROM agent WHERE task = ?'
        cur.execute(query1, (id,))
        agents = cur.fetchall()

        query2 = 'SELECT * FROM simenv WHERE task = ?'
        cur.execute(query2, (id,))
        simenvs = cur.fetchall()

        return {'task': task, 'agents': agents, 'simenvs': simenvs}
    except sqlite3.Error as e:
        traceback.print_exc()
        return f'Unknown error: {e.args}', 500
    finally:
        con.close()


@app.delete('/api/db/task/del')
def del_task():
    args = request.args

    try:
        con, cur = get_db()

        ids = args.getlist('ids', type=int)
        if len(ids) == 0:
            return 'ids not found', 400

        query = f'DELETE FROM task WHERE id in ({", ".join([str(id) for id in ids])})'
        cur.execute(query)
        con.commit()
        return 'OK', 200
    except sqlite3.Error as e:
        traceback.print_exc()
        return f'Unknown error: {e.args}', 500
    finally:
        con.close()
