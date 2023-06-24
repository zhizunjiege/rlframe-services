import logging
import mimetypes
import pathlib
import sqlite3
import sys

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
        logger.error(f'SQLite3 error: {e.args}')
        logger.error(f'Exception class is: {e.__class__}')
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
        logger.error(f'SQLite3 error: {e.args}')
        logger.error(f'Exception class is: {e.__class__}')
        res = f'Unknown error: {e.args}', 500
    finally:
        con.close()

    return res


@app.put('/api/db/<string:table>')
def update(table: str):
    data = request.json
    try:
        if 'id' not in data:
            return 'Id not found', 400
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
        logger.error(f'SQLite3 error: {e.args}')
        logger.error(f'Exception class is: {e.__class__}')
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
            return 'Ids not found', 400
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
        logger.error(f'SQLite3 error: {e.args}')
        logger.error(f'Exception class is: {e.__class__}')
        res = f'Unknown error: {e.args}', 500
    finally:
        con.close()

    return res
