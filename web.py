import base64
import logging
import mimetypes
import pathlib
import sqlite3

from flask import Flask, request, redirect


def dict_factory(cursor, row):
    fields = [column[0] for column in cursor.description]
    return {key: value for key, value in zip(fields, row)}


def get_db(db_path):
    con = sqlite3.connect(db_path)
    con.row_factory = dict_factory
    cur = con.cursor()
    return con, cur


logging.basicConfig(
    format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s',
    level='ERROR',
)

db_path = pathlib.Path('data/db.sqlite3')
need_init = not db_path.exists()
con, cur = get_db(db_path)
if need_init:
    with open('store/main.sql', 'r') as f:
        script = f.read()
    cur.executescript(script)
    con.commit()

types = {
    'INTEGER': int,
    'REAL': float,
    'TEXT': str,
    'BLOB': bytes,
    'JSON': str,
}
jsons = {
    'simenv': ['args', 'params'],
    'agent': ['hypers', 'status'],
    'task': ['services', 'routes'],
}
tables = {}
for table in ['simenv', 'agent', 'task']:
    tables[table] = {}
    cur.execute(f'PRAGMA TABLE_INFO({table})')
    rows = cur.fetchall()
    for row in rows:
        if table in jsons and row['name'] in jsons[table]:
            row['type'] = 'JSON'
        tables[table][row['name']] = {
            'type': row['type'].upper(),
            'notnull': row['notnull'] and row['dflt_value'] is None and not row['pk'],
        }


def bytes_to_b64str(data):
    return base64.b64encode(data).decode('utf-8')


def b64str_to_bytes(data):
    return base64.b64decode(data.encode('utf-8'))


mimetypes.add_type('application/javascript', '.js')

app = Flask(__name__, static_url_path='', static_folder='static')


@app.route('/')
def index():
    return redirect('/index.html')


@app.route('/api/db')
def meta():
    return tables


@app.get('/api/db/<string:table>')
def select(table):
    args = request.args
    try:
        id = args.get('id', type=int)
        limit = args.get('limit', type=int)
        offset = args.get('offset', type=int)
        columns = args.getlist('columns')
    except Exception:
        return 'Invalid query parameters', 400

    if table not in tables:
        return f'Table {table} not found', 404
    else:
        for col in columns:
            if col not in tables[table]:
                return f'Column {col} not found in table {table}', 404

    columns = columns if len(columns) > 0 else list(tables[table].keys())
    query = f'SELECT {", ".join(columns)} FROM {table}'
    if id is not None:
        query += ' WHERE id = ?'
        params = (id,)
    elif limit is not None and offset is not None:
        query += ' ORDER BY id DESC LIMIT ? OFFSET ?'
        params = (limit, offset)
    else:
        params = ()

    try:
        _, cur = get_db(db_path)
        cur.execute(query, params)
        rows = cur.fetchall()
        for row in rows:
            for col in row:
                if row[col] is not None:
                    col_type = tables[table][col]['type']
                    if col_type == 'BLOB':
                        row[col] = bytes_to_b64str(row[col])
        return rows
    except sqlite3.Error as e:
        logging.error(f'SQLite3 error: {e.args}')
        logging.error(f'Exception class is: {e.__class__}')
        return 'Unknown error', 500


@app.post('/api/db/<string:table>')
def insert(table):
    data = request.json

    if table not in tables:
        return f'Table {table} not found', 404
    if 'id' in data:
        del data['id']
    for col in data.keys():
        if col not in tables[table]:
            return f'Column {col} not found in table {table}', 404
        if data[col] is not None:
            col_type = tables[table][col]['type']
            if col_type == 'BLOB':
                try:
                    data[col] = b64str_to_bytes(data[col])
                except Exception:
                    return f'Column {col} is not a valid base64 string', 400
            elif not isinstance(data[col], types[col_type]):
                return f'Column {col} should be a {col_type}', 400
    for col in tables[table]:
        if tables[table][col]['notnull'] and (col not in data or data[col] is None):
            return f'Column {col} can not be NULL', 400

    query = f'INSERT INTO {table} ({", ".join(data.keys())}) VALUES ({", ".join(["?" for _ in data])})'
    params = tuple(data.values())

    try:
        con, cur = get_db(db_path)
        cur.execute(query, params)
        con.commit()
        data = {'lastrowid': cur.lastrowid}
        return data
    except sqlite3.Error as e:
        logging.error(f'SQLite3 error: {e.args}')
        logging.error(f'Exception class is: {e.__class__}')
        return 'Unknown error', 500


@app.put('/api/db/<string:table>')
def update(table):
    data = request.json

    if table not in tables:
        return f'Table {table} not found', 404
    if 'id' not in data:
        return 'Missing id', 400
    else:
        id = data['id']
        del data['id']
    for col in data.keys():
        if col not in tables[table]:
            return f'Column {col} not found in table {table}', 404
        if data[col] is not None:
            col_type = tables[table][col]['type']
            if col_type == 'BLOB':
                try:
                    data[col] = b64str_to_bytes(data[col])
                except Exception:
                    return f'Column {col} is not a valid base64 string', 400
            elif not isinstance(data[col], types[col_type]):
                return f'Column {col} should be a {col_type}', 400

    query = f'UPDATE {table} SET {", ".join([f"{key} = ?" for key in data.keys()])} WHERE id = ?'
    params = tuple(data.values()) + (id,)

    try:
        con, cur = get_db(db_path)
        cur.execute(query, params)
        con.commit()
        data = {'rowcount': cur.rowcount}
        return data
    except sqlite3.Error as e:
        logging.error(f'SQLite3 error: {e.args}')
        logging.error(f'Exception class is: {e.__class__}')
        return 'Unknown error', 500


@app.delete('/api/db/<string:table>')
def delete(table):
    data = request.json

    if table not in tables:
        return f'Table {table} not found', 404
    if 'ids' not in data:
        return 'Missing ids', 400
    elif not isinstance(data['ids'], list):
        return 'Data ids should be a list', 400
    else:
        ids = data['ids']

    query = f'DELETE FROM {table} WHERE id in ({", ".join([str(id) for id in ids])})'
    params = ()

    try:
        con, cur = get_db(db_path)
        cur.execute(query, params)
        con.commit()
        data = {'rowcount': cur.rowcount}
        return data
    except sqlite3.Error as e:
        logging.error(f'SQLite3 error: {e.args}')
        logging.error(f'Exception class is: {e.__class__}')
        return 'Unknown error', 500
