import base64
import mimetypes
import os.path
import sqlite3

from flask import Flask, request, redirect

mimetypes.add_type('application/javascript', '.js')

app = Flask(__name__, static_url_path='', static_folder='static')

db = 'store/db.sqlite3'
need_init = not os.path.exists(db)
con = sqlite3.connect('store/db.sqlite3', check_same_thread=False)
cur = con.cursor()
if need_init:
    with open('store/main.sql') as f:
        script = f.read()
    cur.executescript(script)
    con.commit()

types = {
    'INTEGER': int,
    'REAL': float,
    'TEXT': str,
    'BLOB': bytes,
}

structs = {}
for table in ['simenv', 'agent', 'task']:
    cur.execute(f'PRAGMA TABLE_INFO({table})')
    rst = cur.fetchall()
    structs[table] = {el[1]: {'type': el[2], 'notnull': el[3], 'dflt': el[4], 'pk': el[5]} for el in rst}


def response(code=0, msg='', data=None):
    return {
        'code': code,
        'msg': msg,
        'data': data,
    }


def check_exists(table, columns):
    for column in columns:
        if column not in structs[table]:
            return response(code=1, msg=f'Column {column} not found in table {table}')
    return None


def check_types(table, data):
    for key in data:
        field = structs[table][key]
        if not isinstance(data[key], types[field['type']]):
            return response(code=2, msg=f'Column {key} should be a {field["type"]}')
    return None


def check_null(table, data):
    for key in structs[table]:
        field = structs[table][key]
        if (key not in data or data[key] is None) and field['pk'] == 0 and field['notnull'] == 1 and field['dflt'] is None:
            return response(code=2, msg=f'Column {key} can not be null')
    return None


def bytes_to_b64str(data):
    return base64.b64encode(data).decode('utf-8')


def b64str_to_bytes(data):
    return base64.b64decode(data.encode('utf-8'))


@app.route('/')
def index():
    return redirect('/index.html')


@app.get('/api/db/<string:table>')
def select(table):
    args = request.args
    columns = args.getlist('columns')

    res = check_exists(table, columns)
    if res is not None:
        return res

    query = f'SELECT {", ".join(columns) if len(columns) > 0 else "*"} FROM {table}'
    if 'id' in args:
        query += ' WHERE id = ?'
        params = (int(args['id']),)
    elif 'limit' in args and 'offset' in args:
        query += ' ORDER BY id DESC LIMIT ? OFFSET ?'
        params = (int(args['limit']), int(args['offset']))
    else:
        params = ()

    try:
        cur.execute(query, params)
        rst = cur.fetchall()
        for row in rst:
            for i, col in enumerate(row):
                if type(col) == types['BLOB']:
                    row[i] = bytes_to_b64str(col)
        rst = [list(el) for el in rst]
        return response(data=rst)
    except sqlite3.Error as e:
        print(f'SQLite3 error: {e.args}')
        print(f'Exception class is: {e.__class__}')
        return response(code=3, msg='Unknown error')


@app.post('/api/db/<string:table>')
def insert(table):
    data = request.json

    for key, val in structs[table].items():
        if key in data and val['type'] == 'BLOB':
            data[key] = b64str_to_bytes(data[key])

    res = check_exists(table, data.keys())
    if res is not None:
        return res
    res = check_types(table, data)
    if res is not None:
        return res
    res = check_null(table, data)
    if res is not None:
        return res

    query = f'INSERT INTO {table} ({", ".join(data.keys())}) VALUES ({", ".join(["?" for _ in data])})'
    params = tuple(data.values())

    try:
        cur.execute(query, params)
        con.commit()
        rst = {'rowid': cur.lastrowid}
        return response(data=rst)
    except sqlite3.Error as e:
        print(f'SQLite3 error: {e.args}')
        print(f'Exception class is: {e.__class__}')
        return response(code=3, msg='Unknown error')


@app.put('/api/db/<string:table>/<int:id>')
def update(table, id):
    data = request.json

    for key, val in structs[table].items():
        if key in data and val['type'] == 'BLOB':
            data[key] = b64str_to_bytes(data[key])

    res = check_exists(table, data.keys())
    if res is not None:
        return res
    res = check_types(table, data)
    if res is not None:
        return res

    query = f'UPDATE {table} SET {", ".join([f"{key} = ?" for key in data.keys()])} WHERE id = ?'
    params = tuple(data.values()) + (id,)

    try:
        cur.execute(query, params)
        con.commit()
        return response()
    except sqlite3.Error as e:
        print(f'SQLite3 error: {e.args}')
        print(f'Exception class is: {e.__class__}')
        return response(code=3, msg='Unknown error')


@app.delete('/api/db/<string:table>/<int:id>')
def delete(table, id):
    query = f'DELETE FROM {table} WHERE id = ?'
    params = (id,)

    try:
        cur.execute(query, params)
        con.commit()
        return response()
    except sqlite3.Error as e:
        print(f'SQLite3 error: {e.args}')
        print(f'Exception class is: {e.__class__}')
        return response(code=3, msg='Unknown error')
