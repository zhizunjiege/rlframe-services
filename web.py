import base64
import mimetypes
import os.path
import sqlite3

from flask import Flask, request, redirect

db = 'data/db.sqlite3'
need_init = not os.path.exists(db)
con = sqlite3.connect(db, check_same_thread=False)
cur = con.cursor()
if need_init:
    with open('store/main.sql') as f:
        script = f.read()
    cur.executescript(script)
    con.commit()

tables = {}
for table in ['simenv', 'agent', 'task']:
    cur.execute(f'PRAGMA TABLE_INFO({table})')
    rst = cur.fetchall()
    tables[table] = {el[1]: {'type': el[2].upper(), 'notnull': el[3] and el[4] is None and not el[5]} for el in rst}
types = {
    'INTEGER': int,
    'REAL': float,
    'TEXT': str,
    'BLOB': bytes,
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
def info():
    return tables


@app.get('/api/db/<string:table>')
def select(table):
    args = request.args
    columns = args.getlist('columns')

    if table not in tables:
        return f'Table {table} not found', 404
    else:
        for col in columns:
            if col not in tables[table]:
                return f'Column {col} not found in table {table}', 404

    columns = columns if len(columns) > 0 else tables[table].keys()
    query = f'SELECT {", ".join(columns)} FROM {table}'
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
        data = []
        for row in rst:
            record = {}
            for i, col in enumerate(row):
                if type(col) == types['BLOB']:
                    row[i] = bytes_to_b64str(col)
                record[columns[i]] = row[i]
            data.append(record)
        return data
    except sqlite3.Error as e:
        print(f'SQLite3 error: {e.args}')
        print(f'Exception class is: {e.__class__}')
        return 'Unknown error', 500


@app.post('/api/db/<string:table>')
def insert(table):
    data = request.json

    if table not in tables:
        return f'Table {table} not found', 404
    for col in data.keys():
        if col not in tables[table]:
            return f'Column {col} not found in table {table}', 404
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
        cur.execute(query, params)
        con.commit()
        data = {'rowid': cur.lastrowid}
        return data
    except sqlite3.Error as e:
        print(f'SQLite3 error: {e.args}')
        print(f'Exception class is: {e.__class__}')
        return 'Unknown error', 500


@app.put('/api/db/<string:table>/<int:id>')
def update(table, id):
    data = request.json

    if table not in tables:
        return f'Table {table} not found', 404
    for col in data.keys():
        if col not in tables[table]:
            return f'Column {col} not found in table {table}', 404
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
        cur.execute(query, params)
        con.commit()
        return '', 204
    except sqlite3.Error as e:
        print(f'SQLite3 error: {e.args}')
        print(f'Exception class is: {e.__class__}')
        return 'Unknown error', 500


@app.delete('/api/db/<string:table>/<int:id>')
def delete(table, id):
    query = f'DELETE FROM {table} WHERE id = ?'
    params = (id,)

    try:
        cur.execute(query, params)
        con.commit()
        return '', 204
    except sqlite3.Error as e:
        print(f'SQLite3 error: {e.args}')
        print(f'Exception class is: {e.__class__}')
        return 'Unknown error', 500
