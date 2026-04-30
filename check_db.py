from src import db
conn = db.connect()
db.init_db(conn)
tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
print('Tables after init:', tables)
conn.close()