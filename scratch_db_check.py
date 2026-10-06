import sqlite3, os

p = os.path.join('backend_integration', 'events.db')
print('exists', os.path.exists(p))
if not os.path.exists(p):
    raise SystemExit

con = sqlite3.connect(p)
cur = con.cursor()
print('tables', cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall())
rows = cur.execute("SELECT event_id, timestamp, threat_score, transformer_confidence, vae_anomaly_score, dga_probability, predicted_class FROM events ORDER BY timestamp DESC LIMIT 50").fetchall()
print('count', len(rows))
for r in rows:
    print(r)
near = [r for r in rows if abs(float(r[2]) - 10.8) < 0.5 or abs(float(r[2]) - 8.9) < 0.5]
print('near10.8or8.9', near)
