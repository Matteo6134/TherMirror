"""
home_brain.py — MEMORIA EPISODICA della casa per TherMirror (Fase 2)
====================================================================

Il "grande cervello" che registra la vita in casa nel tempo, cosi' Mira puo' rispondere a
domande come "chi e' passato oggi?", "quando e' venuto Luca l'ultima volta?", "dove ho
lasciato le chiavi?".

Due sorgenti di eventi:
  - AUTOMATICI: le presenze (chi viene riconosciuto, quando, a quale "camera") vengono
    registrate dal programma quando il riconoscimento volti vede qualcuno;
  - OSSERVAZIONI: Mira stessa, che VEDE dalla camera, puo' annotare cose notevoli
    (es. "Matteo ha posato le chiavi sul tavolo dell'ingresso") con log_observation, e
    ritrovarle con recall.

Archivio: SQLite locale (semplice, robusto, interrogabile). Piu' avanti si potranno
aggiungere embedding per una ricerca semantica; per ora la ricerca e' per parole/tempo.
Tutto in locale, sulla macchina.
"""

import time
import sqlite3
import threading
from datetime import datetime


def _human_time(ts: float, now: float = None) -> str:
    """Timestamp -> stringa in italiano ('poco fa', 'oggi 14:30', 'ieri 20:10', '3 mar 09:15')."""
    now = now if now is not None else time.time()
    dt = datetime.fromtimestamp(ts)
    delta = now - ts
    if delta < 90:
        return "poco fa"
    if delta < 3600:
        return f"{int(delta // 60)} min fa"
    today = datetime.fromtimestamp(now).date()
    day = dt.date()
    hm = dt.strftime("%H:%M")
    if day == today:
        return f"oggi {hm}"
    if (today - day).days == 1:
        return f"ieri {hm}"
    return dt.strftime("%d/%m %H:%M")


class HomeBrain:
    """Registro eventi persistente. Thread-safe (una connessione condivisa + lock)."""

    def __init__(self, path="home_events.db"):
        self.path = path
        self._lock = threading.Lock()
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS events ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " ts REAL, kind TEXT, who TEXT, place TEXT, detail TEXT)")
        self.db.commit()
        # anti-doppioni per le presenze (stessa persona ravvicinata -> un solo evento)
        self._last_presence = {}

    # ------------------------------------------------------------------ scrittura
    def log(self, kind, who=None, place=None, detail=None):
        with self._lock:
            self.db.execute(
                "INSERT INTO events (ts, kind, who, place, detail) VALUES (?,?,?,?,?)",
                (time.time(), kind, who, place, detail))
            self.db.commit()

    def log_presence(self, who, place="specchio", min_gap_s=120):
        """Registra che `who` e' stato visto in `place`. Ignora ripetizioni troppo
        ravvicinate (stessa presenza) per non riempire il registro."""
        now = time.time()
        key = (who, place)
        if now - self._last_presence.get(key, 0) < min_gap_s:
            return
        self._last_presence[key] = now
        self.log("presence", who=who, place=place)

    # ------------------------------------------------------------------ lettura
    def who_was_here(self, hours=24):
        since = time.time() - hours * 3600
        with self._lock:
            rows = self.db.execute(
                "SELECT who, MAX(ts) FROM events WHERE kind='presence' AND ts>=? "
                "GROUP BY who ORDER BY MAX(ts) DESC", (since,)).fetchall()
        if not rows:
            return f"nelle ultime {hours} ore non ho visto nessuno."
        parts = [f"{who or 'qualcuno'} ({_human_time(ts)})" for who, ts in rows]
        return "ho visto: " + "; ".join(parts)

    def last_seen(self, name):
        with self._lock:
            row = self.db.execute(
                "SELECT MAX(ts) FROM events WHERE who=? AND kind='presence'",
                (name,)).fetchone()
        if not row or row[0] is None:
            return f"non risulta che io abbia mai visto {name}."
        return f"{name} l'ho visto/a l'ultima volta {_human_time(row[0])}."

    def recall(self, query="", hours=None, limit=8):
        """Cerca tra osservazioni/note/presenze per parole chiave (e opzionalmente tempo).
        Ritorna un riassunto testuale degli eventi piu' recenti che combaciano."""
        words = [w.lower() for w in (query or "").split() if len(w) > 2]
        clauses, params = ["kind IN ('observation','note','presence')"], []
        if hours:
            clauses.append("ts>=?")
            params.append(time.time() - hours * 3600)
        sql = ("SELECT ts, who, place, detail, kind FROM events WHERE "
               + " AND ".join(clauses) + " ORDER BY ts DESC LIMIT 200")
        with self._lock:
            rows = self.db.execute(sql, params).fetchall()
        hits = []
        for ts, who, place, detail, kind in rows:
            blob = " ".join(str(x) for x in (who, place, detail, kind) if x).lower()
            if not words or all(w in blob for w in words):
                desc = detail or (f"visto/a {who}" if who else kind)
                where = f" ({place})" if place else ""
                hits.append(f"{_human_time(ts)}: {desc}{where}")
            if len(hits) >= limit:
                break
        if not hits:
            return "non trovo niente del genere nei miei ricordi."
        return " | ".join(hits)

    def recent(self, limit=10):
        with self._lock:
            rows = self.db.execute(
                "SELECT ts, who, detail, kind FROM events ORDER BY ts DESC LIMIT ?",
                (limit,)).fetchall()
        return [(ts, who, detail, kind) for ts, who, detail, kind in rows]
