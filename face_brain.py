"""
face_brain.py — riconoscimento e MEMORIA dei volti per TherMirror (Fase 1)
==========================================================================

Da' a Mira degli "occhi che ricordano le persone":
  - rileva il volto piu' in vista dalla camera e lo trasforma in un'impronta numerica;
  - lo confronta con le persone gia' conosciute (PeopleMemory, salvata su file);
  - dice se vede QUALCUNO che CONOSCE (con il nome), uno SCONOSCIUTO, o NESSUNO;
  - puo' IMPARARE un volto nuovo al volo (enroll_current) associandolo a un nome:
    da li' in poi quella persona verra' riconosciuta.

Il riconoscimento gira in un THREAD in background (e' pesante) e aggiorna un risultato
condiviso; il programma principale legge current() a costo ~zero.

Backend: insightface (ArcFace) via onnxruntime.
  pip install insightface onnxruntime
Se non e' installato, il modulo si DISABILITA in modo pulito: lo specchio continua a
funzionare, semplicemente senza riconoscimento dei volti.
"""

import os
import json
import time
import threading
import warnings
import numpy as np

# insightface allinea i volti via scikit-image, che sta deprecando estimate(): e' solo
# rumore (non tocca il riconoscimento), ma stamperebbe a ogni volto -> lo zittiamo.
warnings.filterwarnings("ignore", category=FutureWarning,
                        message=r".*`?estimate`? is deprecated.*")


def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


class PeopleMemory:
    """Persone conosciute: nome -> lista di impronte del volto (embedding L2-normalizzati).
    Persistente su file JSON, cosi' Mira 'costruisce' nel tempo chi conosce."""

    MAX_PER_PERSON = 8   # quante impronte tenere per persona (angoli/luci diversi)

    def __init__(self, path: str):
        self.path = path
        self.people = {}          # name -> list[np.ndarray]
        self._lock = threading.Lock()
        self._load()

    def _load(self):
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.people = {n: [np.asarray(e, dtype=np.float32) for e in embs]
                           for n, embs in data.items()}
            print(f"[face] memoria persone caricata: {', '.join(self.people) or 'vuota'}")
        except Exception:
            self.people = {}

    def _save(self):
        try:
            data = {n: [e.tolist() for e in embs] for n, embs in self.people.items()}
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(data, f)
        except Exception as exc:            # pragma: no cover
            print(f"[face] salvataggio memoria persone fallito: {repr(exc)[:80]}")

    def add(self, name: str, emb: np.ndarray):
        name = (name or "").strip()
        if not name:
            return
        emb = _normalize(emb)
        with self._lock:
            embs = self.people.setdefault(name, [])
            embs.append(emb)
            self.people[name] = embs[-self.MAX_PER_PERSON:]
            self._save()
        print(f"[face] imparata/aggiornata persona: {name} ({len(self.people[name])} volti)")

    def identify(self, emb: np.ndarray, threshold: float):
        """Ritorna (nome, similarita') se supera la soglia, altrimenti (None, migliore_sim)."""
        emb = _normalize(emb)
        best_name, best = None, -1.0
        with self._lock:
            for name, embs in self.people.items():
                for e in embs:
                    s = float(np.dot(emb, e))     # coseno (vettori normalizzati)
                    if s > best:
                        best, best_name = s, name
        return (best_name, best) if best >= threshold else (None, best)

    def names(self):
        with self._lock:
            return list(self.people.keys())


class FaceBrain:
    """Riconoscimento volti + memoria persone. Uso:
        fb = FaceBrain("people_memory.json")
        fb.start()                 # avvia il thread di analisi
        ...ogni frame...  fb.submit(frame_bgr)
        status, name, score = fb.current()   # 'known'|'unknown'|'none'|'disabled'
        fb.enroll_current("Luca")  # impara il volto attuale come 'Luca'
    """

    def __init__(self, memory_path="people_memory.json", threshold=0.35,
                 model="buffalo_sc", every_s=0.5, read=True):
        self.mem = PeopleMemory(memory_path)
        self.threshold = threshold
        self.every_s = every_s
        self.app = None
        self.last_embedding = None          # impronta dell'ultimo volto visto (per enroll)
        self.last_sex = None                # 'M' / 'F' stimato (se il modello lo supporta)
        self.last_age = None                # eta' stimata
        self.reader = None                  # lettura espressione/stato (person_read.FaceRead)
        self.last_read = {"face": False}    # ultima lettura mood/attenzione/vicinanza
        self._pending = None                # ultimo frame da analizzare
        self._status = ("none", None, -1.0)
        self._lock = threading.Lock()
        self._init_backend(model)
        if read:
            try:
                from person_read import FaceRead
                r = FaceRead()
                self.reader = r if r.available else None
            except Exception as exc:
                print(f"[read] lettura espressione non avviata: {repr(exc)[:120]}")

    def _init_backend(self, model):
        try:
            from insightface.app import FaceAnalysis
            last_err = None
            for name in (model, "buffalo_l"):
                try:
                    # 'genderage' -> stima uomo/donna ed eta' (presente in buffalo_l;
                    # in buffalo_sc puo' mancare: in tal caso il modulo viene ignorato)
                    app = FaceAnalysis(name=name,
                                       allowed_modules=["detection", "recognition", "genderage"],
                                       providers=["CPUExecutionProvider"])
                    app.prepare(ctx_id=-1, det_size=(320, 320))
                    self.app = app
                    print(f"[face] riconoscimento volti attivo (insightface '{name}')")
                    return
                except Exception as exc:
                    last_err = exc
            raise RuntimeError(last_err)
        except Exception as exc:
            print(f"[face] riconoscimento volti NON attivo: {repr(exc)[:120]}")
            print("[face] per attivarlo:  pip install insightface onnxruntime")

    @property
    def available(self) -> bool:
        return self.app is not None

    # -------- thread di analisi (il riconoscimento e' pesante -> fuori dal render) -----
    def start(self):
        if self.app is None and self.reader is None:
            return
        threading.Thread(target=self._loop, daemon=True).start()

    def submit(self, frame_bgr):
        """Consegna l'ultimo frame (costo ~zero): il thread analizza al suo ritmo."""
        self._pending = frame_bgr

    def _loop(self):
        while True:
            time.sleep(self.every_s)
            frame = self._pending
            if frame is None:
                continue
            self._status = self.analyze(frame)

    def current(self):
        return self._status

    # ------------------------------------------------------------------ analisi
    def analyze(self, frame_bgr):
        """(status, name, score). status: 'known' | 'unknown' | 'none' | 'disabled'."""
        # lettura espressione/stato (indipendente dall'identita': gira comunque)
        if self.reader is not None:
            try:
                self.last_read = self.reader.analyze(frame_bgr)
            except Exception:
                pass
        if self.app is None:
            return ("disabled", None, -1.0)
        try:
            faces = self.app.get(frame_bgr)
        except Exception as exc:            # pragma: no cover
            print(f"[face] analisi fallita: {repr(exc)[:80]}")
            return ("none", None, -1.0)
        if not faces:
            self.last_embedding = None
            self.last_sex = self.last_age = None
            return ("none", None, -1.0)
        # il volto piu' grande = quello piu' vicino / in primo piano
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        emb = getattr(face, "normed_embedding", None)
        if emb is None:
            emb = face.embedding
        self.last_embedding = np.asarray(emb, dtype=np.float32)
        # stima uomo/donna ed eta' (solo se il modello genderage e' presente)
        try:
            self.last_sex = getattr(face, "sex", None)
            a = getattr(face, "age", None)
            self.last_age = int(a) if a is not None else None
        except Exception:
            self.last_sex = self.last_age = None
        name, score = self.mem.identify(self.last_embedding, self.threshold)
        if name:
            return ("known", name, score)
        return ("unknown", None, score)

    def describe(self) -> str:
        """Plain-words description of who is in front now: identity (name if known,
        otherwise estimated man/woman + age) PLUS how they seem right now (mood /
        looking at you). Used to let Mira adapt and as greeting context."""
        status, name, _ = self._status
        if status == "known" and name:
            who = f"{name}"
        elif self.last_sex in ("M", "F"):
            g = "a man" if self.last_sex == "M" else "a woman"
            age = f", around {self.last_age}" if self.last_age else ""
            who = f"{g}{age} you don't know"
        elif status == "unknown":
            who = "someone you don't know"
        else:
            who = ""
        # aggiunge lo stato momentaneo (umore / attenzione), se disponibile
        try:
            from person_read import FaceRead
            state = FaceRead.phrase(self.last_read)
        except Exception:
            state = ""
        if who and state:
            return f"{who}; {state}"
        return who or state

    def read(self) -> dict:
        """Ultima lettura espressione/stato (mood, attention, proximity, ...).
        Per far reagire anche i VISUAL (colore termico, blob) allo stato della persona."""
        return dict(self.last_read or {})

    def enroll_current(self, name: str) -> str:
        """Impara il volto attualmente visto come `name`. Messaggio per Mira."""
        emb = self.last_embedding
        if emb is None:
            return "non riesco a vedere bene un volto adesso; mettiti davanti allo specchio."
        self.mem.add(name, emb)
        return f"ok, ora ricordero' il volto di {name}."

    def known_names(self):
        return self.mem.names()
