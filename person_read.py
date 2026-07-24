"""
person_read.py — legge l'ESPRESSIONE e lo STATO di chi e' davanti (TherMirror)
=============================================================================

Se face_brain da' a Mira "occhi che RICONOSCONO", questo modulo le da' "occhi che
CAPISCONO come stai": guarda il volto piu' in vista e ne ricava una lettura semplice:

  - mood        una parola sull'umore:  happy | tired | focused | down | surprised | neutral
  - smile / brow / eyes / jaw   i segnali grezzi (0..1) da cui deriva il mood
  - attention   True se la persona sta GUARDANDO lo specchio (non girata di lato)
  - proximity   near | mid | far  (quanto e' vicina, dalla dimensione del volto)

Backend: MediaPipe FaceLandmarker (Tasks API) con "blendshapes" (i 52 coefficienti stile
ARKit: mouthSmile, browDown, jawOpen, eyeBlink, ...). Robusto e leggero, gira su CPU
(anche su Raspberry Pi). Al primo avvio SCARICA il modello (~3.7 MB) accanto agli altri file.

Se MediaPipe manca o il modello non si scarica, il modulo si DISABILITA in modo pulito:
lo specchio continua a funzionare, semplicemente senza lettura dell'espressione.

Uso:
    r = FaceRead()                 # crea (scarica il modello se serve)
    read = r.analyze(frame_bgr)    # dict con mood/attention/... (o {"face": False})
"""

import os
import urllib.request
import numpy as np

# modello ufficiale MediaPipe (float16, con blendshapes)
_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
              "face_landmarker/float16/1/face_landmarker.task")
_MODEL_FILE = "face_landmarker.task"

# --- soglie (tarabili) -------------------------------------------------------
# quanto un segnale deve essere "acceso" per contare (blendshapes vanno 0..1)
_SMILE_ON = 0.35     # sorriso evidente
_JAW_ON = 0.35       # bocca aperta (sorpresa / parla)
_BROW_UP_ON = 0.30   # sopracciglia alzate (sorpresa / preoccupazione)
_BROW_DN_ON = 0.35   # sopracciglia aggrottate (concentrazione / tensione)
_FROWN_ON = 0.25     # bocca all'ingiu'
_EYES_SHUT = 0.15    # sotto questa "apertura" gli occhi sono di fatto chiusi
# attenzione / vicinanza (dalla geometria dei landmark, coordinate normalizzate 0..1)
_FACING_TOL = 0.18   # |naso fuori centro| oltre questo = sta guardando altrove
_NEAR_W = 0.30       # larghezza volto (frazione del frame) -> vicino
_FAR_W = 0.13        # ... -> lontano
# lisciatura temporale: media esponenziale, evita che il mood "sfarfalli"
_SMOOTH = 0.5        # 0 = nessuna memoria, ->1 = molto pigro

# indici landmark del "canonical face" MediaPipe usati per posa/vicinanza
_NOSE_TIP = 1
_EYE_OUTER_A = 33
_EYE_OUTER_B = 263


def _ensure_model(path: str) -> bool:
    """Assicura che il file del modello esista (scarica al primo avvio). True se pronto."""
    if os.path.exists(path):
        return True
    try:
        print(f"[read] scarico il modello volto (~3.7MB) -> {path} ...")
        urllib.request.urlretrieve(_MODEL_URL, path)
        print("[read] modello volto scaricato.")
        return True
    except Exception as exc:            # niente rete / storage: si disabilita pulito
        print(f"[read] impossibile scaricare il modello: {repr(exc)[:100]}")
        return False


class FaceRead:
    """Lettura espressione/stato del volto piu' in vista. Vedi docstring del modulo."""

    def __init__(self, model_path: str = _MODEL_FILE, max_width: int = 512):
        self.landmarker = None
        self.max_width = max_width          # ridimensiona il frame per risparmiare CPU
        self._ema = {}                      # segnali lisciati nel tempo
        self._init_backend(model_path)

    def _init_backend(self, model_path):
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
            if not _ensure_model(model_path):
                return
            opts = vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=model_path),
                running_mode=vision.RunningMode.IMAGE,   # senza stato: un frame alla volta
                num_faces=1,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=False,
            )
            self._mp = mp
            self.landmarker = vision.FaceLandmarker.create_from_options(opts)
            print("[read] lettura espressione attiva (MediaPipe FaceLandmarker)")
        except Exception as exc:
            print(f"[read] lettura espressione NON attiva: {repr(exc)[:120]}")
            print('[read] per attivarla:  pip install "numpy<2" mediapipe')

    @property
    def available(self) -> bool:
        return self.landmarker is not None

    # ------------------------------------------------------------------ analisi
    def analyze(self, frame_bgr) -> dict:
        """Ritorna un dict di lettura. Se non vede un volto: {'face': False}."""
        if self.landmarker is None:
            return {"face": False}
        try:
            import cv2
            img = frame_bgr
            if self.max_width and img.shape[1] > self.max_width:
                s = self.max_width / img.shape[1]
                img = cv2.resize(img, None, fx=s, fy=s)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
            res = self.landmarker.detect(mp_img)
        except Exception as exc:            # pragma: no cover
            print(f"[read] analisi fallita: {repr(exc)[:80]}")
            return {"face": False}

        if not res.face_blendshapes or not res.face_landmarks:
            self._ema.clear()               # nessun volto -> dimentica lo stato lisciato
            return {"face": False}

        bs = {c.category_name: c.score for c in res.face_blendshapes[0]}
        pts = res.face_landmarks[0]

        # segnali grezzi (0..1), poi lisciati nel tempo per non sfarfallare
        raw = {
            "smile": max(bs.get("mouthSmileLeft", 0), bs.get("mouthSmileRight", 0)),
            "frown": max(bs.get("mouthFrownLeft", 0), bs.get("mouthFrownRight", 0)),
            "brow_dn": max(bs.get("browDownLeft", 0), bs.get("browDownRight", 0)),
            "brow_up": bs.get("browInnerUp", 0),
            "jaw": bs.get("jawOpen", 0),
            # "apertura occhi" = 1 - quanto sono chiusi (media dei due)
            "eyes": 1.0 - min(1.0, (bs.get("eyeBlinkLeft", 0) + bs.get("eyeBlinkRight", 0)) / 2.0),
        }
        sig = self._smooth(raw)

        # --- posa/vicinanza dalla geometria dei landmark (coordinate 0..1) ---
        xs = [p.x for p in pts]
        face_w = max(xs) - min(xs)
        eA, eB, nose = pts[_EYE_OUTER_A], pts[_EYE_OUTER_B], pts[_NOSE_TIP]
        span = abs(eB.x - eA.x)
        # naso ~ a meta' tra gli angoli esterni degli occhi quando si guarda dritto
        facing_ratio = (nose.x - min(eA.x, eB.x)) / span if span > 1e-6 else 0.5
        attention = abs(facing_ratio - 0.5) < _FACING_TOL and sig["eyes"] > _EYES_SHUT

        proximity = "near" if face_w > _NEAR_W else "far" if face_w < _FAR_W else "mid"
        mood = self._mood(sig)

        return {
            "face": True,
            "mood": mood,
            "attention": bool(attention),
            "proximity": proximity,
            "smile": round(sig["smile"], 2),
            "brow": round(sig["brow_up"] - sig["brow_dn"], 2),   # + alzate / - aggrottate
            "eyes_open": round(sig["eyes"], 2),
            "mouth_open": round(sig["jaw"], 2),
        }

    # --------------------------------------------------------------- interni
    def _smooth(self, raw: dict) -> dict:
        for k, v in raw.items():
            prev = self._ema.get(k, v)
            self._ema[k] = _SMOOTH * prev + (1 - _SMOOTH) * v
        return dict(self._ema)

    @staticmethod
    def _mood(s: dict) -> str:
        """Regola semplice e spiegabile da segnali lisciati (0..1)."""
        if s["eyes"] < _EYES_SHUT:
            return "tired"                       # occhi socchiusi a lungo
        if s["smile"] > _SMILE_ON:
            return "happy"
        if s["jaw"] > _JAW_ON and s["brow_up"] > _BROW_UP_ON:
            return "surprised"
        if s["brow_dn"] > _BROW_DN_ON:
            return "focused"                     # sopracciglia aggrottate
        if s["frown"] > _FROWN_ON or (s["brow_up"] > 0.40 and s["smile"] < 0.10):
            return "down"
        return "neutral"

    @staticmethod
    def phrase(read: dict) -> str:
        """Frase breve e naturale (inglese, come la persona di Mira). '' se niente da dire."""
        if not read or not read.get("face"):
            return ""
        mood = read.get("mood", "neutral")
        bits = []
        mood_txt = {
            "happy": "looks happy",
            "tired": "seems tired",
            "focused": "looks focused",
            "down": "seems a bit down",
            "surprised": "looks surprised",
        }.get(mood)                              # 'neutral' -> niente (evita rumore)
        if mood_txt:
            bits.append(mood_txt)
        if read.get("attention") is False:
            bits.append("not looking at you")
        return ", ".join(bits)
