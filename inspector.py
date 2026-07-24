"""
inspector.py — finestra di CONTROLLO/DEBUG per TherMirror (solo per i test)
==========================================================================

Lo specchio resta PULITO (solo il termico, niente riquadri/scritte). Questa e' una
finestra SEPARATA che mostra, in tempo reale:

  - cosa VEDE Mira (il fotogramma della camera);
  - se sta vedendo un volto e CHI e' (conosciuto/sconosciuto);
  - come ti LEGGE: mood, se stai guardando lo specchio, quanto sei vicino, e i segnali
    grezzi (sorriso/sopracciglia/occhi/bocca);
  - lo stato VOCE: se Mira parla adesso, da quanto c'e' silenzio, uscita audio, mic;
  - la ROUTINE proattiva (accesa? intervallo corrente).

E permette di REGOLARE dal vivo alcuni parametri (cursori in alto nella finestra):
  - quanto e' severo il riconoscimento del volto;
  - ogni quanti secondi Mira "guarda" (manda un frame a Gemini);
  - la chiacchiera proattiva: on/off, dopo quanti secondi di silenzio, intervallo minimo.

Tasto 'd' nel programma principale per aprire/chiudere questa finestra.
Nessuna dipendenza extra: usa OpenCV (gia' in uso).
"""

import cv2
import numpy as np

WINDOW = "TherMirror - vision & tuning"

# cursori: (etichetta, minimo, massimo, chiave in `live`, fattore di scala)
# il valore reale = posizione_cursore * scala  (i cursori OpenCV sono interi)
_TRACKBARS = [
    ("recognize x100",  10,  90, "face_threshold",    0.01),   # soglia coseno 0.10..0.90
    ("wide view 0/1",    0,   1, "wide_view",         1.0),    # 1 = tutta la camera (bande nere), 0 = ritaglio
    ("view zoom x100", 100, 300, "person_view_scale", 0.01),   # zoom termico digitale: 1.0=naturale, >1=avvicina
    ("see every s",      1,  15, "video_every_s",     1.0),    # frame a Gemini ogni N s
    ("chatter on/off",   0,   1, "proactive_on",      1.0),    # routine proattiva
    ("quiet s",          5,  90, "proactive_quiet_s", 1.0),    # silenzio prima di parlare
    ("min gap s",       10, 180, "proactive_min_gap", 1.0),    # intervallo minimo
]

_C_LABEL = (150, 150, 150)
_C_VALUE = (235, 245, 255)
_C_GOOD = (140, 240, 170)
_C_WARN = (120, 200, 255)
_C_OFF = (110, 110, 120)


class Inspector:
    """Finestra separata di monitor + regolazioni dal vivo. Vedi docstring del modulo."""

    def __init__(self, live: dict, view_w: int = 460):
        self.live = live               # dizionario condiviso dei parametri modificabili
        self.view_w = view_w
        self.visible = False
        self._built = False

    # ------------------------------------------------------------------ finestra
    def toggle(self):
        self.close() if self.visible else self.open()

    def open(self):
        if self.visible:
            return
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        for label, lo, hi, key, scale in _TRACKBARS:
            start = int(round(float(self.live.get(key, lo)) / scale))
            start = max(lo, min(hi, start))
            cv2.createTrackbar(label, WINDOW, start, hi, lambda _v: None)
            cv2.setTrackbarMin(label, WINDOW, lo)
        self.visible = True
        self._built = True

    def close(self):
        if not self.visible:
            return
        try:
            cv2.destroyWindow(WINDOW)
        except Exception:
            pass
        self.visible = False

    # ------------------------------------------------ leggi i cursori -> `live`
    def _pull_trackbars(self):
        for label, lo, hi, key, scale in _TRACKBARS:
            try:
                pos = cv2.getTrackbarPos(label, WINDOW)
            except Exception:
                continue
            self.live[key] = pos * scale

    # ------------------------------------------------------------------- render
    def render(self, cam_bgr, face_brain, assistant, status: dict):
        """Aggiorna la finestra (se visibile). `status` = stato extra dal loop:
        {'person_here': bool, 'proactive_gap': float}."""
        if not self.visible:
            return
        self._pull_trackbars()

        # 1) cosa vede Mira: il fotogramma, ridimensionato a larghezza fissa
        if cam_bgr is None or cam_bgr.size == 0:
            view = np.zeros((260, self.view_w, 3), np.uint8)
        else:
            h, w = cam_bgr.shape[:2]
            vh = max(1, int(self.view_w * h / w))
            view = cv2.resize(cam_bgr, (self.view_w, vh))

        # 2) pannello testo sotto al video
        lines = self._lines(face_brain, assistant, status)
        panel_h = 18 + 22 * len(lines)
        panel = np.zeros((panel_h, self.view_w, 3), np.uint8)
        panel[:] = (18, 18, 20)
        y = 22
        for text, color in lines:
            cv2.putText(panel, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1, cv2.LINE_AA)
            y += 22

        canvas = np.vstack([view, panel])
        cv2.imshow(WINDOW, canvas)

    # ------------------------------------------------------------ righe di stato
    def _lines(self, face_brain, assistant, status):
        L = []
        # --- chi vede + come ti legge ---
        read = {}
        who = "nobody"
        if face_brain is not None:
            try:
                st, name, score = face_brain.current()
                read = face_brain.read()
                if st == "known" and name:
                    who = f"{name}  (known, {score:.2f})"
                elif st == "unknown":
                    who = f"unknown  (best {score:.2f})"
                elif st == "disabled":
                    who = "face-recognition OFF"
            except Exception:
                pass
        L.append((f"SEE: {who}", _C_VALUE if who != "nobody" else _C_OFF))

        if read.get("face"):
            mood = read.get("mood", "?")
            look = "looking at you" if read.get("attention") else "looking away"
            dist = read.get("proximity", "?")
            L.append((f"READ: {mood.upper()}   {look}   dist:{dist}",
                      _C_GOOD if read.get("attention") else _C_WARN))
            L.append((f"  smile {read.get('smile',0):.2f}  brow {read.get('brow',0):+.2f}  "
                      f"eyes {read.get('eyes_open',0):.2f}  mouth {read.get('mouth_open',0):.2f}",
                      _C_LABEL))
        else:
            on = face_brain is not None and getattr(face_brain, "reader", None) is not None
            L.append(("READ: no face in view" if on else "READ: expression module OFF", _C_OFF))

        # --- voce / audio ---
        if assistant is not None and getattr(assistant, "running", False):
            try:
                speaking = assistant.speaking
                quiet = assistant.quiet_seconds()
                voice = "Mira SPEAKING" if speaking else f"quiet {quiet:.0f}s"
                mic = "muted" if getattr(assistant, "muted", False) else "on"
                out = getattr(assistant, "output_name", "?")
                L.append((f"VOICE: {voice}   out:{out}   mic:{mic}",
                          _C_GOOD if speaking else _C_LABEL))
            except Exception:
                L.append(("VOICE: (unavailable)", _C_OFF))
        else:
            L.append(("VOICE: assistant not running", _C_OFF))

        # --- routine proattiva ---
        on = bool(self.live.get("proactive_on", 0) >= 0.5)
        gap = status.get("proactive_gap", 0.0)
        L.append((f"CHATTER: {'ON' if on else 'off'}   after {self.live.get('proactive_quiet_s',0):.0f}s quiet"
                  f"   gap now {gap:.0f}s", _C_VALUE if on else _C_OFF))

        # --- sguardo (ogni quanto manda un frame a Gemini) ---
        L.append((f"SEES via camera every {self.live.get('video_every_s',0):.0f}s"
                  f"   |   person in frame: {'yes' if status.get('person_here') else 'no'}", _C_LABEL))
        return L
