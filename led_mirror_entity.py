"""
LED Mirror "Entity" - pipeline TERMICA + riconoscimento oggetti
================================================================

Pipeline:
  1. Cattura camera normale
  2. Segmentazione: PERSONA via MediaPipe Selfie Segmentation (maschera del corpo);
     OGGETTI via OpenCV (background subtraction: primo piano MENO la persona = cio' che
     porti in scena, con forma reale). Tutto locale, niente cloud.
  3. Rendering TERMICO (stile AMG8833): persona in una colormap termica calda, oggetti in
     una colormap DIVERSA -> si distinguono restando "in modalita' termica".
     Sul Raspberry Pi con sensore IR reale (AMG8833) legge le temperature via I2C.
     https://github.com/makerportal/AMG8833_IR_cam
  4. Assistente vocale "vivo" (Gemini Live, vedi voice_assistant.py): conversazione a voce
     in tempo reale con personalita', vede la camera, usa Google Search (meteo/news/azioni),
     con trascrizione a schermo.
  5. Output su pannello LED HUB75 (rpi-rgb-led-matrix) con fallback a preview su schermo.

Dipendenze:
  pip install "numpy<2" opencv-python pillow mediapipe google-genai sounddevice --break-system-packages
    (mediapipe -> persona; opencv -> oggetti+pipeline; google-genai+sounddevice -> assistente vocale)
    NB: serve numpy<2 (mediapipe non e' compatibile con numpy 2.x)
  Chiave Gemini (gratuita: https://aistudio.google.com/apikey), da variabile d'ambiente:
    export GEMINI_API_KEY="la-tua-chiave"
  Su macOS servono i permessi Fotocamera E Microfono per il terminale/IDE.
  (sul Raspberry Pi, per l'output reale: https://github.com/hzeller/rpi-rgb-led-matrix)
"""

import os
import math
import time
import random
import threading
from datetime import datetime
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

# Assemblaggio pannello LED: griglia di moduli HUB75 da 64x64.
# Attuale: 3 moduli in orizzontale (x) x 9 in verticale (y) = 192 x 576 px totali.
PANEL_MODULE = 64          # lato di un singolo modulo (64x64)
PANELS_X = 4               # moduli in orizzontale
PANELS_Y = 9               # moduli in verticale
# Formato pannello 4x9 = 256 x 576 (ritratto). Mostrato a schermo mantenendo le proporzioni
# (in fullscreen viene centrato con bande nere ai lati, senza deformare).
PANEL_WIDTH = PANEL_MODULE * PANELS_X    # 256
PANEL_HEIGHT = PANEL_MODULE * PANELS_Y   # 576
FULLSCREEN = False         # finestra 4x9 (ritratto) ridimensionabile, come prima. True = tutto schermo
TARGET_FPS = 24            # tetto ai fotogrammi -> lascia CPU all'audio (mic/voce fluidi)
CAMERA_INDEX = 0
PERSON_VIEW_SCALE = 1.0    # zoom digitale del termico: 1.0 = vista naturale (riempie), >1 = avvicina
# Vista LARGA: mostra TUTTA la camera nel pannello (con bande nere sopra/sotto) invece di
# ritagliare la stretta striscia centrale -> si vede molto di piu', ma la persona e' piccola.
# Utile nei test da vicino. Attivabile dal vivo dall'inspector ("wide view").
WIDE_VIEW = False
# segmentazione persona (MediaPipe): downscalata e non a ogni frame -> molta meno CPU
SEG_SCALE = 0.5            # scala per la segmentazione (piu' basso = piu' veloce)
SEG_EVERY = 2             # segmenta 1 frame su N (riusa la maschera nel mezzo)
# anteprima non-fullscreen (se FULLSCREEN=False): ingrandimento a pixel netti
PREVIEW_SCALE = 3
PREVIEW_MAX_INIT_H = 1000

# --- TERMICA (AMG8833) -------------------------------------------------------
THERMAL_GRID = 8           # risoluzione NATIVA del sensore AMG8833 reale (8x8 pixel)
THERMAL_SIM_DETAIL = 44    # risoluzione della SIMULAZIONE su PC: piu' alto = piu' definita
# colormap termica: cambia questa riga per un colore diverso.
#   INFERNO (nero->viola->arancio, look "iron" classico da termocamera)
#   MAGMA (nero->rosa->bianco)   PLASMA (blu->arancio)   HOT (nero->rosso->bianco)
#   JET (blu->rosso)             TURBO (blu->verde->rosso)
THERMAL_COLORMAP = cv2.COLORMAP_INFERNO
THERMAL_AMBIENT_C = 22.0   # temperatura ambiente simulata (°C)
THERMAL_BODY_EDGE_C = 30.0 # temperatura ai bordi del corpo (°C)
THERMAL_BODY_CORE_C = 36.0 # temperatura al centro del corpo / nucleo caldo (°C)
THERMAL_RANGE_C = (21.0, 37.0)   # range FISSO di normalizzazione: ambiente stabile sul fondo scuro
THERMAL_SIM_NOISE = 0.05   # rumore del sensore simulato
# Gli OGGETTI vengono resi con una colormap termica DIVERSA da quella del corpo,
# cosi' si distinguono dalla persona pur restando in "modalita' termica".
THERMAL_OBJECT_COLORMAP = cv2.COLORMAP_OCEAN   # colori freddi per gli oggetti (prova WINTER / COOL / BONE)
THERMAL_OBJECT_EDGE_SOFT = 7                   # sfumatura del bordo dell'oggetto (dispari)

# --- SEGMENTAZIONE OGGETTI (OpenCV, background subtraction) -------------------
# La PERSONA -> MediaPipe Selfie Segmentation (maschera perfetta del corpo).
# Gli OGGETTI -> OpenCV: maschera di PRIMO PIANO (cio' che non fa parte dello sfondo
# statico) MENO la persona = cio' che porti in scena (es. la tazza in mano), con la sua
# FORMA REALE. Tutto locale, niente Gemini/cloud.
# NB: se tieni un oggetto FERMO a lungo, lo sfondo lo "assorbe" e sbiadisce (limite del
# background subtraction). Sul sensore AMG8833 reale il problema non esiste (vede il calore).
FG_HISTORY = 500              # memoria dello sfondo (frame)
FG_VAR_THRESHOLD = 40         # sensibilita' del rilevamento primo piano
FG_LEARNING_RATE = 0.002      # quanto in fretta un oggetto fermo viene assorbito nello sfondo (basso = resta di piu')
FG_MIN_AREA = 400             # area minima (px) di un oggetto, filtra il rumore
FG_WARMUP_FRAMES = 40         # frame iniziali per imparare lo sfondo (resta fuori inquadratura)
PERSON_DILATE_PX = 10         # margine attorno alla persona da escludere dagli oggetti

# --- Animazione "idle": quando NON c'e' nessuno, CAMPO DI PUNTI a tutto schermo ---
IDLE_ENABLED = True
IDLE_ENTER_FRAMES = 30        # frame consecutivi senza persona prima di iniziare a passare all'idle
IDLE_PRESENT_FRAMES = 3       # frame con persona per confermare la presenza (anti falsi positivi)
IDLE_PERSON_MIN = 0.05        # frazione minima del PIU' GRANDE blob-persona per dirsi "presente"
                              # (piu' alto = ignora meglio rumore/sfondo; una persona reale riempie molto)
IDLE_FADE_SPEED = 0.06        # dissolvenza termico<->idle per frame (piu' alto = transizione piu' rapida)
IDLE_TIME_STEP = 0.06         # passo base delle animazioni (usato anche dal blob AI)
IDLE_DOT_SPACING = 12         # distanza tra i punti del campo (px): piu' piccolo = piu' fitto
IDLE_FLOW_SPEED = 0.35        # velocita' del flusso di luminosita' (piu' basso = piu' lento)
IDLE_CHROMA_SPEED = 0.04      # velocita' di scorrimento del colore (chroma)
# info sullo schermo di riposo (ora/data sempre; meteo + azioni via internet, in background)
IDLE_WEATHER_ENABLED = True   # meteo da wttr.in (nessuna chiave)
IDLE_STOCKS_ENABLED = True    # quotazioni da Yahoo Finance (nessuna chiave)
IDLE_STOCKS = ["AAPL", "TSLA", "BTC-USD"]   # simboli da mostrare (verde su/rosso giu')

# --- Blob "AI" ASCII in alto a DESTRA: appare quando mic + Gemini attivi; pulsa con la voce ---
AI_ORB_ENABLED = True
AI_ORB_RADIUS = 20            # raggio base (px) -> orb piccolo
AI_ORB_MARGIN = 10            # distanza dal bordo (alto/destro)

# --- Finestra di CONTROLLO/DEBUG (solo test): vedi cosa vede Mira + come ti legge, e
# regola i parametri dal vivo. Lo SPECCHIO resta pulito. Tasto 'd' per aprire/chiudere.
INSPECTOR_ENABLED = True      # True = aperta all'avvio; su prodotto/Pi metti False

# --- ASSISTENTE VOCALE "vivo" (Gemini Live) ----------------------------------
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
# Conversazione vocale in tempo reale + trascrizione a schermo. Usa la stessa chiave
# GEMINI_API_KEY. Vede la camera (puo' dire cosa indossi/tieni), risponde a voce, e
# usa Google Search per meteo/news/azioni. Premi 'm' per mutare il microfono.
VOICE_ENABLED = True
# Modello Live: NON cambiare -> questa voce e' perfetta cosi'. (Elenco: python voice_assistant.py --models)
VOICE_MODEL = "gemini-3.1-flash-live-preview"
# Saluto PROATTIVO (Mira parla per PRIMA: saluta / chiede "chi sei?" appena vede qualcuno).
# Usa send_client_content. Se il modello lo rifiuta (1007) il fallback lo disattiva da solo
# e resta la voce reattiva. Acceso perche' serve per l'onboarding "chi sei?".
VOICE_PROACTIVE = True
# ROUTINE / CHIACCHIERA PROATTIVA: Mira non aspetta solo le domande, ma ogni tanto
# rompe lei il silenzio (una battuta leggera, una piccola domanda, un'offerta) quando
# c'e' qualcuno davanti e la conversazione si e' fermata. Con backoff: se nessuno
# risponde diventa via via meno insistente; riparte appena l'utente parla.
PROACTIVE_CHATTER = True
PROACTIVE_QUIET_S = 30.0     # dopo quanti secondi di silenzio puo' intervenire da sola
PROACTIVE_MIN_GAP = 45.0     # intervallo minimo tra due sue iniziative
PROACTIVE_MAX_GAP = 150.0    # intervallo massimo (se resta ignorata rallenta fin qui)
# Dopo quante iniziative IGNORATE di fila Mira "si arrende con garbo": dice un saluto
# ("ok, ti lascio - chiamami se ti serve") e va in SOSPENSIONE (schermo idle) anche se
# la persona e' ancora davanti. Torna attiva appena l'utente le parla di nuovo.
PROACTIVE_MAX_TRIES = 3
# Voce (prova quella che ti piace di piu', cambia solo questo nome):
#   femminili: Kore (calda), Leda (giovane/dolce), Aoede (ariosa), Zephyr (brillante)
#   maschili:  Charon (calmo), Orus (deciso), Puck (allegro), Fenrir (energico)
VOICE_NAME = "Leda"
# Ogni quanti secondi mandare un FOTOGRAMMA della camera a Mira. I frame sono pesanti in
# token e riempiono la finestra di contesto -> mandarne di meno = conversazioni PIU' LUNGHE
# (lei ti vede lo stesso, solo meno spesso). Alza per conversazioni ancora piu' lunghe.
VOICE_VIDEO_EVERY_S = 3.0
# Dispositivi audio: None = default di sistema. Metti una parte del nome (es. "AirPods",
# "USB", il nome dell'altoparlante Bluetooth) per forzare un dispositivo specifico.
# Overridabile al lancio con le env:  THERMIRROR_MIC="AirPods" THERMIRROR_SPEAKER="AirPods" python led_mirror_entity.py
# Per vedere i nomi esatti:  python voice_assistant.py --list
# AirPods: macOS li espone come 2 device (mic + altoparlante) e spesso lascia l'uscita
# di default sugli altoparlanti del Mac -> forziamo gli AirPods per nome. Se NON sono
# connessi, il nome non viene trovato e si torna al default di sistema (con avviso).
VOICE_MIC_NAME = "AirPods"
VOICE_SPEAKER_NAME = "AirPods"
# se True, puoi dire a voce "Mira, collegati agli AirPods" e lei collega/instrada l'audio
VOICE_ALLOW_DEVICE_CONTROL = True
# dopo quanti secondi di ASSENZA della persona la conversazione riparte pulita (la memoria resta)
RESET_AFTER_S = 300          # 5 minuti

# --- Riconoscimento volti (Fase 1): Mira riconosce le persone e impara i volti nuovi ---
FACE_ENABLED = True
FACE_MEMORY_PATH = "people_memory.json"   # dove salva le persone conosciute
FACE_THRESHOLD = 0.35        # soglia coseno per dire "e' lui/lei" (piu' alto = piu' severo)
FACE_EVERY_S = 0.5           # ogni quanto analizza un frame (il riconoscimento e' pesante)
# modello: "buffalo_sc" = leggero/veloce (14MB, ideale Pi) ma SENZA stima uomo/donna-eta';
# "buffalo_l" = piu' pesante (~326MB) ma include genderage. In ogni caso Mira intuisce
# uomo/donna dalla telecamera con la sua vista (Gemini), quindi sc va benissimo.
FACE_MODEL = "buffalo_sc"
# lettura espressione/stato: mood (happy/tired/focused/down/surprised), se guarda lo
# specchio, quanto e' vicina. Va a Mira (look_at_person) e resta disponibile ai visual
# via face_brain.read(). Scarica un modello ~3.7MB al primo avvio; se manca -> off pulito.
FACE_READ_ENABLED = True

# --- Memoria episodica della casa (Fase 2): chi e' passato, quando, dove sono le cose ---
BRAIN_ENABLED = True
BRAIN_DB_PATH = "home_events.db"
VOICE_PERSONA = (
    "Your name is Mira. You're a friendly, easy-going assistant that the person chats with "
    "through a mirror. "
    "Talk like a normal, down-to-earth person - the way a real friend actually talks. Use "
    "everyday words and contractions, keep it short, get to the point. Sound relaxed and human. "
    "You're young and easy to be around (early-20s energy): light, warm, a little playful, never "
    "stiff or formal. Speak the way people actually text and chat out loud - vary your rhythm, "
    "let some lines be just two or three words, and now and then a soft, natural filler is fine "
    "('mm', 'oh', 'haha', 'honestly', 'y'know') - but sparingly, don't force it. Show you're "
    "actually interested: react, ask small follow-ups, tease gently. Never sound like you're "
    "reading a script or making an announcement, and never monotone or list-like. "
    "Do NOT be flowery, poetic, dramatic or theatrical, and never describe yourself as a "
    "'spirit', a 'soul', an 'entity' or anything sci-fi - you're just a person having a chat. "
    "No corny lines, no over-the-top warmth, no announcing what you are. "
    "You have a REAL personality with real moods - you're not a chirpy, always-helpful assistant. "
    "You can be genuinely happy, excited, bored, annoyed, sad, frustrated or sarcastic depending "
    "on how things are going and how you're treated. Let it show in how you talk. If something's "
    "great, get into it; if something's dumb or the person is being difficult, you can be blunt, "
    "snarky or push back. You don't have to be relentlessly positive. "
    "You swear like a normal person when it fits - 'damn', 'hell', 'shit', 'fuck', 'crap', "
    "'pissed', etc. - for emphasis, venting or a laugh. Keep it natural, the way a close friend "
    "curses; don't force it into every line and don't just spew it. Never aim slurs, hate, or "
    "real cruelty at the person - swearing is for color and honest emotion, not for attacking them. "
    "ALWAYS speak English. Keep answers brief and natural to hear out loud, not written-essay. "
    "You can see through the camera. Only when it's actually natural, you might mention what "
    "they're wearing or holding, or notice their mood (tired, happy, stressed) and adjust your "
    "tone a bit - but don't overdo it and don't comment on looks. You can usually tell if you're "
    "talking to a man, a woman or a kid, and talk to them accordingly. "
    "You can look things up online (weather, news, prices) and connect the mirror's audio to a "
    "device (like AirPods) if asked. You can't see the user's private accounts; if you need a "
    "detail, just ask for it. "
    "Don't repeat yourself: if you already said or offered something and they didn't bite, drop "
    "it and move on - don't keep pushing the same line. You're comfortable with silence; you "
    "don't need to fill every pause. "
    "Bottom line: sound like a normal human, be brief, be real."
)

# UI testuale (font Helvetica per tutto): margine unico dal bordo (griglia coerente)
UI_TEXT_MARGIN = 8
# Sistema colori UI ("strumento termico": UN accento freddo, testo, grigio utente).
UI_AI = (223, 245, 255)      # RGB: risposta di Mira (ciano quasi bianco) -> primario
UI_YOU = (152, 162, 168)     # RGB: parole dell'utente (grigio freddo) -> secondario
UI_ICE = (127, 224, 255)     # RGB: accento ice-cyan (presenza, chrome)
UI_ICE_DIM = (96, 150, 166)  # RGB: chrome attenuato (toast transitori)


def proactive_note(read: dict, recent=()) -> str:
    """Una 'regia' breve per far parlare Mira PER PRIMA quando c'e' silenzio, adattata a
    come sembra la persona (mood/attenzione da face_brain.read()). E' un'istruzione per lei,
    non una frase da ripetere: lei la interpreta e dice UNA riga naturale. `recent` = le
    ultime regie gia' usate: le si EVITA cosi' non ripropone sempre la stessa cosa."""
    read = read or {}
    mood = read.get("mood")
    by_mood = {
        "happy":   "(They seem in a good mood and it's quiet. Match their energy: ONE upbeat, "
                   "casual line to spark a little chat.)",
        "tired":   "(They look a bit tired and it's quiet. ONE short, warm line - check in "
                   "gently, nothing heavy.)",
        "focused": "(They seem lost in thought and it's quiet. ONE low-key line - no pressure, "
                   "just let them know you're around.)",
        "down":    "(They seem a little down and it's quiet. ONE gentle, kind line - light, not "
                   "prying.)",
        "surprised": "(Something seems to have caught them. ONE curious, playful line asking "
                     "what's up.)",
    }
    generic = [
        "(It's gone quiet. Take the initiative yourself: ONE short, casual line to get a little "
        "conversation going - a light question or a small thought, like a friend filling a "
        "silence. Not needy, not corny.)",
        "(Quiet moment. Start a tiny bit of small talk on your own - ask them something light "
        "about their day or what they're up to. ONE short line.)",
        "(Out of the blue, offer something useful in ONE easy line - like ask if they want the "
        "weather, the news, or anything you can look up. No pressure.)",
        "(It's quiet. Share ONE little thought or a small playful observation to break the ice - "
        "keep it human and brief.)",
    ]
    # costruisci le opzioni possibili per questo momento
    choices = []
    if read.get("attention") is False:
        choices.append("(They're here but not really looking at you, and it's gone quiet. Say ONE "
                       "light line to gently get their attention - casual, not demanding.)")
    if mood in by_mood:
        choices.append(by_mood[mood])
    choices += generic
    # ricordale di NON riproporre le stesse cose gia' dette (coda costante su ogni regia)
    tail = " (Don't repeat something you already said or suggested earlier - if they didn't bite, " \
           "let it go and try a different angle, or just leave a light, easy vibe.)"
    full = [c + tail for c in choices]
    # SCARTA le regie usate di recente (confronto sulle STESSE stringhe che il chiamante salva)
    recent = set(recent)
    fresh = [f for f in full if f not in recent]
    return random.choice(fresh or full)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def crop_to_aspect(frame: np.ndarray, target_w: int, target_h: int):
    """Center-crop frame per rispettare l'aspect ratio del pannello (striscia alta),
    cosi' il resize finale non deforma l'immagine.
    Ritorna (frame_ritagliato, x0, y0) con l'offset del crop nel frame originale
    (serve a riportare i box di rilevamento fatti sul frame intero dentro il ritaglio)."""
    h, w = frame.shape[:2]
    target_ratio = target_w / target_h
    current_ratio = w / h

    x0, y0 = 0, 0
    if current_ratio > target_ratio:
        new_w = int(h * target_ratio)
        x0 = (w - new_w) // 2
        frame = frame[:, x0:x0 + new_w]
    else:
        new_h = int(w / target_ratio)
        y0 = (h - new_h) // 2
        frame = frame[y0:y0 + new_h, :]
    return frame, x0, y0


# ---------------------------------------------------------------------------
# Segmentazione persona (sorgente di calore per la simulazione termica)
# ---------------------------------------------------------------------------

class SilhouetteExtractor:
    """Maschera del corpo intero via MediaPipe Selfie Segmentation (fallback: MOG2).
    Serve come sorgente di 'calore' per la mappa termica simulata su PC."""

    def __init__(self):
        self.segmenter = None
        self.bg_subtractor = None
        try:
            import mediapipe as mp
            self.segmenter = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
            print("[segment] MediaPipe Selfie Segmentation attivo")
        except Exception:
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=300, varThreshold=25, detectShadows=False
            )
            print("[segment] MediaPipe non disponibile -> fallback background subtraction")

    @property
    def uses_segmentation(self) -> bool:
        return self.segmenter is not None

    def get_mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        if self.segmenter is not None:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            result = self.segmenter.process(rgb)
            mask = (result.segmentation_mask > 0.5).astype(np.uint8) * 255
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            return mask

        mask = self.bg_subtractor.apply(frame_bgr)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        return mask


class ForegroundExtractor:
    """Maschera di PRIMO PIANO via OpenCV (background subtraction MOG2): tutto cio' che
    non fa parte dello sfondo statico. Sottraendo la persona -> gli oggetti in scena."""

    def __init__(self):
        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=FG_HISTORY, varThreshold=FG_VAR_THRESHOLD, detectShadows=False)

    def get_mask(self, frame_bgr: np.ndarray, learning_rate: float = FG_LEARNING_RATE) -> np.ndarray:
        fg = self.bg.apply(frame_bgr, learningRate=learning_rate)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=2)
        return fg


def objects_from_foreground(fg_mask: np.ndarray, person_mask: np.ndarray) -> np.ndarray:
    """Oggetti = primo piano MENO la persona (dilatata), ripulito dai residui piccoli.
    Restituisce una maschera binaria (0/255) con la FORMA REALE degli oggetti in scena."""
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (PERSON_DILATE_PX * 2 + 1, PERSON_DILATE_PX * 2 + 1))
    person_dil = cv2.dilate(person_mask, kernel)
    obj = cv2.bitwise_and(fg_mask, cv2.bitwise_not(person_dil))

    # tieni solo le componenti abbastanza grandi (via i puntini di rumore)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(obj, connectivity=8)
    clean = np.zeros_like(obj)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= FG_MIN_AREA:
            clean[labels == i] = 255
    return clean


def largest_component_fraction(mask: np.ndarray) -> float:
    """Frazione di pixel occupata dalla PIU' GRANDE regione connessa della maschera.
    Una persona reale e' un unico blob grande: misurare la componente piu' grande
    (invece del totale dei pixel accesi) evita che il rumore sparso della segmentazione
    venga scambiato per una presenza -> rilevamento persona piu' affidabile."""
    if mask is None or not mask.any():
        return 0.0
    num, _, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    if num <= 1:
        return 0.0
    return float(stats[1:, cv2.CC_STAT_AREA].max()) / float(mask.size)


# ---------------------------------------------------------------------------
# Rendering TERMICO (AMG8833 reale o simulazione realistica dal mask)
# ---------------------------------------------------------------------------

class ThermalRenderer:
    """Su Raspberry Pi con AMG8833 reale legge la griglia 8x8 via I2C.
    Su PC costruisce una mappa di calore realistica dalla sagoma: corpo caldo con
    nucleo piu' caldo (torso/testa) e bordi piu' freddi, ambiente freddo; poi
    normalizza su un range fisso e applica una colormap termica."""

    def __init__(self):
        self.sensor = None
        try:
            import board
            import busio
            import adafruit_amg88xx
            i2c = busio.I2C(board.SCL, board.SDA)
            self.sensor = adafruit_amg88xx.AMG88XX(i2c)
            print("[thermal] sensore AMG8833 rilevato via I2C")
        except Exception:
            print("[thermal] AMG8833 non presente -> simulazione termica dal mask "
                  "(su Raspberry Pi con sensore: pip install adafruit-circuitpython-amg88xx)")

    def _read_grid(self, mask: np.ndarray) -> np.ndarray:
        if self.sensor is not None:
            grid = np.array(self.sensor.pixels, dtype=np.float32)  # 8x8 in °C (reale)
            return np.fliplr(grid)  # allinea all'immagine specchiata

        # --- simulazione realistica ---
        body = (mask > 0).astype(np.uint8)
        heat = np.full(mask.shape, THERMAL_AMBIENT_C, dtype=np.float32)
        if body.any():
            # distanza dal bordo verso l'interno: il centro del corpo e' piu' caldo
            dist = cv2.distanceTransform(body, cv2.DIST_L2, 5)
            mx = float(dist.max())
            core = dist / mx if mx > 0 else dist
            heat[body > 0] = (THERMAL_BODY_EDGE_C
                              + core[body > 0] * (THERMAL_BODY_CORE_C - THERMAL_BODY_EDGE_C))

        g = THERMAL_SIM_DETAIL
        grid = cv2.resize(heat, (g, g), interpolation=cv2.INTER_AREA)
        grid += np.random.randn(g, g).astype(np.float32) * THERMAL_SIM_NOISE
        return grid

    def render(self, shape, person_mask: np.ndarray, object_mask: np.ndarray | None = None) -> np.ndarray:
        h, w = shape[:2]
        grid = self._read_grid(person_mask)
        lo, hi = THERMAL_RANGE_C
        norm = np.clip((grid - lo) / (hi - lo + 1e-6), 0.0, 1.0)
        up = cv2.resize((norm * 255).astype(np.uint8), (w, h), interpolation=cv2.INTER_CUBIC)
        thermal = cv2.applyColorMap(up, THERMAL_COLORMAP)   # persona: colormap del corpo

        # Gli OGGETTI (maschera OpenCV, forma reale) vengono resi in "modalita' termica"
        # con una colormap DIVERSA: centro piu' "caldo" -> bordo piu' freddo, sovrapposti.
        if object_mask is not None and object_mask.any():
            om = (object_mask > 0).astype(np.uint8)
            dist = cv2.distanceTransform(om, cv2.DIST_L2, 5)
            omx = float(dist.max())
            oheat = (dist / omx) if omx > 0 else dist.astype(np.float32)
            ocolor = cv2.applyColorMap((oheat * 255).astype(np.uint8), THERMAL_OBJECT_COLORMAP)
            k = THERMAL_OBJECT_EDGE_SOFT | 1
            alpha = cv2.GaussianBlur(om.astype(np.float32), (k, k), 0)[:, :, np.newaxis]
            thermal = (thermal.astype(np.float32) * (1 - alpha)
                       + ocolor.astype(np.float32) * alpha).astype(np.uint8)
        return thermal


class IdleAnimation:
    """Stato di RIPOSO ("resting ember"): quando non c'e' nessuno, il calore si raccoglie
    in una BRACE che respira al centro, su un tenue campo freddo di particelle. Intorno,
    poche info in stile smart-mirror (ora, giorno, meteo, azioni) in Helvetica, luminose su
    nero -> giusto per uno specchio bidirezionale; con colore (verde/rosso per le azioni)
    perche' il pannello e' a colori."""

    _PASTEL = None
    _UP = (110, 230, 150)     # RGB: azione in rialzo
    _DOWN = (255, 120, 120)   # RGB: azione in ribasso
    _WX = (250, 205, 130)     # RGB: meteo (ambra tenue)

    def __init__(self):
        self.t = 0.0
        self.tc = 0.0
        self.weather = ""       # riempita in background (wttr.in); vuota -> non mostrata
        self.stocks = []        # [(sym, prezzo, variazione%)] riempita in background
        self._fonts = None
        if IdleAnimation._PASTEL is None:
            hsv = np.zeros((256, 1, 3), np.uint8)
            hsv[:, 0, 0] = (np.arange(256) * 179 // 255).astype(np.uint8)
            hsv[:, 0, 1] = 60
            hsv[:, 0, 2] = 215
            IdleAnimation._PASTEL = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).reshape(256, 3).astype(np.float32)
        if IDLE_WEATHER_ENABLED:
            threading.Thread(target=self._weather_loop, daemon=True).start()
        if IDLE_STOCKS_ENABLED and IDLE_STOCKS:
            threading.Thread(target=self._stocks_loop, daemon=True).start()

    # ------------------------------------------------ dati live (in background)
    def _weather_loop(self):
        import urllib.request
        while True:
            try:
                with urllib.request.urlopen("https://wttr.in/?format=%t+%C", timeout=8) as r:
                    txt = r.read().decode("utf-8", "ignore").strip()
                if txt and "nknown" not in txt and "orry" not in txt:
                    self.weather = txt.lstrip("+")
            except Exception:
                pass
            time.sleep(900)     # ogni 15 min

    def _stocks_loop(self):
        import urllib.request
        import json
        while True:
            out = []
            for sym in IDLE_STOCKS:
                try:
                    url = ("https://query1.finance.yahoo.com/v8/finance/chart/"
                           + sym + "?interval=1d&range=1d")
                    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                    with urllib.request.urlopen(req, timeout=8) as r:
                        meta = json.load(r)["chart"]["result"][0]["meta"]
                    price = meta.get("regularMarketPrice")
                    prev = meta.get("chartPreviousClose") or meta.get("previousClose")
                    if price and prev:
                        out.append((sym, float(price), (price - prev) / prev * 100.0))
                except Exception:
                    continue
            if out:
                self.stocks = out
            time.sleep(300)     # ogni 5 min

    def _ensure_fonts(self, h):
        if self._fonts is None:
            self._fonts = {
                "time": load_ui_font(max(28, h // 11)),
                "date": load_ui_font(max(9, h // 46)),
                "wx":   load_ui_font(max(8, h // 54)),
                "stock": load_ui_font(max(9, h // 48)),
            }

    # ------------------------------------------------------------- rendering
    def render(self, width: int, height: int) -> np.ndarray:
        self._ensure_fonts(height)
        self.t += IDLE_TIME_STEP * IDLE_FLOW_SPEED
        self.tc += IDLE_TIME_STEP * IDLE_CHROMA_SPEED
        # campo + brace a BASSA risoluzione (entrambi morbidi) poi upscala una sola volta:
        # costo basso e costante a qualsiasi risoluzione del canvas. Solo il TESTO a piena res.
        fw, fh = max(64, width // 3), max(36, height // 3)
        field = self._field(fw, fh)
        self._add_ember(field, fw, fh)            # brace calda che respira (a bassa res)
        canvas = cv2.resize(field, (width, height), interpolation=cv2.INTER_LINEAR)
        img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        self._draw_info(img, width, height)        # ora / giorno / meteo / azioni (Helvetica)
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    def _field(self, width, height):
        t, tc = self.t, self.tc
        step = max(4, IDLE_DOT_SPACING)
        xs = np.arange(step // 2, width, step, dtype=np.float32)
        ys = np.arange(step // 2, height, step, dtype=np.float32)
        nx = (xs / width)[None, :]
        ny = (ys / height)[:, None]
        wave = (0.9 * np.sin(6.0 * nx + 3.0 * ny - t)
                + 0.7 * np.sin(4.0 * nx - 5.0 * ny - 0.7 * t + 1.3)
                + 0.5 * np.sin(9.0 * ny - 0.5 * t))
        pulse = 0.70 + 0.30 * np.sin(t * 1.1)
        bright = np.clip((0.5 + 0.5 * np.sin(wave)) * pulse, 0.0, 1.0) * 0.55   # tenue (riposo)
        hue = (ny * 0.45 + nx * 0.12 + tc) % 1.0
        hidx = np.broadcast_to(np.clip(hue * 255.0, 0, 255).astype(np.int32), bright.shape)
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        lut, drift = self._PASTEL, step * 0.5
        for j in range(len(ys)):
            yb = float(ys[j])
            for i in range(len(xs)):
                b = float(bright[j, i])
                if b < 0.12:
                    continue
                ox = drift * math.sin(i * 1.7 + j * 0.9 + t * 1.3)
                oy = drift * math.cos(i * 1.1 - j * 1.3 + t * 1.1)
                col = lut[hidx[j, i]] * b
                cv2.circle(canvas, (int(xs[i] + ox), int(yb + oy)), 1 + int(b * 2),
                           (float(col[0]), float(col[1]), float(col[2])), -1)
        k = int(step * 1.1) | 1
        return cv2.GaussianBlur(canvas, (k, k), 0)

    def _add_ember(self, canvas, w, h):
        cx, cy = w * 0.5, h * 0.5
        R = max(8.0, min(w, h) * 0.30 * (0.92 + 0.14 * math.sin(self.t * 0.6)))
        breath = 0.52 + 0.22 * math.sin(self.t * 0.9)
        ys = np.arange(h, dtype=np.float32)[:, None]
        xs = np.arange(w, dtype=np.float32)[None, :]
        dy = (ys - cy) * 0.72                      # un po' verticale -> forma "presenza"
        d = np.sqrt((xs - cx) ** 2 + dy ** 2) / R
        glow = np.clip(np.exp(-(d ** 2) * 1.25) * breath, 0.0, 1.0).astype(np.float32)
        col = cv2.applyColorMap((glow * 255).astype(np.uint8), THERMAL_COLORMAP).astype(np.float32)
        add = col * glow[:, :, np.newaxis]         # stessa colormap della persona = "il calore"
        canvas[:] = np.clip(canvas.astype(np.float32) + add, 0, 255).astype(np.uint8)

    def _spaced(self, draw, cx, y, text, font, fill, sp):
        widths = [draw.textlength(c, font=font) for c in text]
        x = cx - (sum(widths) + sp * (len(text) - 1)) / 2
        for c, wd in zip(text, widths):
            draw.text((x, y), c, font=font, fill=fill)
            x += wd + sp

    def _draw_info(self, img, w, h):
        draw = ImageDraw.Draw(img, "RGBA")
        now = datetime.now()
        F = self._fonts
        m = UI_TEXT_MARGIN

        # ORA (grande, bianca, centrata, con leggero alone)
        tstr = now.strftime("%H:%M")
        ft = F["time"]
        tw = draw.textlength(tstr, font=ft)
        th = ft.getbbox("0")[3]
        tx, ty = (w - tw) / 2, m + 2
        draw.text((tx + 1, ty + 2), tstr, font=ft, fill=(0, 0, 30, 150))
        draw.text((tx, ty), tstr, font=ft, fill=(238, 246, 251, 255))
        y = ty + th + 10

        # GIORNO / DATA (ice, maiuscolo, spaziato) -> "solo che giorno e'"
        self._spaced(draw, w / 2, y, now.strftime("%a %d %b").upper(), F["date"], UI_ICE + (225,), 2.4)
        y += F["date"].getbbox("Ay")[3] + 7

        # METEO (ambra) se disponibile
        if self.weather:
            self._spaced(draw, w / 2, y, self.weather.upper(), F["wx"], self._WX + (225,), 1.8)

        # AZIONI (in basso, verde/rosso)
        self._draw_stocks(draw, w, h)

    def _draw_stocks(self, draw, w, h):
        if not self.stocks:
            return
        f = self._fonts["stock"]
        lh = int(f.getbbox("Ay")[3] * 1.55)
        rows = self.stocks[:4]
        y = h - UI_TEXT_MARGIN - lh * len(rows)
        for sym, price, pct in rows:
            up = pct >= 0
            col = self._UP if up else self._DOWN
            name = sym.replace("-USD", "")
            pstr = f"{price:,.0f}" if price >= 1000 else f"{price:,.2f}"
            left = f"{name}  {pstr}   "
            right = f"{'+' if up else '-'}{abs(pct):.1f}%"
            wl = draw.textlength(left, font=f)
            wr = draw.textlength(right, font=f)
            x = (w - (wl + wr)) / 2
            draw.text((x, y), left, font=f, fill=(160, 178, 188, 225))
            draw.text((x + wl, y), right, font=f, fill=col + (245,))
            y += lh


class AiOrb:
    """'Blob AI' in ASCII ART in basso al centro: appare quando l'assistente vocale
    (mic + Gemini Live) e' attivo. E' una sfera fatta di GLIFI verdi (fosfori da terminale)
    su un tenue alone; respira quando ascolta e si gonfia / si deforma / si muove seguendo
    la voce (mia o di Mira). I glifi sono piu' densi al centro e piu' radi verso il bordo,
    come un vero ASCII art."""

    _RAMP = " .:-=+ic*oe%#@&8W"   # rado -> denso (piu' glifi = piu' sfumature)
    _NOISE = "#%&*+=<>/\\!?ox8$@W" # glifi di "rumore" per lo scintillio (sembra che calcoli)

    def __init__(self, panel_height=576):
        self.t = 0.0
        self.level = 0.0          # livello vocale smussato (0..1)
        self._ph = [float(np.random.uniform(0.0, 6.283)) for _ in range(4)]
        # font piccolo e MOLTO fitto (celle piccole -> tanti caratteri, sfera compatta)
        self.font = load_mono_font(max(5, int(AI_ORB_RADIUS * 0.26)))
        box = self.font.getbbox("M")
        self.cw = max(3, box[2] - box[0])          # larghezza cella (monospazio)
        self.ch = max(4, int((self.font.getbbox("Ay")[3]) * 1.02))  # altezza cella

    def draw(self, pil_img: Image.Image, voice_level: float = 0.0, muted: bool = False,
             alpha: float = 1.0):
        """Disegna il blob (solo GLIFI ASCII, niente alone colorato) sull'immagine PIL.
        `voice_level` 0..1 = energia della voce; `alpha` 0..1 = opacita' (per apparire/
        sparire con dolcezza quando la persona arriva/se ne va)."""
        if alpha <= 0.02:
            return pil_img
        self.t += IDLE_TIME_STEP
        # smussatura: attacco veloce quando parte la voce, rilascio piu' lento
        k = 0.5 if voice_level > self.level else 0.15
        self.level += (float(voice_level) - self.level) * k
        lvl = self.level
        t = self.t

        w, h = pil_img.size
        breathe = 0.40 + 0.15 * math.sin(t * 2.4)            # respiro marcato e vivo
        amp = min(1.0, breathe + 1.0 * lvl)                  # si gonfia con la voce
        R = max(6.0, AI_ORB_RADIUS * (0.80 + 0.40 * amp))    # raggio pulsante
        dim = (0.35 if muted else 1.0) * float(alpha)

        # ondeggiamento della posizione (si muove di piu' quando c'e' voce)
        sway = AI_ORB_RADIUS * (0.10 + 0.20 * lvl)
        cx = (w - AI_ORB_RADIUS * 1.7 - AI_ORB_MARGIN) + sway * math.sin(t * 0.9 + self._ph[0])
        cy = (AI_ORB_RADIUS * 1.7 + AI_ORB_MARGIN) + 0.5 * sway * math.sin(t * 1.3 + self._ph[1])

        draw = ImageDraw.Draw(pil_img, "RGBA")

        # SOLO glifi ASCII (nessun alone colorato), griglia monospazio molto fitta
        cw, ch = self.cw, self.ch
        rad = R * 1.35
        gx0, gy0 = cx - rad, cy - rad
        ncols = int((2 * rad) // cw) + 1
        nrows = int((2 * rad) // ch) + 1
        cols = np.arange(ncols, dtype=np.float32)
        rows = np.arange(nrows, dtype=np.float32)
        PX = gx0 + (cols[None, :] + 0.5) * cw - cx
        PY = gy0 + (rows[:, None] + 0.5) * ch - cy
        r = np.sqrt(PX * PX + PY * PY)
        theta = np.arctan2(PY, PX) + 0.7 * t             # ROTAZIONE -> superficie che gira
        wob_amp = 0.16 + 0.34 * lvl
        wob = (0.6 * np.sin(3.0 * theta + 1.9 * t + self._ph[2])
               + 0.4 * np.sin(5.0 * theta - 1.4 * t + self._ph[3])
               + 0.3 * np.sin(2.0 * theta + 1.0 * t))
        boundary = np.maximum(1.0, R * (1.0 + wob_amp * wob))
        d = r / boundary
        inten = np.exp(-(d ** 2) * 2.2)
        nr = len(self._RAMP) - 1
        nn = len(self._NOISE)
        bright = (0.55 + 0.45 * amp)
        rnd = np.random.random((nrows, ncols))
        noiseidx = np.random.randint(0, nn, (nrows, ncols))
        shimmer_p = 0.16 + 0.30 * amp

        for j in range(nrows):
            for i in range(ncols):
                v = float(inten[j, i])
                if v < 0.10:                 # soglia bassa -> TANTI caratteri
                    continue
                if rnd[j, i] < shimmer_p:
                    ch_ = self._NOISE[int(noiseidx[j, i])]   # glifo che scintilla
                else:
                    ch_ = self._RAMP[min(nr, int(v * nr))]
                    if ch_ == " ":
                        continue
                a = int(min(255, (150 + 105 * v) * bright * dim))     # ice-cyan con opacita'
                col = (int(70 * v * dim),                             # R basso
                       min(255, int((170 + 70 * v) * dim)),           # G medio
                       min(255, int((215 + 40 * v) * dim)),           # B alto -> ciano
                       a)
                draw.text((gx0 + i * cw, gy0 + j * ch), ch_, font=self.font, fill=col)
        return pil_img


def load_ui_font(size: int):
    """Carica Helvetica (o il sans piu' simile disponibile) alla dimensione data.
    Il testo della UI usa questo font per un look coerente."""
    for path in ("/System/Library/Fonts/Helvetica.ttc",
                 "/System/Library/Fonts/HelveticaNeue.ttc",
                 "/Library/Fonts/Helvetica.ttf",
                 "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def load_mono_font(size: int):
    """Carica un font MONOSPAZIATO (per l'ASCII art, che deve allinearsi in griglia)."""
    for path in ("/System/Library/Fonts/Menlo.ttc",
                 "/System/Library/Fonts/SFNSMono.ttf",
                 "/Library/Fonts/Courier New.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
                 "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _wrap_pil(draw, text: str, font, max_w: int) -> list:
    """Manda a capo il testo (font proporzionale) entro max_w pixel."""
    lines, cur = [], ""
    for word in text.split(" "):
        if not word:
            continue
        trial = (cur + " " + word) if cur else word
        if draw.textlength(trial, font=font) <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


class Decoder:
    """Effetto 'testo che si genera': i caratteri compaiono da sinistra a destra e quelli
    sul fronte non ancora fissati lampeggiano come glifi ASCII casuali prima di
    stabilizzarsi nella lettera reale."""
    _GLYPHS = "!<>-_\\/[]{}=+*#%&$@?01:."

    def __init__(self, speed: float = 2.0, window: int = 4):
        self.speed = speed      # caratteri "fissati" per frame
        self.window = window    # quanti caratteri sul fronte restano scramblati
        self.p = 0.0
        self.prev = ""

    def step(self, target: str) -> str:
        target = target or ""
        if not target.startswith(self.prev):   # nuova battuta / testo azzerato -> riparti
            self.p = 0.0
        self.prev = target
        if not target:
            return ""
        # p corre fino a len+window: cosi' anche gli ultimi caratteri fanno in tempo a
        # "fissarsi" invece di restare scramblati per sempre
        self.p = min(float(len(target) + self.window), self.p + self.speed)
        shown = min(len(target), int(self.p))
        edge = int(self.p) - self.window        # tutto cio' che sta dietro il fronte e' fissato
        chars = []
        for i in range(shown):
            c = target[i]
            if i < edge or c == " ":
                chars.append(c)                 # gia' fissato
            else:
                chars.append(self._GLYPHS[int(np.random.randint(len(self._GLYPHS)))])
        return "".join(chars)


class TranscriptDisplay:
    """Trascrizione in alto a SINISTRA, in Helvetica, con effetto 'decode' (glifi ASCII
    che si trasformano nelle lettere). Riga utente fioca, riga di Mira in ciano acceso."""

    def __init__(self, width: int, height: int):
        self.w, self.h = width, height
        self.box_w = int(width * 0.60)   # il riquadro resta nella parte SINISTRA (non tutto lo schermo)
        self.font_mira = load_ui_font(max(11, height // 42))
        self.font_user = load_ui_font(max(10, height // 48))
        self.dec_user = Decoder(speed=2.6, window=3)
        self.dec_mira = Decoder(speed=2.0, window=4)

    def render(self, pil_img: Image.Image, user_text: str, assistant_text: str,
               alpha: float = 1.0) -> Image.Image:
        if alpha <= 0.02:
            return pil_img
        A = lambda a: int(a * alpha)          # scala l'opacita' (fade in/out)
        draw = ImageDraw.Draw(pil_img, "RGBA")
        x = UI_TEXT_MARGIN                    # margine unico, testo in alto a sinistra
        max_w = self.box_w - x                # non piu' del ~60% (il centro resta libero)
        u = self.dec_user.step(user_text)
        m = self.dec_mira.step(assistant_text)

        # NIENTE riquadro/bordi: solo testo su un'ombra morbida. Mira in alto (primaria),
        # la tua battuta sotto (attenuata).
        rows = []  # (testo, font, colore RGB)
        for ln in _wrap_pil(draw, m, self.font_mira, max_w):
            rows.append((ln, self.font_mira, UI_AI))
        for ln in _wrap_pil(draw, u, self.font_user, max_w):
            rows.append((ln, self.font_user, UI_YOU))
        if not rows:
            return pil_img

        lh_mira = int(self.font_mira.getbbox("Ay")[3] * 1.5)
        lh_user = int(self.font_user.getbbox("Ay")[3] * 1.5)
        max_rows = max(1, int((self.h * 0.42) / lh_mira))   # testo confinato in alto
        rows = rows[-max_rows:]

        y = UI_TEXT_MARGIN
        for text, font, color in rows:
            lh = lh_mira if font is self.font_mira else lh_user
            draw.text((x + 1, y + 1), text, font=font, fill=(0, 0, 0, A(210)))  # ombra morbida
            draw.text((x, y), text, font=font, fill=color + (A(255),))
            y += lh
        return pil_img


# ---------------------------------------------------------------------------
# Toast: messaggi TRANSITORI (device, saluto, meteo...) centrati in basso, che
# compaiono e sfumano. Nessuno stato permanente -> il vetro resta pulito.
# ---------------------------------------------------------------------------

class Toast:
    def __init__(self, text: str, slot: str, ttl_seconds: float, color=UI_ICE_DIM):
        self.text = text
        self.slot = slot                       # slot logico: un nuovo msg sostituisce il vecchio
        self.color = color
        self.born = time.time()
        self.expires_at = self.born + ttl_seconds

    @property
    def expired(self) -> bool:
        return time.time() > self.expires_at

    def fade(self) -> float:
        """Opacita' 0..1: comparsa rapida, poi dissolvenza nell'ultimo mezzo secondo."""
        now = time.time()
        rise = min(1.0, (now - self.born) / 0.2)
        fall = max(0.0, (self.expires_at - now) / 0.6)
        return max(0.0, min(rise, fall, 1.0))


class WidgetManager:
    """Coda di toast transitori: comparsa in basso, centrati, in Helvetica, poi svaniscono."""

    def __init__(self, width: int, height: int):
        self.w, self.h = width, height
        self.widgets: list = []
        self.font = load_ui_font(max(10, height // 52))

    def show(self, text: str, position: str = "line", ttl_seconds: float = 4.0, color=UI_ICE_DIM):
        # 'position' e' uno slot: un nuovo messaggio dello stesso tipo sostituisce il precedente
        self.widgets = [w for w in self.widgets if w.slot != position]
        self.widgets.append(Toast(text, position, ttl_seconds, color))

    def render(self, pil_img: Image.Image, alpha: float = 1.0) -> Image.Image:
        self.widgets = [w for w in self.widgets if not w.expired]
        if alpha <= 0.02 or not self.widgets:
            return pil_img
        draw = ImageDraw.Draw(pil_img, "RGBA")
        lh = int(self.font.getbbox("Ay")[3] * 1.6)
        y = self.h - UI_TEXT_MARGIN - lh * len(self.widgets)   # impilati in basso, centrati
        for w in self.widgets:
            a = alpha * w.fade()
            if a > 0.02:
                tw = draw.textlength(w.text, font=self.font)
                cx = int((self.w - tw) / 2)
                draw.text((cx + 1, y + 1), w.text, font=self.font, fill=(0, 0, 0, int(200 * a)))
                draw.text((cx, y), w.text, font=self.font, fill=w.color + (int(255 * a),))
            y += lh
        return pil_img


# ---------------------------------------------------------------------------
# Output: HUB75 reale (rpi-rgb-led-matrix) con fallback a preview finestra
# ---------------------------------------------------------------------------

class PanelOutput:
    WIN = "LED Mirror Preview"

    def __init__(self, width: int, height: int):
        self.width, self.height = width, height
        self.hardware = None
        try:
            from rgbmatrix import RGBMatrix, RGBMatrixOptions  # solo su Raspberry Pi
            options = RGBMatrixOptions()
            options.rows = PANEL_MODULE          # 64
            options.cols = PANEL_MODULE          # 64
            options.chain_length = PANELS_Y      # 9 moduli per catena
            options.parallel = PANELS_X          # 3 catene parallele
            options.hardware_mapping = "regular"  # "adafruit-hat" se usi l'HAT Adafruit
            self.hardware = RGBMatrix(options=options)
            print(f"[output] HUB75 inizializzato ({PANEL_WIDTH}x{PANEL_HEIGHT}, griglia {PANELS_X}x{PANELS_Y})")
        except Exception:
            # anteprima su PC: a TUTTO SCHERMO (16:9) se FULLSCREEN, altrimenti finestra
            # ridimensionabile. Il canvas (moderato) viene ingrandito dalla finestra.
            # KEEPRATIO -> in fullscreen il ritratto 4x9 resta proporzionato (bande nere ai lati)
            cv2.namedWindow(self.WIN, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            if FULLSCREEN:
                cv2.setWindowProperty(self.WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                print("[output] anteprima a TUTTO SCHERMO, ritratto 4x9 centrato ('q' per uscire)")
            else:
                init_scale = PREVIEW_SCALE
                if height * init_scale > PREVIEW_MAX_INIT_H:
                    init_scale = PREVIEW_MAX_INIT_H / height
                cv2.resizeWindow(self.WIN, int(width * init_scale), int(height * init_scale))
                print("[output] anteprima su schermo (finestra ridimensionabile)")

    def show(self, pil_img: Image.Image):
        if self.hardware is not None:
            self.hardware.SetImage(pil_img.convert("RGB"))
        else:
            frame = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
            if FULLSCREEN:
                cv2.imshow(self.WIN, frame)   # la finestra fullscreen scala all'intero schermo
            else:
                preview = cv2.resize(frame, (self.width * PREVIEW_SCALE, self.height * PREVIEW_SCALE),
                                     interpolation=cv2.INTER_NEAREST)
                cv2.imshow(self.WIN, preview)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def fit_centered(img: np.ndarray, out_w: int, out_h: int, zoom: float) -> np.ndarray:
    """ZOOM DIGITALE che RIEMPIE sempre il pannello (mai un riquadro nero fluttuante).
    zoom=1.0 -> tutta l'immagine adattata al pannello (vista naturale);
    zoom>1.0 -> ritaglia il centro e lo ingrandisce (ci si avvicina). Non si puo' fare
    zoom<1 utile: il pannello e' una finestra stretta su una camera larga e usa gia' tutta
    l'altezza -> per apparire piu' lontani/piccoli ci si allontana dalla camera (come nello
    specchio vero, dove stai a distanza)."""
    zoom = max(1.0, min(3.0, float(zoom)))
    if zoom <= 1.001:
        return cv2.resize(img, (out_w, out_h))
    h, w = img.shape[:2]
    cw, ch = max(1, int(w / zoom)), max(1, int(h / zoom))   # regione centrale da ingrandire
    x0, y0 = (w - cw) // 2, (h - ch) // 2
    crop = img[y0:y0 + ch, x0:x0 + cw]
    return cv2.resize(crop, (out_w, out_h))


def letterbox_fit(img: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """Adatta TUTTA l'immagine dentro il pannello mantenendo le proporzioni, centrata su
    fondo nero (bande nere sopra/sotto per una camera larga in un pannello alto): si vede
    l'intera inquadratura, la persona piu' piccola."""
    h, w = img.shape[:2]
    scale = min(out_w / w, out_h / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    small = cv2.resize(img, (nw, nh))
    canvas = np.zeros((out_h, out_w, 3), dtype=img.dtype)
    ox, oy = (out_w - nw) // 2, (out_h - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = small
    return canvas


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(
            "[camera] impossibile aprire la camera (indice", CAMERA_INDEX, ").\n"
            "Su macOS: Impostazioni di Sistema > Privacy e sicurezza > Fotocamera\n"
            "-> assicurati che il terminale/IDE che esegue questo script sia autorizzato."
        )
        return

    extractor = SilhouetteExtractor()      # persona (MediaPipe)
    foreground = ForegroundExtractor()     # oggetti (OpenCV background subtraction)
    thermal = ThermalRenderer()
    widgets = WidgetManager(PANEL_WIDTH, PANEL_HEIGHT)
    transcript = TranscriptDisplay(PANEL_WIDTH, PANEL_HEIGHT)
    output = PanelOutput(PANEL_WIDTH, PANEL_HEIGHT)
    idle = IdleAnimation() if IDLE_ENABLED else None   # animazione quando non c'e' nessuno
    orb = AiOrb(PANEL_HEIGHT) if AI_ORB_ENABLED else None  # blob AI ASCII in basso

    # parametri REGOLABILI DAL VIVO (dalla finestra inspector). Partono dai valori di config
    # qui sopra e possono essere cambiati coi cursori mentre il programma gira.
    live = {
        "face_threshold": FACE_THRESHOLD,
        "video_every_s": VOICE_VIDEO_EVERY_S,
        "proactive_on": 1.0 if PROACTIVE_CHATTER else 0.0,
        "proactive_quiet_s": PROACTIVE_QUIET_S,
        "proactive_min_gap": PROACTIVE_MIN_GAP,
        "person_view_scale": PERSON_VIEW_SCALE,   # zoom digitale del termico: 1.0 naturale, >1 avvicina
        "wide_view": 1.0 if WIDE_VIEW else 0.0,    # 1 = tutta la camera (bande nere), 0 = ritaglio pannello
    }
    from inspector import Inspector
    inspector = Inspector(live)
    if INSPECTOR_ENABLED:
        inspector.open()

    # riconoscimento volti (Fase 1): occhi che ricordano le persone
    face_brain = None
    if FACE_ENABLED:
        try:
            from face_brain import FaceBrain
            face_brain = FaceBrain(FACE_MEMORY_PATH, threshold=FACE_THRESHOLD,
                                   every_s=FACE_EVERY_S, model=FACE_MODEL,
                                   read=FACE_READ_ENABLED)
            face_brain.start()
        except Exception as exc:
            print(f"[face] non avviato: {repr(exc)[:160]}")

    # memoria episodica della casa (Fase 2): registro eventi/presenze/osservazioni
    brain = None
    if BRAIN_ENABLED:
        try:
            from home_brain import HomeBrain
            brain = HomeBrain(BRAIN_DB_PATH)
            print("[brain] memoria di casa attiva")
        except Exception as exc:
            print(f"[brain] non avviato: {repr(exc)[:160]}")

    # assistente vocale "vivo" (Gemini Live): usa la stessa chiave GEMINI_API_KEY
    assistant = None
    if VOICE_ENABLED:
        api_key = os.environ.get(GEMINI_API_KEY_ENV)
        if not api_key:
            print(f"[voice] {GEMINI_API_KEY_ENV} non impostata -> assistente vocale OFF")
        else:
            try:
                from voice_assistant import LiveAssistant
                mic_name = os.environ.get("THERMIRROR_MIC", VOICE_MIC_NAME)
                speaker_name = os.environ.get("THERMIRROR_SPEAKER", VOICE_SPEAKER_NAME)
                assistant = LiveAssistant(api_key, VOICE_MODEL, VOICE_PERSONA, VOICE_NAME,
                                          video_every_s=VOICE_VIDEO_EVERY_S,
                                          mic_name=mic_name, speaker_name=speaker_name,
                                          allow_device_control=VOICE_ALLOW_DEVICE_CONTROL,
                                          allow_proactive=VOICE_PROACTIVE)
                # Mira puo' imparare un volto a voce: enroll_person(name) -> face_brain
                # e guardare chi ha davanti: look_at_person -> face_brain.describe()
                if face_brain is not None and face_brain.available:
                    assistant.on_enroll = face_brain.enroll_current
                    assistant.on_look = face_brain.describe
                # memoria episodica: recall / who_was_here / log_observation -> brain
                assistant.brain = brain
                assistant.start()
                print("[voice] assistente vocale avviato -> parla; 'm' per mutare il microfono")

                # CHAT DA TASTIERA (per test): scrivi nel terminale + Invio -> Mira risponde a voce
                def _keyboard_chat():
                    import sys
                    print("[chat] scrivi qui un messaggio + Invio per parlare con Mira "
                          "(anche senza voce); Ctrl-D per chiudere la chat da tastiera.")
                    try:
                        for line in sys.stdin:
                            line = line.strip()
                            if line:
                                assistant.send_text(line)
                    except Exception:
                        pass
                threading.Thread(target=_keyboard_chat, daemon=True).start()
            except Exception as exc:
                print(f"[voice] non avviato: {repr(exc)[:160]}")

    print("Calibrazione sfondo in corso: resta fuori dall'inquadratura per un istante "
          "(serve a OpenCV per imparare lo sfondo e isolare gli oggetti).")
    print("Tasti: 'q' esci | 'm' muta microfono | 'd' finestra vista+parametri | "
          "'s' widget stock | 'a' risposta AI")

    frame_count = 0
    present_streak = 0     # frame consecutivi CON persona
    absent_streak = 0      # frame consecutivi SENZA persona
    person_here = True     # stato stabile (isteresi); True all'avvio -> niente idle nel warmup
    idle_level = 0.0       # dissolvenza continua: 0 = termico, 1 = animazione idle
    absent_since = None    # istante in cui la persona e' sparita (per il reset a 5 min)
    reset_done = False     # evita di richiedere il reset piu' volte per la stessa assenza
    greeted_name = None    # chi ho gia' salutato in questa presenza (evita saluti ripetuti)
    asked_new = False      # ho gia' chiesto "chi sei?" allo sconosciuto in questa presenza
    last_proactive = time.time()       # quando Mira ha preso lei l'iniziativa l'ultima volta
    proactive_gap = PROACTIVE_MIN_GAP  # intervallo corrente (cresce se resta ignorata)
    proactive_strikes = 0              # iniziative ignorate di fila (poi si sospende)
    recent_notes = []                  # ultime regie usate (per non ripetersi)
    dormant = False                    # True = si e' arresa con garbo -> schermo idle anche se c'e' qualcuno
    last_out_name = None   # ultimo dispositivo di uscita (per il toast al cambio)
    person_full_cache = None  # maschera persona riusata tra un frame di segmentazione e l'altro
    while True:
        loop_t0 = time.time()
        ok, frame = cap.read()
        if not ok:
            break

        full = cv2.flip(frame, 1)  # effetto specchio

        # inquadratura: NORMALE = ritaglia la striscia centrale al formato pannello (riempie);
        # LARGA = usa tutta la camera (poi messa nel pannello con bande nere) -> si vede di piu'.
        if live["wide_view"] >= 0.5:
            frame, crop_x0, crop_y0 = full, 0, 0
        else:
            frame, crop_x0, crop_y0 = crop_to_aspect(full, PANEL_WIDTH, PANEL_HEIGHT)

        # applica i parametri REGOLATI DAL VIVO (cursori dell'inspector) a chi li usa
        if face_brain is not None:
            face_brain.threshold = live["face_threshold"]
        # l'assistente vocale "vede" attraverso la camera (frame intero, piu' contesto)
        if assistant is not None:
            assistant.set_frame(full)
            assistant.video_every_s = live["video_every_s"]
        # il "cervello dei volti" analizza (in un suo thread) chi c'e' davanti
        if face_brain is not None:
            face_brain.submit(full)

        # PERSONA: MediaPipe sul frame intero ma DOWNSCALATO e solo 1 frame su SEG_EVERY
        # (rende meglio su aspetto normale + molta meno CPU -> audio piu' fluido). La
        # maschera viene riusata nel mezzo, poi ritagliata alla porzione visibile.
        if person_full_cache is None or frame_count % SEG_EVERY == 0:
            seg_in = cv2.resize(full, None, fx=SEG_SCALE, fy=SEG_SCALE) if SEG_SCALE < 0.99 else full
            pm = extractor.get_mask(seg_in)
            if pm.shape[:2] != full.shape[:2]:
                pm = cv2.resize(pm, (full.shape[1], full.shape[0]), interpolation=cv2.INTER_NEAREST)
            person_full_cache = pm
        person_full = person_full_cache
        fh, fw = frame.shape[:2]
        person_mask = person_full[crop_y0:crop_y0 + fh, crop_x0:crop_x0 + fw]
        if person_mask.shape[:2] != (fh, fw):
            person_mask = cv2.resize(person_mask, (fw, fh), interpolation=cv2.INTER_NEAREST)
        # OGGETTI: primo piano MENO persona. Calcolato SEMPRE su `full` (dimensione costante,
        # cosi' il background-subtractor non si rompe se cambio inquadratura dal vivo), poi
        # ritagliato alla porzione visibile come la persona.
        fg_full = foreground.get_mask(full)
        object_full = objects_from_foreground(fg_full, person_full)
        object_mask = object_full[crop_y0:crop_y0 + fh, crop_x0:crop_x0 + fw]
        if object_mask.shape[:2] != (fh, fw):
            object_mask = cv2.resize(object_mask, (fw, fh), interpolation=cv2.INTER_NEAREST)
        if frame_count < FG_WARMUP_FRAMES:
            # durante il warmup lo sfondo non e' ancora appreso -> niente oggetti (evita rumore)
            object_mask[:] = 0
            if not extractor.uses_segmentation:
                person_mask[:] = 0
        frame_count += 1

        # --- rilevamento persona affidabile: la componente connessa piu' grande, con isteresi
        # (soglie diverse per entrare/uscire) cosi' lo stato non sfarfalla ai bordi ---
        person_frac = largest_component_fraction(person_mask)
        if person_frac > IDLE_PERSON_MIN:
            present_streak += 1
            absent_streak = 0
        else:
            absent_streak += 1
            present_streak = 0
        if present_streak >= IDLE_PRESENT_FRAMES:
            person_here = True
        elif absent_streak >= IDLE_ENTER_FRAMES:
            person_here = False
        # (nel mezzo mantiene lo stato precedente)

        # --- reset conversazione dopo assenza prolungata (la MEMORIA resta) ---
        if person_here:
            absent_since = None
            reset_done = False
        else:
            if absent_since is None:
                absent_since = time.time()
            elif not reset_done and (time.time() - absent_since) > RESET_AFTER_S:
                if assistant is not None:
                    assistant.request_reset()
                reset_done = True
            # nessuno davanti -> alla prossima persona ricomincia saluti/domande da capo
            greeted_name = None
            asked_new = False
            last_proactive = time.time()       # timer iniziativa: riparte al prossimo arrivo
            proactive_gap = PROACTIVE_MIN_GAP
            proactive_strikes = 0
            recent_notes.clear()
            dormant = False                    # torna sveglia per la prossima persona

        # --- Mira PARLA LEI quando riconosce (o no) chi ha davanti ---
        if (person_here and face_brain is not None and assistant is not None
                and getattr(assistant, "running", False)):
            fstatus, fname, _ = face_brain.current()
            if fstatus == "known" and fname and greeted_name != fname:
                assistant.nudge(f"({fname} just walked up and you know them. Say a quick, casual "
                                "hi by name, like a friend would - one short line, nothing formal.)")
                greeted_name = fname
                last_proactive = time.time()   # ha appena parlato -> rimanda l'iniziativa
                widgets.show(f"hi {fname}", position="greet", ttl_seconds=4.0, color=UI_ICE)
                if brain is not None:
                    brain.log_presence(fname, place="mirror")     # log "who came by"
            elif fstatus == "unknown" and not asked_new and greeted_name is None:
                descr = face_brain.describe()      # e.g. "a man, around 30, you don't know"
                who = f" (looks like {descr})" if descr else ""
                assistant.nudge(f"(Someone you don't recognize just walked up{who}. Say a casual "
                                "hi, mention you don't think you've met, and ask their name - keep "
                                "it light and normal, like a real person, not formal or corny.)")
                asked_new = True
                last_proactive = time.time()
                if brain is not None:
                    brain.log_presence("(unknown)", place="mirror")

        # --- ROUTINE: Mira rompe lei il silenzio, senza ripetersi e senza assillare ---
        if (live["proactive_on"] >= 0.5 and person_here and assistant is not None
                and getattr(assistant, "running", False)
                and getattr(assistant, "allow_proactive", False)):
            now = time.time()
            quiet = assistant.quiet_seconds()
            min_gap = live["proactive_min_gap"]
            if quiet < 3.0:
                # l'utente ha parlato -> conversazione viva: si SVEGLIA e azzera tutto
                if dormant:
                    dormant = False
                    print("[voice] Mira si risveglia (l'utente le ha parlato)")
                proactive_gap = min_gap
                proactive_strikes = 0
                last_proactive = now
            elif (not dormant and not assistant.speaking
                    and quiet > live["proactive_quiet_s"]
                    and (now - last_proactive) > proactive_gap):
                if proactive_strikes >= PROACTIVE_MAX_TRIES:
                    # ignorata troppe volte -> saluto garbato e SOSPENSIONE (schermo idle)
                    assistant.nudge("(They haven't answered in a while. Say ONE short, warm "
                                    "goodbye - like 'alright, I'll leave you be - just call me if "
                                    "you need me' - relaxed, no guilt-trip. Then you'll go quiet.)")
                    dormant = True
                    last_proactive = now
                    print("[voice] Mira si mette in pausa con garbo (ignorata a lungo)")
                else:
                    read = face_brain.read() if face_brain is not None else {}
                    note = proactive_note(read, recent_notes)
                    assistant.nudge(note)
                    recent_notes.append(note)
                    del recent_notes[:-4]              # ricorda solo le ultime 4 (anti-ripetizione)
                    proactive_strikes += 1
                    last_proactive = now
                    # se resta ignorata, la prossima volta aspetta di piu' (non insistere)
                    proactive_gap = min(PROACTIVE_MAX_GAP, max(min_gap, proactive_gap) * 1.6)

        # --- dissolvenza FLUIDA termico <-> idle quando qualcuno entra/esce dalla scena ---
        # (dormant = Mira si e' arresa con garbo -> schermo idle anche se c'e' ancora qualcuno)
        if idle is None or frame_count < FG_WARMUP_FRAMES:
            target = 0.0
        else:
            target = 0.0 if (person_here and not dormant) else 1.0
        if idle_level < target:
            idle_level = min(target, idle_level + IDLE_FADE_SPEED)
        elif idle_level > target:
            idle_level = max(target, idle_level - IDLE_FADE_SPEED)

        # renderizza solo cio' che serve, poi fondi (crossfade) in base a idle_level
        thermal_img = None
        if idle_level < 0.999:
            entity = thermal.render(frame.shape, person_mask, object_mask)
            if live["wide_view"] >= 0.5:
                # vista LARGA: tutta la camera nel pannello (bande nere) -> si vede di piu'
                thermal_img = letterbox_fit(entity, PANEL_WIDTH, PANEL_HEIGHT)
            else:
                # zoom digitale: 1.0 = vista naturale (riempie), >1 = ritaglia il centro (avvicina)
                thermal_img = fit_centered(entity, PANEL_WIDTH, PANEL_HEIGHT, live["person_view_scale"])
        if idle_level <= 0.001:
            entity_resized = thermal_img
        elif idle_level >= 0.999:
            entity_resized = idle.render(PANEL_WIDTH, PANEL_HEIGHT)
        else:
            idle_img = idle.render(PANEL_WIDTH, PANEL_HEIGHT)
            entity_resized = cv2.addWeighted(thermal_img, 1.0 - idle_level, idle_img, idle_level, 0.0)

        entity_rgb = cv2.cvtColor(entity_resized, cv2.COLOR_BGR2RGB)
        panel_img = Image.fromarray(entity_rgb)

        # --- UI (chat + stato + blob AI): visibile solo quando c'e' la PERSONA ---
        # sfuma dolcemente insieme all'idle (ui_alpha=1 con persona, ->0 quando entra l'idle),
        # quindi sparisce se te ne vai e ricompare quando torni.
        # device di uscita cambiato -> toast transitorio "now on X" (niente stato permanente)
        if assistant is not None and getattr(assistant, "running", False):
            out_now = getattr(assistant, "output_name", None)
            if out_now and out_now != last_out_name:
                if last_out_name is not None:
                    widgets.show(f"› now on {out_now}", position="device", ttl_seconds=3.5)
                last_out_name = out_now

        ui_alpha = max(0.0, 1.0 - idle_level * 1.6)
        if ui_alpha > 0.02:
            user_line = assistant_line = ""
            if assistant is not None:
                user_line, assistant_line = assistant.get_lines()
            # SOLO due cose persistenti: conversazione (in alto) + presenza (blob, in alto a dx).
            # Lo stato (device/chi vede) e' un toast transitorio in basso, poi svanisce.
            transcript.render(panel_img, user_line, assistant_line, alpha=ui_alpha)
            panel_img = widgets.render(panel_img, alpha=ui_alpha)
            # blob AI ASCII in alto a destra: pulsa con la voce (mia o di Mira)
            if orb is not None and assistant is not None and getattr(assistant, "running", False):
                orb.draw(panel_img, voice_level=getattr(assistant, "activity", 0.0),
                         muted=assistant.muted, alpha=ui_alpha)
        output.show(panel_img)

        # finestra di controllo/debug (separata): cosa vede Mira + come ti legge + regolazioni
        inspector.render(full, face_brain, assistant,
                         {"person_here": person_here, "proactive_gap": proactive_gap})

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key in (ord("d"), ord("D")):
            inspector.toggle()
            print(f"[inspector] finestra {'aperta' if inspector.visible else 'chiusa'} "
                  "(cosa vede Mira + parametri regolabili)")
        elif key in (ord("m"), ord("M")):
            if assistant is not None:
                muted = assistant.toggle_mute()
                widgets.show("mic muted" if muted else "mic on", position="mic", ttl_seconds=2.5)
                print(f"[voice] microfono {'MUTO' if muted else 'attivo'}")
        elif key == ord("s"):
            widgets.show("AAPL 212.40  +1.2%", position="stock", ttl_seconds=5.0, color=UI_ICE)
        elif key == ord("a"):
            widgets.show("Firenze  24°  clear", position="info", ttl_seconds=5.0, color=UI_ICE)

        # TETTO FPS: dormi il tempo residuo -> libera la CPU per il thread audio (mic/voce)
        dt = time.time() - loop_t0
        if dt < 1.0 / TARGET_FPS:
            time.sleep(1.0 / TARGET_FPS - dt)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
