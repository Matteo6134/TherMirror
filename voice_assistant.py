"""
Voice assistant "vivo" per TherMirror (Google Gemini Live API)
==============================================================

Conversazione vocale in tempo reale + trascrizione a schermo:
  - il microfono viene inviato in streaming a Gemini Live;
  - Gemini risponde CON LA VOCE (audio riprodotto sugli altoparlanti);
  - Gemini VEDE la camera (mandiamo dei frame) -> puo' dire cosa indossi/tieni in mano;
  - Google Search grounding -> meteo, news, prezzi/azioni, ecc.;
  - le trascrizioni (utente e assistente) sono esposte per mostrarle sul pannello.

Gira in un thread in background con il proprio loop asyncio, cosi' il rendering
termico del programma principale resta fluido.

Dipendenze:  pip install google-genai sounddevice
NB: su macOS serve il permesso Microfono per il terminale/IDE che esegue lo script.
"""

import asyncio
import threading
import queue
import time
import os
import json
import numpy as np

MIC_RATE = 16000     # Gemini Live vuole PCM 16 kHz in ingresso
SPK_RATE = 24000     # l'audio in uscita di Gemini e' a 24 kHz
MIC_BLOCK = 1024     # campioni per blocco del microfono

# device audio "virtuali"/loopback: NON catturano audio reale -> mai usarli per il microfono
_VIRTUAL_DEVICES = ("teams", "zoom", "aggregate", "blackhole", "loopback", "soundflower",
                    "vb-audio", "vb-cable", "voicemeeter", "virtual", "obs", "ndi")


def _is_virtual(name: str) -> bool:
    n = (name or "").lower()
    return any(v in n for v in _VIRTUAL_DEVICES)


class _ResetSession(Exception):
    """Segnale interno: chiudi la sessione corrente e riparti pulita (assenza prolungata)."""


class Memory:
    """Memoria PERSISTENTE di Mira: fatti che impara e salva su file, ricaricati a ogni
    avvio -> 'costruisce' memoria nel tempo. La conversazione a breve termine puo'
    azzerarsi, ma questi ricordi restano (nome, gusti, cose dette dall'utente...)."""

    MAX_FACTS = 40

    def __init__(self, path):
        self.path = path
        self.facts = []
        self._lock = threading.Lock()
        self._load()

    def _load(self):
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                self.facts = [str(x) for x in data][-self.MAX_FACTS:]
        except Exception:
            self.facts = []

    def _save(self):
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.facts, f, ensure_ascii=False, indent=0)
        except Exception as exc:            # pragma: no cover
            print(f"[voice] memoria: salvataggio fallito: {repr(exc)[:80]}")

    def add(self, fact):
        fact = (fact or "").strip()
        if not fact:
            return "niente da ricordare."
        with self._lock:
            if fact in self.facts:
                return "lo ricordavo gia'."
            self.facts.append(fact)
            self.facts = self.facts[-self.MAX_FACTS:]
            self._save()
        print(f"[voice] memoria +1: {fact[:80]}")
        return "ricordato."

    def text(self):
        with self._lock:
            if not self.facts:
                return "(ancora nessun ricordo)"
            return "\n".join("- " + f for f in self.facts)


class LiveAssistant:
    """Assistente vocale in tempo reale. Uso:
        va = LiveAssistant(api_key, model, persona, voice)
        va.start()
        va.set_frame(frame_bgr)   # ad ogni frame, per dargli la vista
        user, assistant = va.get_lines()   # per la trascrizione a schermo
    """

    def __init__(self, api_key, model, persona, voice="Aoede", video_every_s=1.0,
                 mic_name=None, speaker_name=None, allow_device_control=True,
                 memory_path="mira_memory.json", allow_proactive=False):
        self.api_key = api_key
        self.model = model
        self.persona = persona
        self.voice = voice
        self.video_every_s = video_every_s
        # nome (anche parziale) del dispositivo audio da preferire; None = default di
        # sistema. Utile per forzare AirPods / altoparlante Bluetooth / mic USB.
        self.mic_name = mic_name
        self.speaker_name = speaker_name
        # se True, Mira puo' collegarsi a un dispositivo audio a voce (function calling)
        self.allow_device_control = allow_device_control
        # se True, Mira puo' parlare per prima (nudge/send_client_content); su alcuni
        # modelli native-audio va tenuto False (da' errore 1007)
        self.allow_proactive = allow_proactive
        # memoria persistente + richiesta di reset (assenza prolungata -> riparti pulito)
        self.memory = Memory(memory_path)
        self._reset_requested = False

        # parlato PROATTIVO (Mira inizia lei a parlare per un evento) + enrollment volti
        self._nudge_q = queue.Queue()      # messaggi/eventi da comunicare a voce
        self.on_enroll = None              # callback(name)->str per imparare un volto
        self.on_look = None                # callback()->str: chi c'e' davanti (nome / uomo-donna)
        self.brain = None                  # HomeBrain (memoria episodica) per recall/log

        # ciclo di vita dell'audio (per cambiare dispositivo a runtime)
        self._loop = None          # event loop asyncio (per la callback del mic)
        self._in_dev = None        # indice microfono corrente
        self._out_dev = None       # indice altoparlante corrente
        self._mic_stream = None    # stream del microfono (ricreabile)
        self._out_dirty = False    # segnala al worker di riaprire l'uscita su un nuovo device
        self._out_name = "default" # nome del device di uscita corrente (per la UI/status)
        self._stop = False         # richiesta di arresto (ferma il ciclo di riconnessione)
        self._resume_handle = None # handle per RIPRENDERE la sessione dopo il GoAway (~10 min)

        self.running = False
        self.error = None
        self.muted = False

        self._latest_frame = None
        self._frame_lock = threading.Lock()

        self._user_line = ""       # trascrizione corrente dell'utente
        self._assistant_line = ""  # trascrizione corrente dell'assistente
        self._txt_lock = threading.Lock()
        self._reset_assistant_next = False

        self._play_q = queue.Queue()   # bytes PCM da riprodurre
        self._mic_q = None             # asyncio.Queue creata nel loop

        # energia vocale (per animare il blob AI): 0..1, con decadimento nel tempo
        self._speak_level = 0.0    # voce di Mira (uscita/altoparlante)
        self._level_time = 0.0
        self._mic_level = 0.0      # voce dell'utente (ingresso/microfono)
        self._mic_time = 0.0
        self._voice_time = 0.0     # ultimo istante in cui l'utente ha DAVVERO parlato (non solo rumore)

    @property
    def level(self) -> float:
        """Livello 0..1 della voce di Mira ADESSO: sale quando parla, decade in ~0.25 s
        di silenzio."""
        dt = time.time() - self._level_time
        return float(self._speak_level * max(0.0, 1.0 - dt / 0.25))

    @property
    def mic_level(self) -> float:
        """Livello 0..1 della voce dell'UTENTE dal microfono ADESSO (0 se mutato)."""
        dt = time.time() - self._mic_time
        return float(self._mic_level * max(0.0, 1.0 - dt / 0.25))

    @property
    def activity(self) -> float:
        """Attivita' vocale complessiva 0..1: il massimo tra la voce di Mira e la mia.
        Anima il blob AI sia quando parla lei sia quando parlo io."""
        return max(self.level, self.mic_level)

    @property
    def speaking(self) -> bool:
        """True se Mira sta parlando proprio adesso (per non interromperla)."""
        return self.level > 0.05

    def quiet_seconds(self) -> float:
        """Da quanti secondi c'e' SILENZIO: nessuno (ne' Mira ne' l'utente) parla.
        Grande = pausa lunga -> Mira puo' prendere lei l'iniziativa."""
        last = max(self._level_time, self._voice_time)
        return (time.time() - last) if last > 0 else 1e9

    @property
    def output_name(self) -> str:
        """Nome del dispositivo di uscita audio corrente (per la barra di stato)."""
        return self._out_name

    # ------------------------------------------------------------------ API
    def start(self):
        threading.Thread(target=self._thread_main, daemon=True).start()

    def set_frame(self, frame_bgr):
        with self._frame_lock:
            self._latest_frame = frame_bgr

    def toggle_mute(self) -> bool:
        self.muted = not self.muted
        return self.muted

    def get_lines(self):
        """(riga_utente, riga_assistente) dell'ultimo scambio, per l'overlay a schermo."""
        with self._txt_lock:
            return self._user_line, self._assistant_line

    def request_reset(self):
        """Chiede di ripartire con una conversazione PULITA (contesto azzerato). La
        memoria persistente resta. Usato dal programma quando la persona manca da un po'."""
        self._reset_requested = True

    def nudge(self, text: str):
        """Fa parlare Mira in modo PROATTIVO: `text` e' una regia (es. 'saluta Luca').
        Lei la interpreta e risponde a voce da sola. Attivo solo se allow_proactive=True
        (su alcuni modelli native-audio send_client_content da' errore 1007)."""
        if text and self.allow_proactive:
            self._nudge_q.put(text)

    def send_text(self, text: str):
        """Messaggio di TESTO scritto dall'utente (es. da tastiera): Mira risponde A VOCE.
        Esplicito -> non passa dal gate allow_proactive. Lo mostra anche a schermo."""
        text = (text or "").strip()
        if not text:
            return
        with self._txt_lock:
            self._user_line = text            # mostra a schermo cosa hai scritto
            self._reset_assistant_next = True
        self._nudge_q.put(text)

    # -------------------------------------------------------------- interni
    def _thread_main(self):
        try:
            asyncio.run(self._main())
        except Exception as exc:            # pragma: no cover (dipende dall'hardware audio)
            self.error = repr(exc)
            print(f"[voice] assistente terminato: {self.error[:200]}")

    async def _main(self):
        import sounddevice as sd
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self.api_key)
        self._loop = asyncio.get_running_loop()
        self._mic_q = asyncio.Queue()

        self._in_dev, self._out_dev = self._resolve_devices(sd, self.mic_name, self.speaker_name)
        try:
            self._out_name = sd.query_devices(self._out_dev)["name"]
        except Exception:
            pass

        # riproduzione + microfono restano vivi tra una riconnessione e l'altra
        threading.Thread(target=self._playback_worker, daemon=True).start()
        self._open_mic(sd)

        # ciclo di RICONNESSIONE: l'API Live puo' cadere (errori 1011 transitori). Invece
        # di terminare, riconnettiamo con backoff. Se cade SUBITO mentre usiamo il function
        # calling, la causa probabile e' la combinazione di strumenti non supportata dal
        # modello: si riprova senza, cosi' resta almeno la ricerca Google.
        backoff = 1.0
        allow_fn = self.allow_device_control
        try:
            while not self._stop:
                started = time.time()
                try:
                    config = self._build_config(types, allow_fn)
                    async with client.aio.live.connect(model=self.model, config=config) as session:
                        self.running = True
                        self.error = None
                        backoff = 1.0
                        fn_note = "" if allow_fn else " (senza controllo dispositivi)"
                        print(f"[voice] connesso a Gemini Live -> parla pure{fn_note}")
                        while not self._mic_q.empty():   # scarta l'audio accumulato offline
                            self._mic_q.get_nowait()
                        await asyncio.gather(
                            self._send_audio(session, types),
                            self._send_video(session, types),
                            self._send_nudges(session, types),
                            self._receive(session, types),
                        )
                except asyncio.CancelledError:
                    raise
                except _ResetSession:
                    self.running = False
                    self._resume_handle = None   # NON riprendere: vogliamo un contesto pulito
                    print("[voice] ripartenza pulita (memoria conservata)")
                    continue          # riconnette subito con contesto azzerato
                except Exception as exc:
                    self.running = False
                    lived = time.time() - started
                    self.error = repr(exc)
                    print(f"[voice] sessione interrotta dopo {lived:.0f}s: {self.error[:150]}")
                    low = self.error.lower()
                    if self._stop:
                        break
                    # SCADENZA sessione (GoAway ~10 min) = NORMALE -> riconnetti subito
                    # (con la resumption riprende il contesto). NON e' un errore permanente!
                    if ("goaway" in low or "go away" in low or "session durat" in low
                            or "aborted" in low or "1011" in low
                            or ("1008" in low and "not found" not in low)):
                        print("[voice] sessione scaduta/caduta -> riconnetto")
                        backoff = 1.0
                        await asyncio.sleep(0.3)
                        continue
                    # errore PERMANENTE: il modello NON esiste sulla key
                    if "not found" in low:
                        print("[voice] MODELLO non disponibile per questa API key -> mi fermo. "
                              "Cambia VOICE_MODEL con un Live valido: python voice_assistant.py --models")
                        break
                    # 1007 / "not supported" = la combinazione di tool non piace al modello
                    # native-audio -> riprova senza tool (resta voce + ricerca Google)
                    cfg_err = ("1007" in low or "not supported" in low or "invalid argument" in low)
                    if allow_fn and (lived < 5.0 or cfg_err):
                        allow_fn = False
                        print("[voice] riprovo SENZA i tool (resta voce + ricerca Google)")
                    if cfg_err and self.allow_proactive:
                        self.allow_proactive = False
                        print("[voice] disattivo il saluto proattivo (turni iniettati rifiutati)")
                    await asyncio.sleep(backoff)
                    backoff = min(15.0, backoff * 2)
        finally:
            self.running = False
            self._close_mic()

    def _full_persona(self):
        """Persona + memoria persistente (ricaricata a ogni riconnessione, cosi' i fatti
        salvati con remember() durante la sessione valgono anche dopo un reset)."""
        return (self.persona
                + "\n\n[MEMORY] Things you already remember about the user and past "
                  "conversations:\n" + self.memory.text()
                + "\n\nWhen you learn something important and lasting about the user (their "
                  "name, tastes, personal facts, preferences, what they like), use the "
                  "remember(fact) tool to save it, so you recall it next time. Don't save "
                  "trivia or temporary things. "
                  "You recognize people by face AND read how they seem. When someone starts "
                  "talking to you (or you want to know who's there) use look_at_person: if you "
                  "know them it returns their name -> greet them by name; if new, it tells you if "
                  "they look like a man/woman and their age. It ALSO tells you how they seem right "
                  "now (e.g. 'looks happy', 'seems tired', 'not looking at you') - use that to "
                  "react like a real person would (warmly, lightly), but don't over-read it or "
                  "announce that you're analyzing them. When a NEW person tells you their name (or the user "
                  "introduces them, e.g. 'this is my friend Anna'), use enroll_person(name) to "
                  "save their face so you recognize them next time. "
                  "You also have a home memory: use log_observation to note notable things you "
                  "see (where someone puts objects like keys, who comes in/out) and use "
                  "recall / who_was_here / when_last_seen to answer questions about what "
                  "happened or where things are.")

    def _build_config(self, types, allow_functions):
        """Costruisce la config della sessione. `allow_functions` abilita gli strumenti
        (controllo dispositivi + memoria) in aggiunta alla ricerca Google."""
        tools = [types.Tool(google_search=types.GoogleSearch())]
        if allow_functions:
            tools.append(types.Tool(function_declarations=self._device_tools(types)))
        cfg = dict(
            response_modalities=["AUDIO"],
            system_instruction=types.Content(parts=[types.Part(text=self._full_persona())]),
            tools=tools,
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=self.voice)
                )
            ),
        )
        # NB: l'API Live NON accetta safety_settings nel messaggio di 'setup' (da' 1007
        # "Unknown name safetySettings"): e' una funzione di generateContent, non di Live.
        # Quindi il tono schietto/le parolacce di Mira si guidano SOLO dalla persona.
        # RIPRESA sessione: al GoAway (~10 min) ci si riconnette continuando il contesto.
        # Protetto: se questa versione dell'SDK non lo supporta, si prosegue senza.
        try:
            cfg["session_resumption"] = types.SessionResumptionConfig(handle=self._resume_handle)
        except Exception:
            pass
        # COMPRESSIONE del contesto (finestra scorrevole): comprime i turni vecchi invece di
        # sbattere contro il limite di token -> conversazioni MOLTO piu' lunghe. Protetto.
        try:
            cfg["context_window_compression"] = types.ContextWindowCompressionConfig(
                sliding_window=types.SlidingWindow())
        except Exception:
            pass
        return types.LiveConnectConfig(**cfg)

    # ---------------------------------------------------------- microfono (runtime)
    def _mic_cb(self, indata, frames, time_info, status):
        """Callback del microfono (gira in un thread di PortAudio)."""
        if self.muted:
            return
        b = bytes(indata)
        # energia del blocco (RMS) -> livello 0..1 per animare il blob quando parlo io
        try:
            samp = np.frombuffer(b, dtype=np.int16).astype(np.float32)
            if samp.size:
                rms = float(np.sqrt(np.mean(samp * samp))) / 32768.0
                self._mic_level = min(1.0, rms * 8.0)   # il mic e' piu' debole -> piu' guadagno
                self._mic_time = time.time()
                if self._mic_level > 0.12:              # solo se e' voce vera, non fruscio
                    self._voice_time = time.time()
        except Exception:
            pass
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._mic_q.put_nowait, b)

    def _open_mic(self, sd):
        """(Ri)apre lo stream del microfono sul device corrente (self._in_dev).
        Se non c'e' un microfono (in_dev None) si prosegue senza input vocale."""
        self._close_mic()
        if self._in_dev is None:
            return
        st = sd.RawInputStream(samplerate=MIC_RATE, blocksize=MIC_BLOCK, channels=1,
                               dtype="int16", callback=self._mic_cb, device=self._in_dev)
        st.start()
        self._mic_stream = st

    def _close_mic(self):
        st = self._mic_stream
        self._mic_stream = None
        if st is not None:
            try:
                st.stop()
                st.close()
            except Exception:
                pass

    @staticmethod
    def _resolve_devices(sd, mic_name=None, speaker_name=None):
        """Sceglie un microfono e un altoparlante VALIDI (indice esplicito), invece di
        affidarsi al default che a volte e' -1 -> PortAudioError.

        Ordine di scelta per ciascuno:
          1) se e' stato indicato un nome (mic_name/speaker_name), il primo device il cui
             nome CONTIENE quella stringa (case-insensitive) con i canali del tipo giusto;
          2) il device di default del sistema;
          3) il primo device compatibile.
        Cosi' puoi forzare AirPods / altoparlante Bluetooth / mic USB senza dipendere dal
        default del sistema. Se manca il microfono (di solito permesso macOS negato) alza
        un errore con istruzioni chiare."""
        def by_name(want_input, name):
            key = "max_input_channels" if want_input else "max_output_channels"
            needle = name.lower()
            try:
                for i, dev in enumerate(sd.query_devices()):
                    if dev[key] > 0 and needle in dev["name"].lower() and not _is_virtual(dev["name"]):
                        return i
            except Exception:
                pass
            return None

        def find(want_input):
            key = "max_input_channels" if want_input else "max_output_channels"
            # candidati REALI (mai device virtuali tipo Microsoft Teams / Zoom / loopback)
            try:
                cands = [i for i, dev in enumerate(sd.query_devices())
                         if dev[key] > 0 and not _is_virtual(dev["name"])]
            except Exception:
                cands = []
            if not cands:
                return None
            # 1) il device di default, se e' tra i candidati reali
            try:
                default = sd.default.device
                cand = default[0] if want_input else default[1]
                if cand in cands:
                    return cand
            except Exception:
                pass
            # 2) per il MICROFONO preferisci quello INTEGRATO del Mac
            if want_input:
                for pref in ("macbook", "built-in", "internal", "microphone"):
                    for i in cands:
                        if pref in sd.query_devices(i)["name"].lower():
                            return i
            # 3) primo candidato reale
            return cands[0]

        # nome esplicito ha la precedenza; se non trovato, si ripiega sul default
        in_dev = by_name(True, mic_name) if mic_name else None
        if mic_name and in_dev is None:
            print(f"[voice] microfono '{mic_name}' non trovato -> uso il default")
        if in_dev is None:
            in_dev = find(True)

        out_dev = by_name(False, speaker_name) if speaker_name else None
        if speaker_name and out_dev is None:
            print(f"[voice] altoparlante '{speaker_name}' non trovato -> uso il default")
        if out_dev is None:
            out_dev = find(False)

        if out_dev is None:
            raise RuntimeError("nessun altoparlante disponibile per la riproduzione audio.")
        if in_dev is None:
            # NIENTE crash: si va avanti SENZA microfono (Mira parla lo stesso). Quasi sempre
            # e' il PERMESSO Microfono di macOS: concesso, il mic integrato/AirPods compare.
            try:
                names = ", ".join(f"{i}:{d['name']}" for i, d in enumerate(sd.query_devices()))
            except Exception:
                names = "nessuno"
            print("[voice] NESSUN microfono reale disponibile -> proseguo SENZA input vocale.")
            print("[voice] Fix: collega gli AirPods, oppure Impostazioni di Sistema > Privacy e "
                  "Sicurezza > Microfono -> abilita il Terminale/IDE, poi riavvia.")
            print("[voice] (i device virtuali tipo Microsoft Teams sono ignorati). Visti: " + names)
            try:
                out_nm = sd.query_devices(out_dev)["name"]
            except Exception:
                out_nm = "?"
            print(f"[voice] audio: microfono NESSUNO, altoparlante #{out_dev} ({out_nm})")
            return None, out_dev
        try:
            in_nm = sd.query_devices(in_dev)["name"]
            out_nm = sd.query_devices(out_dev)["name"]
        except Exception:
            in_nm = out_nm = "?"
        print(f"[voice] audio: microfono #{in_dev} ({in_nm}), altoparlante #{out_dev} ({out_nm})")
        return in_dev, out_dev

    def _playback_worker(self):
        """Riproduce l'audio di Mira. Lo stream viene RIAPERTO quando cambia il device
        di uscita (self._out_dirty), cosi' si puo' passare a un altro altoparlante a runtime."""
        import sounddevice as sd
        while True:
            try:
                self._out_dirty = False
                with sd.RawOutputStream(samplerate=SPK_RATE, channels=1, dtype="int16",
                                        device=self._out_dev) as spk:
                    while True:
                        data = self._play_q.get()
                        if self._out_dirty:      # device cambiato -> esci e riapri lo stream
                            break
                        if data is None:
                            continue
                        # Gemini manda l'audio a CHUNK grossi (anche un'intera frase) e
                        # spk.write() BLOCCA finche' non e' tutto suonato: se aggiornassimo il
                        # livello una volta sola per chunk, il blob si accenderebbe un istante e
                        # poi resterebbe fermo per tutta la frase. Quindi suoniamo a PICCOLI
                        # blocchi (~30 ms) e aggiorniamo il livello a ogni blocco -> il blob
                        # pulsa davvero con la voce di Mira, in tempo reale.
                        try:
                            buf = np.frombuffer(data, dtype=np.int16)
                        except Exception:
                            buf = None
                        if buf is None or buf.size == 0:
                            spk.write(data)
                            continue
                        step = max(1, int(SPK_RATE * 0.03))     # ~30 ms per blocco
                        for off in range(0, buf.size, step):
                            if self._out_dirty:                 # cambio device -> molla e riapri
                                break
                            sub = buf[off:off + step]
                            s = sub.astype(np.float32)
                            rms = float(np.sqrt(np.mean(s * s))) / 32768.0
                            self._speak_level = min(1.0, rms * 4.0)
                            self._level_time = time.time()
                            spk.write(sub.tobytes())
            except Exception as exc:            # pragma: no cover
                print(f"[voice] audio in uscita non disponibile: {repr(exc)[:150]}")
                return
            # se siamo qui e' per un cambio device: piccola pausa e riapertura
            if not self._out_dirty:
                return

    async def _send_audio(self, session, types):
        while True:
            data = await self._mic_q.get()
            await session.send_realtime_input(
                audio=types.Blob(data=data, mime_type=f"audio/pcm;rate={MIC_RATE}")
            )

    async def _send_video(self, session, types):
        import cv2
        while True:
            await asyncio.sleep(self.video_every_s)
            # reset richiesto (assenza prolungata): svuota la trascrizione e riparti pulito
            if self._reset_requested:
                self._reset_requested = False
                with self._txt_lock:
                    self._user_line = ""
                    self._assistant_line = ""
                    self._reset_assistant_next = False
                raise _ResetSession()
            with self._frame_lock:
                frame = self._latest_frame
            if frame is None:
                continue
            ok, jpg = cv2.imencode(".jpg", frame)
            if ok:
                await session.send_realtime_input(
                    video=types.Blob(data=jpg.tobytes(), mime_type="image/jpeg")
                )

    async def _send_nudges(self, session, types):
        """Invia gli eventi proattivi (accodati con nudge()) come un turno testuale:
        Mira risponde a voce da sola. Cosi' 'entra lei nella conversazione'."""
        while True:
            await asyncio.sleep(0.25)
            try:
                txt = self._nudge_q.get_nowait()
            except queue.Empty:
                continue
            try:
                await session.send_client_content(
                    turns=types.Content(role="user", parts=[types.Part(text=txt)]),
                    turn_complete=True,
                )
            except Exception as exc:            # pragma: no cover
                # il modello non accetta i turni iniettati -> spegni il proattivo e non
                # riprovare (resta la voce reattiva). NB: se invece chiude la sessione (1007),
                # ci pensa il ciclo di riconnessione.
                print(f"[voice] messaggio iniettato rifiutato -> proattivo OFF: {repr(exc)[:100]}")
                self.allow_proactive = False
                while not self._nudge_q.empty():
                    try:
                        self._nudge_q.get_nowait()
                    except queue.Empty:
                        break

    # ----------------------------------------- controllo dispositivi a voce (tool)
    def _device_tools(self, types):
        """Dichiarazione delle funzioni che Mira puo' chiamare a voce."""
        return [
            types.FunctionDeclaration(
                name="connect_audio_device",
                description=("Collega e instrada l'audio del mirror (microfono e/o "
                             "altoparlante) a un dispositivo, es. AirPods o una cassa "
                             "Bluetooth. Usa quando l'utente chiede di collegarsi o "
                             "passare a un dispositivo audio."),
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={"name": types.Schema(
                        type=types.Type.STRING,
                        description="nome anche parziale del dispositivo, es. 'AirPods'")},
                    required=["name"],
                ),
            ),
            types.FunctionDeclaration(
                name="list_audio_devices",
                description="Elenca i dispositivi audio disponibili a cui collegarsi.",
                # nessun parametro: NON passare uno Schema OBJECT vuoto (viene rifiutato)
            ),
            types.FunctionDeclaration(
                name="remember",
                description=("Salva nella memoria a lungo termine un fatto importante e "
                             "duraturo sull'utente (nome, gusti, preferenze, cose personali "
                             "che ha detto), cosi' lo ricordi nelle prossime conversazioni. "
                             "Non usare per banalita' o cose temporanee."),
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={"fact": types.Schema(
                        type=types.Type.STRING,
                        description="il fatto da ricordare, breve e in prima persona sull'utente")},
                    required=["fact"],
                ),
            ),
            types.FunctionDeclaration(
                name="look_at_person",
                description=("Guarda chi hai davanti allo specchio ADESSO: ti dice se lo "
                             "riconosci (il nome) oppure, se e' nuovo, se sembra un uomo/una "
                             "donna e l'eta'. Usalo quando qualcuno inizia a parlarti o vuoi "
                             "sapere con chi stai parlando, per salutarlo bene."),
                # nessun parametro
            ),
            types.FunctionDeclaration(
                name="enroll_person",
                description=("Memorizza il VOLTO della persona che hai davanti adesso "
                             "associandolo al suo nome, cosi' la riconoscerai la prossima "
                             "volta che la vedi (a qualsiasi camera). Usa quando una persona "
                             "nuova si presenta e ti dice come si chiama."),
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={"name": types.Schema(
                        type=types.Type.STRING,
                        description="il nome della persona da riconoscere")},
                    required=["name"],
                ),
            ),
            types.FunctionDeclaration(
                name="log_observation",
                description=("Annota nel diario di casa qualcosa di NOTEVOLE che vedi o "
                             "che ti viene detto e che potrebbe servire ricordare piu' "
                             "tardi: es. dove qualcuno posa un oggetto (chiavi, telefono), "
                             "chi entra/esce, cosa succede. Viene salvato con data e ora."),
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "detail": types.Schema(type=types.Type.STRING,
                                               description="cosa e' successo, breve e concreto"),
                        "who": types.Schema(type=types.Type.STRING,
                                            description="persona coinvolta, se c'e' (facoltativo)"),
                    },
                    required=["detail"],
                ),
            ),
            types.FunctionDeclaration(
                name="recall",
                description=("Cerca nella memoria di casa eventi/osservazioni passate per "
                             "rispondere a domande tipo 'dove ho lasciato le chiavi?', "
                             "'cosa e' successo stamattina?'. Ritorna gli episodi che combaciano."),
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "query": types.Schema(type=types.Type.STRING,
                                              description="parole chiave di cosa cerchi, es. 'chiavi'"),
                        "hours": types.Schema(type=types.Type.NUMBER,
                                              description="entro quante ore fa cercare (facoltativo)"),
                    },
                    required=["query"],
                ),
            ),
            types.FunctionDeclaration(
                name="who_was_here",
                description="Chi e' stato visto in casa di recente e quando.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={"hours": types.Schema(
                        type=types.Type.NUMBER,
                        description="entro quante ore fa (default 24)")},
                ),
            ),
            types.FunctionDeclaration(
                name="when_last_seen",
                description="Quando e' stata vista l'ultima volta una certa persona.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={"name": types.Schema(
                        type=types.Type.STRING, description="il nome della persona")},
                    required=["name"],
                ),
            ),
        ]

    async def _handle_tool_call(self, session, tool_call, types):
        """Esegue le funzioni richieste da Mira. Le operazioni bloccanti (Bluetooth,
        riapertura audio) girano in un executor per non congelare il loop audio."""
        responses = []
        for fc in tool_call.function_calls:
            try:
                if fc.name == "connect_audio_device":
                    name = (fc.args or {}).get("name", "")
                    result = await self._loop.run_in_executor(None, self._do_connect, name)
                elif fc.name == "list_audio_devices":
                    devs = await self._loop.run_in_executor(None, self._devices_str)
                    result = "dispositivi disponibili: " + devs
                elif fc.name == "remember":
                    result = self.memory.add((fc.args or {}).get("fact", ""))
                elif fc.name == "look_at_person":
                    if self.on_look is not None:
                        who = await self._loop.run_in_executor(None, self.on_look)
                        result = who or "non vedo bene nessuno adesso."
                    else:
                        result = "non ho occhi per riconoscere i volti adesso."
                elif fc.name == "enroll_person":
                    name = (fc.args or {}).get("name", "")
                    if self.on_enroll is not None:
                        result = await self._loop.run_in_executor(None, self.on_enroll, name)
                    else:
                        result = "riconoscimento volti non disponibile."
                elif fc.name == "log_observation":
                    a = fc.args or {}
                    if self.brain is not None:
                        self.brain.log("observation", who=a.get("who"), detail=a.get("detail", ""))
                        result = "annotato."
                    else:
                        result = "memoria di casa non disponibile."
                elif fc.name == "recall":
                    a = fc.args or {}
                    if self.brain is not None:
                        result = await self._loop.run_in_executor(
                            None, self.brain.recall, a.get("query", ""), a.get("hours"))
                    else:
                        result = "memoria di casa non disponibile."
                elif fc.name == "who_was_here":
                    hours = int((fc.args or {}).get("hours") or 24)
                    result = self.brain.who_was_here(hours) if self.brain else "memoria non disponibile."
                elif fc.name == "when_last_seen":
                    name = (fc.args or {}).get("name", "")
                    result = self.brain.last_seen(name) if self.brain else "memoria non disponibile."
                else:
                    result = f"funzione sconosciuta: {fc.name}"
            except Exception as exc:                # pragma: no cover
                result = f"errore: {repr(exc)[:100]}"
            responses.append(types.FunctionResponse(
                id=getattr(fc, "id", None), name=fc.name, response={"result": result}))
        try:
            await session.send_tool_response(function_responses=responses)
        except Exception as exc:                    # pragma: no cover
            print(f"[voice] invio tool response fallito: {repr(exc)[:100]}")

    @staticmethod
    def _index_by_name(sd, name, want_input):
        key = "max_input_channels" if want_input else "max_output_channels"
        needle = (name or "").lower()
        try:
            for i, dev in enumerate(sd.query_devices()):
                if dev[key] > 0 and needle in dev["name"].lower() and not _is_virtual(dev["name"]):
                    return i
        except Exception:
            pass
        return None

    def _do_connect(self, name):
        """1) prova a collegare il Bluetooth a livello OS; 2) instrada l'audio del mirror
        sul dispositivo. Ritorna un messaggio che Mira leggera' all'utente."""
        name = (name or "").strip()
        if not name:
            return "quale dispositivo? dimmi il nome, es. 'AirPods'."
        os_msg = self._os_bt_connect(name)
        route_msg = self._route_audio_to(name)
        msg = (os_msg + route_msg).strip()
        print(f"[voice] connect_audio_device('{name}') -> {msg}")
        return msg

    def _route_audio_to(self, name):
        """Sposta microfono e/o altoparlante del mirror sul dispositivo indicato
        (se il sistema lo vede gia' tra i device audio)."""
        import sounddevice as sd
        out_idx = self._index_by_name(sd, name, want_input=False)
        in_idx = self._index_by_name(sd, name, want_input=True)
        changed = []
        if out_idx is not None and out_idx != self._out_dev:
            self._out_dev = out_idx
            self._out_dirty = True
            self._play_q.put(None)          # sblocca il worker perche' riapra lo stream
            changed.append("altoparlante")
        if in_idx is not None and in_idx != self._in_dev:
            self._in_dev = in_idx
            try:
                self._open_mic(sd)
                changed.append("microfono")
            except Exception as exc:
                return f"non riesco ad aprire il microfono su '{name}': {repr(exc)[:80]}"
        if out_idx is None and in_idx is None:
            return (f"non vedo nessun dispositivo audio chiamato '{name}'. "
                    "Controlla che sia connesso nelle impostazioni di sistema.")
        if not changed:
            return f"sto gia' usando '{name}'."
        try:
            nm = sd.query_devices(out_idx if out_idx is not None else in_idx)["name"]
        except Exception:
            nm = name
        if out_idx is not None:
            self._out_name = nm
        return f"fatto: audio su '{nm}' ({' e '.join(changed)})."

    def _os_bt_connect(self, name):
        """Best-effort: se il device Bluetooth e' accoppiato ma non connesso, prova a
        collegarlo. Linux/Raspberry Pi -> bluetoothctl; macOS -> blueutil (se installato).
        Non solleva eccezioni: in caso di problemi si prosegue con l'instradamento."""
        import platform
        import shutil
        import subprocess
        system = platform.system()
        needle = name.lower()
        try:
            if system == "Linux":
                if shutil.which("bluetoothctl") is None:
                    return ""
                out = subprocess.run(["bluetoothctl", "devices"], capture_output=True,
                                     text=True, timeout=8).stdout
                mac = None
                for line in out.splitlines():
                    parts = line.split(maxsplit=2)   # "Device AA:BB:.. Nome"
                    if len(parts) >= 3 and parts[0] == "Device" and needle in parts[2].lower():
                        mac = parts[1]
                        break
                if mac is None:
                    return ""
                subprocess.run(["bluetoothctl", "connect", mac], capture_output=True,
                               text=True, timeout=15)
                return f"collegato via Bluetooth ({mac}). "
            if system == "Darwin":
                if shutil.which("blueutil") is None:
                    return ("(su Mac, per collegare un dispositivo non gia' connesso, "
                            "serve 'brew install blueutil') ")
                out = subprocess.run(["blueutil", "--paired"], capture_output=True,
                                     text=True, timeout=8).stdout
                mac = None
                for line in out.splitlines():
                    if needle in line.lower():
                        for tok in line.split(","):
                            tok = tok.strip()
                            if tok.startswith("address:"):
                                mac = tok.split(":", 1)[1].strip()
                                break
                        if mac:
                            break
                if mac is None:
                    return ""
                subprocess.run(["blueutil", "--connect", mac], capture_output=True,
                               text=True, timeout=15)
                return f"collegato via Bluetooth ({mac}). "
        except Exception as exc:
            return f"(collegamento Bluetooth OS non riuscito: {repr(exc)[:60]}) "
        return ""

    def _devices_str(self):
        import sounddevice as sd
        names, seen = [], set()
        try:
            for d in sd.query_devices():
                n = d["name"]
                if n not in seen and (d["max_input_channels"] > 0 or d["max_output_channels"] > 0):
                    seen.add(n)
                    names.append(n)
        except Exception:
            pass
        return ", ".join(names) if names else "nessun dispositivo audio trovato."

    async def _receive(self, session, types):
        while True:
            async for resp in session.receive():
                if resp.data:                       # audio in uscita -> altoparlanti
                    self._play_q.put(resp.data)

                # handle di RIPRESA: salvalo, cosi' alla riconnessione continuiamo il contesto
                sru = getattr(resp, "session_resumption_update", None)
                if sru is not None and getattr(sru, "resumable", False) and getattr(sru, "new_handle", None):
                    self._resume_handle = sru.new_handle

                # GoAway: la sessione sta per scadere (~10 min) -> chiudiamo NOI e riconnettiamo
                # (evita l'abort 1008 "failed to close"); la ripresa mantiene la conversazione
                if getattr(resp, "go_away", None) is not None:
                    raise RuntimeError("goaway: session duration limit -> reconnect")

                # richiesta di funzione (es. "collegati agli AirPods")
                if getattr(resp, "tool_call", None):
                    await self._handle_tool_call(session, resp.tool_call, types)
                    continue

                sc = resp.server_content
                if not sc:
                    continue

                if sc.input_transcription and sc.input_transcription.text:
                    with self._txt_lock:
                        self._user_line += sc.input_transcription.text

                if sc.output_transcription and sc.output_transcription.text:
                    with self._txt_lock:
                        if self._reset_assistant_next:
                            self._assistant_line = ""
                            self._reset_assistant_next = False
                        self._assistant_line += sc.output_transcription.text

                if getattr(sc, "interrupted", False):
                    # l'utente ha interrotto: svuota l'audio in coda
                    while not self._play_q.empty():
                        try:
                            self._play_q.get_nowait()
                        except queue.Empty:
                            break

                if sc.turn_complete:
                    # inizio di un nuovo turno: la prossima battuta dell'utente riparte pulita
                    with self._txt_lock:
                        self._user_line = ""
                        self._reset_assistant_next = True


def list_audio_devices():
    """Elenca i dispositivi audio con indice, nome e canali in/out.
    Serve per scoprire il nome esatto da passare a mic_name/speaker_name
    (o alle env THERMIRROR_MIC / THERMIRROR_SPEAKER)."""
    import sounddevice as sd
    print("Dispositivi audio (indice: nome  [in / out canali]):")
    for i, d in enumerate(sd.query_devices()):
        print(f"  {i}: {d['name']}  [in {d['max_input_channels']} / out {d['max_output_channels']}]")


def list_live_models():
    """Elenca i modelli della TUA API key che supportano la Live API (bidiGenerateContent):
    sono gli unici usabili come VOICE_MODEL. Serve GEMINI_API_KEY nell'ambiente."""
    import os
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        print("Imposta prima GEMINI_API_KEY:  export GEMINI_API_KEY=la_tua_chiave")
        return
    from google import genai
    client = genai.Client(api_key=key)
    print("Modelli Live (bidiGenerateContent) disponibili sulla tua chiave:")
    found = False
    for m in client.models.list():
        acts = getattr(m, "supported_actions", None) or []
        if "bidiGenerateContent" in acts:
            found = True
            print("  ", m.name)
    if not found:
        print("  (nessuno segnalato con bidiGenerateContent; elenco completo:)")
        for m in client.models.list():
            print("  ", m.name, "->", getattr(m, "supported_actions", ""))


if __name__ == "__main__":
    import sys
    if "--list" in sys.argv:
        list_audio_devices()
    elif "--models" in sys.argv:
        list_live_models()
    else:
        print("Modulo da importare da led_mirror_entity.py. Comandi:\n"
              "  python voice_assistant.py --list     # dispositivi audio\n"
              "  python voice_assistant.py --models   # modelli Live della tua API key")
