"""
Conversation display per TherMirror (pannello LED nudo 192x192, P3 64x64 x3x3)
==============================================================================

Mostra la conversazione con l'assistente vocale "Mira" direttamente sul pannello:
  - una "presenza" luminosa (orb) al centro che reagisce allo stato
    (in ascolto / sta pensando / sta parlando / a riposo);
  - un piccolo equalizzatore quando parla;
  - i sottotitoli: la domanda dell'utente (fioca) e la risposta di Mira (in evidenza),
    con testo a capo, pensati per essere leggibili su un pannello LED piccolo.

Sfondo NERO (pannello nudo, i LED spenti = neri). Quando ci sara' la termocamera reale,
l'immagine termica potra' essere passata come sfondo (thermal_bg).
"""

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

# stato -> colore della presenza (BGR) e testo
STATE_COLORS = {
    "idle":      (150, 110, 60),    # blu tenue: a riposo
    "listening": (255, 200, 90),    # ciano: ti sto ascoltando
    "thinking":  (220, 120, 210),   # viola: sto ragionando/cercando
    "speaking":  (120, 210, 255),   # ambra/caldo: sto parlando
}
STATE_LABEL = {
    "idle": "", "listening": "ascolto", "thinking": "penso...", "speaking": "",
}


def _load_font(size):
    for path in ("/System/Library/Fonts/SFNSRounded.ttf",
                 "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                 "/System/Library/Fonts/Helvetica.ttc",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


class ConversationDisplay:
    def __init__(self, width=192, height=192):
        self.w, self.h = width, height
        # dimensioni font pensate per un pannello ~192px (poi il pannello LED e' piccolo)
        self.font_mira = _load_font(max(12, height // 13))
        self.font_user = _load_font(max(10, height // 17))
        self.font_state = _load_font(max(9, height // 20))
        # griglia coordinate per l'orb (cache)
        ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
        self._xs, self._ys = xs, ys

    # ---------------------------------------------------------------- rendering
    def _draw_presence(self, canvas, t, state):
        """Orb luminoso al centro-alto, pulsa in base allo stato."""
        color = np.array(STATE_COLORS.get(state, STATE_COLORS["idle"]), np.float32)
        cx, cy = self.w / 2.0, self.h * 0.30
        base_r = self.h * 0.12

        # pulsazione: piu' viva quando parla/ascolta
        speed = {"idle": 1.2, "listening": 3.0, "thinking": 5.0, "speaking": 6.5}.get(state, 1.2)
        amp = {"idle": 0.12, "listening": 0.25, "thinking": 0.18, "speaking": 0.35}.get(state, 0.12)
        pulse = 1.0 + amp * np.sin(t * speed)
        sigma = base_r * pulse

        d2 = (self._xs - cx) ** 2 + (self._ys - cy) ** 2
        glow = np.exp(-d2 / (2.0 * sigma * sigma))          # alone morbido
        core = np.exp(-d2 / (2.0 * (sigma * 0.35) ** 2))    # nucleo brillante
        field = np.clip(glow * 0.8 + core * 0.9, 0, 1)[:, :, None]
        canvas[:] = np.clip(canvas + field * color, 0, 255)

    def _draw_equalizer(self, canvas, t, state):
        """Barrette animate sotto l'orb quando parla/ascolta."""
        if state not in ("speaking", "listening"):
            return
        color = STATE_COLORS[state]
        n = 7
        cx = self.w / 2.0
        gap = self.w * 0.045
        bw = max(2, int(self.w * 0.02))
        y0 = int(self.h * 0.50)
        maxh = self.h * 0.10
        for i in range(n):
            phase = i * 0.7
            hgt = int(maxh * (0.35 + 0.65 * abs(np.sin(t * 6.0 + phase))))
            x = int(cx + (i - n // 2) * gap)
            cv2.rectangle(canvas, (x - bw // 2, y0 - hgt), (x + bw // 2, y0 + hgt),
                          color, -1, cv2.LINE_AA)

    def _wrap(self, draw, text, font, max_w):
        lines, cur = [], ""
        for word in text.split():
            trial = (cur + " " + word).strip()
            if draw.textlength(trial, font=font) <= max_w or not cur:
                cur = trial
            else:
                lines.append(cur)
                cur = word
        if cur:
            lines.append(cur)
        return lines

    def _draw_captions(self, canvas_bgr, user_text, mira_text):
        """Sottotitoli in basso: utente (fioco) sopra, Mira (acceso) sotto."""
        if not user_text and not mira_text:
            return canvas_bgr
        img = Image.fromarray(cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        margin = max(4, self.w // 32)
        max_w = self.w - 2 * margin

        rows = []  # (testo, font, colore RGB)
        for ln in self._wrap(draw, user_text, self.font_user, max_w):
            rows.append((ln, self.font_user, (150, 150, 155)))
        for ln in self._wrap(draw, mira_text, self.font_mira, max_w):
            rows.append((ln, self.font_mira, (255, 245, 225)))

        # altezza riga ~ dal font piu' grande
        lh = int((self.font_mira.getbbox("Ay")[3]) * 1.25)
        # tieni le ultime righe che entrano nella meta' bassa
        max_rows = max(1, int((self.h * 0.42) / lh))
        rows = rows[-max_rows:]

        y = self.h - margin - lh * len(rows)
        for text, font, color in rows:
            # leggero contorno scuro per staccare dal fondo
            draw.text((margin, y), text, font=font, fill=(0, 0, 0))
            draw.text((margin - 1, y - 1), text, font=font, fill=color)
            y += lh
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    def render(self, t, state="idle", user_text="", mira_text="", thermal_bg=None):
        """Ritorna il frame BGR (H, W, 3) da mandare al pannello."""
        if thermal_bg is not None:
            canvas = cv2.resize(thermal_bg, (self.w, self.h)).astype(np.float32) * 0.55
        else:
            canvas = np.zeros((self.h, self.w, 3), np.float32)   # pannello nudo: nero

        self._draw_presence(canvas, t, state)
        canvas = np.clip(canvas, 0, 255).astype(np.uint8)
        self._draw_equalizer(canvas, t, state)

        # etichetta stato piccola sotto l'orb
        label = STATE_LABEL.get(state, "")
        if label:
            (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)
            cv2.putText(canvas, label, (self.w // 2 - tw // 2, int(self.h * 0.44)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (160, 160, 160), 1, cv2.LINE_AA)

        canvas = self._draw_captions(canvas, user_text, mira_text)
        return canvas


# ---------------------------------------------------------------------------
# Demo: renderizza alcuni fotogrammi di una conversazione di esempio
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    disp = ConversationDisplay(192, 192)

    scenes = [
        (0.0, "idle", "", "Ciao, sono Mira."),
        (0.5, "listening", "che tempo fa oggi a Firenze?", ""),
        (1.0, "thinking", "che tempo fa oggi a Firenze?", ""),
        (1.5, "speaking", "che tempo fa oggi a Firenze?",
         "A Firenze oggi sole, 24 gradi. Bella giornata!"),
    ]
    names = ["idle", "listening", "thinking", "speaking"]
    tiles = []
    for (t, state, u, m), name in zip(scenes, names):
        frame = disp.render(t, state, u, m)
        big = cv2.resize(frame, (192 * 3, 192 * 3), interpolation=cv2.INTER_NEAREST)
        cv2.putText(big, name, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 2, cv2.LINE_AA)
        tiles.append(big)
        cv2.imwrite(f"{out_dir}/conv_{name}.png", big)

    strip = cv2.hconcat(tiles)
    cv2.imwrite(f"{out_dir}/conv_all.png", strip)
    print(f"wrote {out_dir}/conv_all.png and 4 single frames")
