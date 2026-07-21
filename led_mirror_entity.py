"""
LED Mirror "Entity" - scheletro pipeline
=========================================

Pipeline:
  1. Cattura camera normale (no termocamera)
  2. Segmentazione sagoma persona (background subtraction)
  3. Estrazione contorno (Canny)
  4. Colorazione gradiente blu -> viola sul contorno
  5. Glow neon (blur gaussiano sommato sopra il contorno netto)
  6. Overlay widget (stock / risposta AI) agli angoli
  7. Output su pannello LED HUB75 (rpi-rgb-led-matrix), con fallback
     a preview a schermo (cv2.imshow) se la libreria hardware non e' disponibile
     -> utile per sviluppare/testare l'effetto su PC prima di portarlo sul Pi

Dipendenze:
  pip install opencv-python numpy pillow --break-system-packages
  (sul Raspberry Pi, per l'output reale: https://github.com/hzeller/rpi-rgb-led-matrix)
"""

import time
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

PANEL_WIDTH = 192          # risoluzione pannello LED (cambiare in base al tuo assemblaggio)
PANEL_HEIGHT = 576
CAMERA_INDEX = 0

# gradiente colore contorno: viola in alto -> blu in basso
COLOR_TOP = np.array([200, 60, 140])     # BGR: viola
COLOR_BOTTOM = np.array([255, 90, 40])   # BGR: blu

EDGE_THRESH1, EDGE_THRESH2 = 60, 150     # soglie Canny
GLOW_BLUR_KSIZE = 15                     # dimensione blur per il glow (dispari)
GLOW_INTENSITY = 0.9                     # quanto peso ha il layer sfocato nel blend

WIDGET_FONT_SIZE = 14


# ---------------------------------------------------------------------------
# 1-2. Cattura + segmentazione sagoma
# ---------------------------------------------------------------------------

class SilhouetteExtractor:
    """Estrae la maschera binaria della/e persona/e in scena via background subtraction.
    Assunzione: camera fissa, sfondo relativamente statico (stanza normale)."""

    def __init__(self):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=25, detectShadows=False
        )

    def get_mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        mask = self.bg_subtractor.apply(frame_bgr)
        # pulizia morfologica: rimuove rumore, chiude buchi nella sagoma
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        return mask


# ---------------------------------------------------------------------------
# 3-4-5. Contorno + colorazione gradiente + glow neon
# ---------------------------------------------------------------------------

def extract_colored_edges(mask: np.ndarray) -> np.ndarray:
    """Da maschera binaria a immagine BGR con solo i contorni, colorati
    con gradiente verticale viola -> blu."""
    edges = cv2.Canny(mask, EDGE_THRESH1, EDGE_THRESH2)

    h, w = edges.shape
    # gradiente verticale precalcolato (h x 3)
    t = np.linspace(0, 1, h).reshape(h, 1)
    gradient = (COLOR_TOP * (1 - t) + COLOR_BOTTOM * t).astype(np.uint8)  # (h, 3)
    gradient_img = np.repeat(gradient[:, np.newaxis, :], w, axis=1)      # (h, w, 3)

    colored = np.zeros((h, w, 3), dtype=np.uint8)
    colored[edges > 0] = gradient_img[edges > 0]
    return colored


def add_neon_glow(colored_edges: np.ndarray) -> np.ndarray:
    """Aggiunge un alone diffuso (glow) sommando una versione sfocata
    sopra il contorno netto, effetto 'entita' luminosa."""
    k = GLOW_BLUR_KSIZE | 1  # forza dispari
    blurred = cv2.GaussianBlur(colored_edges, (k, k), 0)
    blurred2 = cv2.GaussianBlur(colored_edges, (k * 2 + 1, k * 2 + 1), 0)

    glow = cv2.addWeighted(blurred, GLOW_INTENSITY, blurred2, 0.5, 0)
    result = cv2.add(colored_edges, glow)  # somma additiva, satura a 255 (buono per neon)
    return result


# ---------------------------------------------------------------------------
# 6. Widget overlay (stock / risposta AI)
# ---------------------------------------------------------------------------

class Widget:
    def __init__(self, text: str, position: str, ttl_seconds: float, color=(255, 255, 255)):
        self.text = text
        self.position = position  # "top-left" | "top-right" | "bottom"
        self.expires_at = time.time() + ttl_seconds
        self.color = color

    @property
    def expired(self) -> bool:
        return time.time() > self.expires_at


class WidgetManager:
    def __init__(self):
        self.widgets: list[Widget] = []
        try:
            self.font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", WIDGET_FONT_SIZE
            )
        except OSError:
            self.font = ImageFont.load_default()

    def show(self, text: str, position: str = "top-right", ttl_seconds: float = 8.0, color=(255, 255, 255)):
        self.widgets = [w for w in self.widgets if w.position != position]  # sostituisce eventuale widget nella stessa zona
        self.widgets.append(Widget(text, position, ttl_seconds, color))

    def render(self, base_img: Image.Image) -> Image.Image:
        self.widgets = [w for w in self.widgets if not w.expired]
        if not self.widgets:
            return base_img

        draw = ImageDraw.Draw(base_img, "RGBA")
        w, h = base_img.size
        margin = 4

        for widget in self.widgets:
            bbox = draw.textbbox((0, 0), widget.text, font=self.font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

            if widget.position == "top-left":
                xy = (margin, margin)
            elif widget.position == "top-right":
                xy = (w - tw - margin, margin)
            elif widget.position == "bottom":
                xy = ((w - tw) // 2, h - th - margin)
            else:
                xy = (margin, margin)

            # sfondo semi-trasparente per leggibilita' sopra lo sfondo nero
            pad = 3
            draw.rectangle(
                [xy[0] - pad, xy[1] - pad, xy[0] + tw + pad, xy[1] + th + pad],
                fill=(0, 0, 0, 140),
            )
            draw.text(xy, widget.text, font=self.font, fill=widget.color)

        return base_img


# ---------------------------------------------------------------------------
# 7. Output: HUB75 reale (rpi-rgb-led-matrix) con fallback a preview finestra
# ---------------------------------------------------------------------------

class PanelOutput:
    def __init__(self, width: int, height: int):
        self.width, self.height = width, height
        self.hardware = None
        try:
            from rgbmatrix import RGBMatrix, RGBMatrixOptions  # disponibile solo su Raspberry Pi
            options = RGBMatrixOptions()
            options.rows = 32
            options.cols = 64
            options.chain_length = 9   # da adattare al numero di moduli/catena reale
            options.parallel = 1
            options.hardware_mapping = "regular"
            self.hardware = RGBMatrix(options=options)
            print("[output] HUB75 hardware inizializzato")
        except Exception:
            print("[output] libreria rgbmatrix non trovata -> modalita' preview su schermo")

    def show(self, pil_img: Image.Image):
        if self.hardware is not None:
            self.hardware.SetImage(pil_img.convert("RGB"))
        else:
            frame = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
            preview = cv2.resize(frame, (self.width * 2, self.height * 2), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("LED Mirror Preview", preview)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    extractor = SilhouetteExtractor()
    widgets = WidgetManager()
    output = PanelOutput(PANEL_WIDTH, PANEL_HEIGHT)

    print("Premi 'q' per uscire, 's' per simulare un widget stock, 'a' per simulare risposta AI (in modalita' preview)")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        mask = extractor.get_mask(frame)
        colored_edges = extract_colored_edges(mask)
        entity = add_neon_glow(colored_edges)

        # resize al formato pannello e conversione in PIL per il compositing widget
        entity_resized = cv2.resize(entity, (PANEL_WIDTH, PANEL_HEIGHT))
        entity_rgb = cv2.cvtColor(entity_resized, cv2.COLOR_BGR2RGB)
        panel_img = Image.fromarray(entity_rgb)

        panel_img = widgets.render(panel_img)
        output.show(panel_img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            widgets.show("AAPL  212.40", position="top-right", ttl_seconds=6, color=(120, 200, 255))
        elif key == ord("a"):
            widgets.show("Fuori piove", position="bottom", ttl_seconds=6, color=(255, 255, 255))

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
