# TherMirror — Project Brief / Prompt

I'm building **TherMirror**, an interactive "soul mirror": a mirror that shows a live
thermal-camera-style visualization of whoever stands in front of it, and has a
conversational AI personality you can talk to like a real person.

## The concept
A two-way mirror with an LED display behind it. When you stand in front, you see a
**thermal-camera-style image of yourself** glowing inside the mirror. You can **talk to it**
and it talks back — a warm, witty personality ("Mira") that sees you and answers questions.

## Physical build
- **Display:** 27× HUB75 RGB LED panels, 64×64 each, arranged **3 wide × 9 tall = 192×576 px**,
  mounted behind a **two-way mirror** so the glowing image appears within your reflection.
- **Brain + LED driver:** **Raspberry Pi 5 (8 GB)** with an **Adafruit RGB Matrix Bonnet**
  (wired as 3 parallel chains of 9 panels). Driven via `rpi-rgb-led-matrix`.
- **Power:** dedicated high-current **5 V PSU(s)** for the panels (content is mostly dark, but
  size for headroom), with power injection + fuses. Pi powered separately (27 W USB-C).
- **Vision camera:** Raspberry Pi Camera Module 3 or a USB webcam (for the AI to see).
- **Thermal sensor:** **MLX90640** (32×24 IR array over I2C) for the real thermal image
  (better than the AMG8833 8×8). The code also supports AMG8833.
- **Audio:** USB / far-field microphone + USB speaker (Pi 5 has no 3.5 mm jack).
- **Network:** internet required for the cloud AI (Ethernet preferred for low-latency voice).

## What it shows on screen
- A **thermal-camera look**: the person rendered as a heat signature in a warm thermal colormap.
- **Objects** the person holds/brings are shown in a **different color** from the person, still
  in "thermal mode," so they stand out.
- **Only the clean thermal image** on screen — no bounding boxes, labels, or debug clutter.

## The voice assistant ("Mira")
- **Real-time voice conversation** (you speak, she replies with a natural voice) — interruptible,
  feels like talking to a real person, with a warm/curious/slightly witty personality.
- **She sees through the camera** — can comment on what you're wearing or holding.
- **Live data via web search:** weather, news, stock prices/news, general updates.
- **On-screen transcription** (subtitles) of both sides of the conversation.
- **Limits:** no access to private accounts (portfolio/orders) — she asks for the details
  (e.g. which stocks you own) and then looks up prices/news.

## Tech stack (decided)
- **Python**, **OpenCV** for the camera pipeline, rendering, and (in simulation) object masks.
- **MediaPipe Selfie Segmentation** for the person mask (used in the PC simulation).
- **OpenCV background subtraction** for object masks in simulation (real sensor makes this moot).
- **MLX90640** for real per-pixel heat on the hardware (no segmentation needed for thermal).
- **Google Gemini Live API** (`gemini-2.5-flash-native-audio`) for the voice conversation;
  **Gemini + Google Search grounding** for live data. API key from an env var (free tier).
- `rpi-rgb-led-matrix` to drive the panels; a `cv2` preview window as fallback off-Pi.
- Requires **numpy < 2** (MediaPipe compatibility).

## Current status
- Working **simulation on a MacBook**: `led_mirror_entity.py` (thermal render + person/object
  coloring) and `voice_assistant.py` (Gemini Live voice + on-screen transcription).
- On the Mac the thermal is *simulated* from the camera; the real MLX90640 path runs on the Pi.

## What I want help with next
(pick one) → Port to the Raspberry Pi and drive the real LED panels · Add MLX90640 real-thermal
support · Set up auto-start on boot · Tune the visuals · Build the mirror enclosure.
