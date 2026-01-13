# 👻 GHOST_AUTOVID_WORLD v2.1

**The World's Best Automated Video Production System.**
*Now with AI Intelligence & GPU Acceleration.*

---

## 🚀 Features

### 🎬 Auto-Pilot
- **One-Click Production:** Scan a folder, analyze footage, and generate a fully edited video automatically.
- **Smart Editing:** Removes silence, inserts B-Roll, and auto-zooms on faces.
- **Audio Mastering:** Automatically normalizes audio to broadcast standards (-14 LUFS).

### 🎛️ Advanced Editor
- **Fine-Tune Controls:**
  - **Silence Sensitivity:** Adjust how aggressively silence is cut.
  - **Zoom Factor:** Control the intensity of face zooms.
  - **B-Roll Frequency:** Decide how often overlay footage appears.
- **GPU Acceleration:** Detects NVIDIA (NVENC) and Intel (QSV) hardware to speed up rendering by 5-10x.

### 🤖 AI Intelligence Center
- **Offline Transcription:** Uses `faster-whisper` to generate subtitles (.srt) locally.
- **No API Keys:** Runs entirely on your machine. No cloud costs.

### 🔊 Audio Lab
- **Loudness Normalization:** Fix audio levels for YouTube, Spotify, or Broadcast.
- **Batch Processing:** Handle video or audio files.

### 📸 Photo Studio
- **Thumbnail Generator:** Extract high-quality frames from video.
- **Manual Selection:** Scrub through video to find the perfect moment.
- **Text Overlay:** Auto-add titles to thumbnails.

---

## 🛠️ Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-repo/ghost_autovid_world.git
   cd ghost_autovid_world
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   *Note: This project requires Python 3.8+.*

3. **Install FFmpeg**
   - **Windows:** Download from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/) and add to PATH.
   - **Mac:** `brew install ffmpeg`
   - **Linux:** `sudo apt install ffmpeg`

---

## 🎮 Usage

### GUI Mode (Recommended)
Launch the advanced dashboard:
```bash
streamlit run ghost_autovid_world/app.py
```

### Headless Mode (Server/CLI)
Run the auto-pilot on a specific folder:
```bash
python ghost_autovid_world/main.py /path/to/footage_folder
```

---

## ⚡ Hardware Acceleration

The system automatically detects your hardware capabilities:
- **NVIDIA:** Uses `h264_nvenc` (Preset: p4)
- **Intel:** Uses `h264_qsv` (Preset: medium)
- **Apple Silicon:** Uses `h264_videotoolbox`
- **CPU:** Fallback to `libx264`

Check the "System Status" sidebar in the GUI to see active acceleration.

---

## ⚠️ Troubleshooting

- **"FFmpeg not found":** Ensure FFmpeg is installed and accessible in your terminal.
- **"faster_whisper not found":** Run `pip install faster-whisper`.
- **Render crashes:** Reduce thread count or check available RAM.

---

**License:** MIT
**Author:** Jules (AI Agent)
