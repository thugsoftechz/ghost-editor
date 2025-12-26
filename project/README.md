
# Jules: Automatic Video Decision System

**Jules** is a fully autonomous video editing engine. It takes a folder of raw footage and produces an upload-ready video without any user interaction.

## Core Capabilities

*   **Recursive Scanning:** Finds all videos in subfolders.
*   **Offline Analysis:** Uses Computer Vision (OpenCV) and Audio Processing (NumPy/SciPy) to understand content.
*   **Auto-Selection:** Intelligently picks the Main video and B-Roll clips.
*   **Auto-Editing:** Removes silence, centers faces, and inserts B-Roll.
*   **Audio Mastering:** Normalizes audio to -14 LUFS.
*   **Captioning:** Generates SRT subtitles using faster-whisper.

## Requirements

*   Python 3.8+
*   moviepy
*   opencv-python
*   mediapipe
*   numpy
*   scipy
*   librosa
*   pyloudnorm
*   faster-whisper

## Usage

```bash
python project/main.py /path/to/raw_footage
```

## Output

The system generates a `project/output` folder containing:

*   `final_video.mp4`: The edited video.
*   `captions.srt`: Subtitles.
*   `thumbnail.jpg`: A representative frame.
*   `metadata.json`: Processing details.
*   `upload_ready.txt`: Checklist.

## Architecture

*   `engine/folder_scanner.py`: File discovery.
*   `engine/analyzer.py`: Audio/Visual metrics extraction.
*   `engine/selector.py`: Decision logic (Main vs B-Roll).
*   `engine/auto_editor.py`: Cutting and compositing.
*   `engine/audio_engine.py`: Mastering.
*   `engine/output_manager.py`: Export handling.

**Note:** This system is designed to run offline and unattended.
