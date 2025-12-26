
# Pro AI Video Editor

A professional-grade automated video editing application built with Python, Streamlit, and MoviePy.

## Features

- **Automated Silence Removal**: Detects and removes silent segments based on a configurable sensitivity threshold.
- **Smart Logic**: Keeps only meaningful segments (ignores clips < 1 second).
- **Audio Engineering**:
  - Automatic Audio Normalization.
  - Cross-fades (0.05s) on every cut to prevent audio clicks.
- **Color Grading**: "Pop" effect (Saturation +20%, Contrast +10%) applied to all clips.
- **Cinematography**:
  - 1-second Fade In at the start.
  - 1-second Fade Out at the end.
- **Transitions**: Adds a sound effect (`whoosh.mp3`) at every cut point.
- **Batch Processing**: Recursively scans folders, ignoring the output folder `Pro_Edited_Results`.
- **GUI**: User-friendly interface to control settings and monitor progress.

## Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: Requires `ffmpeg` installed on the system.*

2. Place a `whoosh.mp3` file in the root directory (optional, for transitions).

## Usage

1. Run the application:
   ```bash
   streamlit run editapp.py
   ```
2. Enter the "Root Folder Path" containing your videos.
3. Adjust settings in the sidebar (Sensitivity, Grading, Normalization).
4. Click "Start Batch Processing".
5. Results will be saved in a `Pro_Edited_Results` folder inside your root folder.

## Compatibility

Works with both MoviePy v1.x and v2.x.
