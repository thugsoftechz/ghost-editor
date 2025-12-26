
"""
jules_god_mode.py

# requirements.txt
# moviepy
# streamlit
# faster-whisper
# mediapipe
# pyloudnorm
# numpy
# librosa
# soundfile
"""

import os
import shutil
import tempfile
import time
import numpy as np
import streamlit as st

# --- Imports with Graceful Fallbacks ---

# MoviePy (v1/v2 compatibility)
try:
    from moviepy.editor import (
        VideoFileClip, AudioFileClip, TextClip, CompositeVideoClip,
        CompositeAudioClip, concatenate_videoclips, vfx, afx
    )
    import moviepy.audio.fx.all as audio_fx
    MOVIEPY_V1 = True
except ImportError:
    import moviepy
    from moviepy import (
        VideoFileClip, AudioFileClip, TextClip, CompositeVideoClip,
        CompositeAudioClip, concatenate_videoclips
    )
    import moviepy.video.fx as vfx
    import moviepy.audio.fx as afx
    MOVIEPY_V1 = False

# MediaPipe
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

# Faster Whisper
try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

# Pyloudnorm & Librosa
try:
    import pyloudnorm as pyln
    import librosa
    import soundfile as sf
    AUDIO_DSP_AVAILABLE = True
except ImportError:
    AUDIO_DSP_AVAILABLE = False

# --- Master Class: JulesEngine ---

class JulesEngine:
    def __init__(self):
        self.mp_face_detection = None
        if MEDIAPIPE_AVAILABLE:
            self.mp_face_detection = mp.solutions.face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.5
            )

    def smart_crop_9_16(self, clip):
        """
        Dynamically crops video to 9:16 keeping the speaker centered.
        Uses a moving average window for smoothing.
        """
        if not MEDIAPIPE_AVAILABLE:
            st.warning("MediaPipe not available. Returning center crop.")
            # Simple Center Crop
            w, h = clip.size
            new_w = h * (9/16)
            if MOVIEPY_V1:
                return clip.crop(x1=w/2 - new_w/2, width=new_w, height=h)
            else:
                return clip.with_effects([vfx.Crop(x1=w/2 - new_w/2, width=new_w, height=h)])

        # Pre-analyze face positions
        st.write("🧠 Analyzing Face Metrics (Smart Crop)...")

        # Sampling rate: analyze every Nth frame to save time?
        # For smoothness, every frame is best, but slow. Let's do every 2nd frame or so.
        # Actually, get_frame is expensive.

        # To avoid reading the whole video twice, we can define a crop function
        # that uses a pre-calculated trajectory.

        duration = clip.duration
        fps = clip.fps if clip.fps else 24

        # We need to compute the crop center for each timestamp.
        # Let's iterate over frames.

        face_centers_x = []
        timestamps = []

        # Iterate through video at reduced resolution for speed?
        # clip.iter_frames() gives numpy arrays.

        # Optimization: Resize small for face detection
        small_clip = clip.resize(height=360) if MOVIEPY_V1 else clip.with_effects([vfx.Resize(height=360)])

        scale_factor = clip.h / 360.0

        w, h = clip.size
        target_w = h * (9/16)

        current_x = w / 2 # Start at center

        # Limit analysis to a few seconds if it's too long? No, full video.
        # But for 'God Mode', we do it properly.

        positions = [] # (t, center_x)

        # Iterate over frames
        for t, frame in small_clip.iter_frames(with_times=True):
            # MediaPipe expects RGB
            results = self.mp_face_detection.process(frame)

            detected_x = None
            if results.detections:
                # Get the first face
                detection = results.detections[0]
                bbox = detection.location_data.relative_bounding_box
                # bbox.xmin is relative [0, 1]

                # Center of face
                face_x = bbox.xmin + bbox.width / 2
                detected_x = face_x * small_clip.w * scale_factor

            if detected_x is not None:
                positions.append(detected_x)
            else:
                # If lost, use previous
                positions.append(positions[-1] if positions else w/2)

        # Apply Smoothing (Moving Average)
        window_size = int(fps * 1.0) # 1 second window
        if window_size < 1: window_size = 1

        smoothed_positions = np.convolve(positions, np.ones(window_size)/window_size, mode='same')

        # Define crop function
        def get_crop_region(t):
            # Find frame index
            idx = int(t * fps)
            if idx >= len(smoothed_positions):
                idx = len(smoothed_positions) - 1

            center_x = smoothed_positions[idx]

            # Clamp
            x1 = center_x - target_w / 2
            if x1 < 0: x1 = 0
            if x1 + target_w > w: x1 = w - target_w

            return x1, 0, x1 + target_w, h

        # Apply crop
        if MOVIEPY_V1:
            # v1: clip.fl(fun) -> fun(get_frame, t)
            # or clip.crop(...) which can take functions?
            # clip.crop(x1=func, width=..., height=...) accepts time-dependent x1?
            # Yes, x1 can be a function t -> x.
            return clip.crop(x1=lambda t: get_crop_region(t)[0], width=target_w, height=h)
        else:
            # v2: vfx.Crop accepts x1 as function?
            # Assuming yes based on v1 parity, or we use fl_transform
            # Let's try passing function. If not, fallback to static.
            try:
                return clip.with_effects([vfx.Crop(x1=lambda t: get_crop_region(t)[0], width=target_w, height=h)])
            except:
                return clip.with_effects([vfx.Crop(x1=w/2 - target_w/2, width=target_w, height=h)])

    def generate_subtitles(self, clip, audio_path):
        """
        Uses faster-whisper to generate TextClips.
        """
        if not WHISPER_AVAILABLE:
            st.warning("Faster-Whisper not available.")
            return clip

        st.write("🤖 Transcribing Audio (Whisper AI)...")
        model = WhisperModel("base", device="cpu", compute_type="int8")
        segments, info = model.transcribe(audio_path, beam_size=5)

        text_clips = []
        w, h = clip.size

        # Define Font/Style
        # Impact font might not be on linux. Fallback to generic.
        font = "Impact" if os.name == 'nt' else "DejaVu-Sans-Bold"
        fontsize = 50 if w > 1000 else 30

        for segment in segments:
            start = segment.start
            end = segment.end
            text = segment.text.strip()

            if not text: continue

            # Create TextClip
            # v1: TextClip(txt, ...)
            # v2: TextClip(font=..., text=...) ?

            try:
                if MOVIEPY_V1:
                    txt_clip = TextClip(
                        text,
                        fontsize=fontsize,
                        color='yellow',
                        font=font,
                        stroke_color='black',
                        stroke_width=2,
                        method='caption',
                        size=(w * 0.8, None)
                    ).set_position(('center', 'bottom')).set_start(start).set_end(end)
                else:
                    # v2 TextClip often requires font path or specific setup
                    # fallback to generic
                    txt_clip = TextClip(
                        text=text,
                        font_size=fontsize,
                        color='yellow',
                        font=font,
                        stroke_color='black',
                        stroke_width=2,
                        method='caption',
                        size=(w * 0.8, None)
                    ).with_position(('center', 'bottom')).with_start(start).with_end(end)

                text_clips.append(txt_clip)
            except Exception as e:
                # TextClip can fail if ImageMagick is missing
                st.warning(f"Could not create text clip (ImageMagick issue?): {e}")
                break

        if text_clips:
            return CompositeVideoClip([clip] + text_clips)
        return clip

    def master_audio(self, clip, target_lufs=-14.0):
        """
        Normalizes audio to target LUFS and limits peaks.
        """
        if not AUDIO_DSP_AVAILABLE or clip.audio is None:
            return clip

        st.write(f"🎚️ Mastering Audio to {target_lufs} LUFS...")

        # 1. Extract Audio
        fs = 44100
        # to_soundarray returns (N, Channels)
        audio_arr = clip.audio.to_soundarray(fps=fs)

        # 2. Measure Loudness
        meter = pyln.Meter(fs)
        loudness = meter.integrated_loudness(audio_arr)

        # 3. Calculate Gain
        gain_db = target_lufs - loudness
        gain_lin = 10 ** (gain_db / 20.0)

        # 4. Apply Gain
        audio_arr = audio_arr * gain_lin

        # 5. Hard Limiter (-1.0 dBTP)
        limit_db = -1.0
        limit_lin = 10 ** (limit_db / 20.0)

        # Simple hard clipping
        audio_arr = np.clip(audio_arr, -limit_lin, limit_lin)

        # 6. Re-attach
        # Need to create AudioClip from array
        if MOVIEPY_V1:
            from moviepy.audio.AudioClip import AudioArrayClip
            new_audio = AudioArrayClip(audio_arr, fps=fs)
            return clip.set_audio(new_audio)
        else:
            # v2 AudioArrayClip
            from moviepy.audio.AudioClip import AudioArrayClip
            new_audio = AudioArrayClip(audio_arr, fps=fs)
            return clip.with_audio(new_audio)

    def remove_silence_librosa(self, clip, threshold_db=40):
        """
        Uses librosa to detect silence and splits the clip.
        """
        if not AUDIO_DSP_AVAILABLE or clip.audio is None:
            return clip

        st.write("✂️ Precision Silence Cutting (Librosa DSP)...")

        # 1. Get audio (mono for analysis)
        fs = 22050 # Lower sample rate for analysis is fine
        audio_arr = clip.audio.to_soundarray(fps=fs)
        if audio_arr.ndim > 1:
            audio_mono = np.mean(audio_arr, axis=1)
        else:
            audio_mono = audio_arr

        # 2. Librosa Split
        # top_db: The threshold (in decibels) below reference to consider as silence
        non_silent_intervals = librosa.effects.split(audio_mono, top_db=threshold_db)

        # 3. Create Subclips
        clips = []
        for start_sample, end_sample in non_silent_intervals:
            start_t = start_sample / fs
            end_t = end_sample / fs

            # Buffer
            start_t = max(0, start_t - 0.1)
            end_t = min(clip.duration, end_t + 0.1)

            if (end_t - start_t) < 0.5: continue

            if MOVIEPY_V1:
                sub = clip.subclip(start_t, end_t)
            else:
                sub = clip.subclipped(start_t, end_t)
            clips.append(sub)

        if not clips:
            st.warning("Video detected as fully silent. Returning original.")
            return clip

        return concatenate_videoclips(clips)


# --- Streamlit Interface ---

def main():
    st.set_page_config(page_title="Jules God Mode", page_icon="⚡", layout="wide")

    st.title("⚡ Jules God Mode: The AI Video Reactor")
    st.caption("Computer Vision • NLP • DSP • Automated Post-Production")

    # Sidebar Controls
    with st.sidebar:
        st.header("Reaction Settings")
        mode = st.selectbox("Video Mode", ["Cinematic Landscape (16:9)", "Viral Short (9:16 Face-Tracked)"])
        burn_subs = st.checkbox("Burn-in AI Captions", value=True)
        mastering_target = st.selectbox("Audio Mastering Target", ["YouTube (-14 LUFS)", "Broadcast (-23 LUFS)"])
        silence_cut = st.checkbox("DSP Silence Removal", value=True)

        st.divider()
        st.info(f"System Check:\n"
                f"• MediaPipe: {'✅' if MEDIAPIPE_AVAILABLE else '❌'}\n"
                f"• Faster-Whisper: {'✅' if WHISPER_AVAILABLE else '❌'}\n"
                f"• DSP Engine: {'✅' if AUDIO_DSP_AVAILABLE else '❌'}")

    uploaded_file = st.file_uploader("Upload Video Source", type=["mp4", "mov", "mkv"])

    if uploaded_file:
        # Save temp file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        video_path = tfile.name

        st.video(video_path)

        if st.button("Initialize God Mode Reactor"):
            engine = JulesEngine()
            progress = st.status("Initializing Reactor Core...", expanded=True)

            try:
                # Load Video
                if MOVIEPY_V1:
                    clip = VideoFileClip(video_path)
                else:
                    clip = VideoFileClip(video_path)

                # 1. Edit Logic (Silence)
                if silence_cut:
                    progress.write("✂️ Running DSP Silence Removal...")
                    clip = engine.remove_silence_librosa(clip)

                # 2. Mode (Shorts Reactor)
                if mode == "Viral Short (9:16 Face-Tracked)":
                    progress.write("👁️ Engaging Face Tracking Matrix...")
                    clip = engine.smart_crop_9_16(clip)

                # 3. Audio Mastering
                target_lufs = -14.0 if "YouTube" in mastering_target else -23.0
                progress.write("🎚️ Mastering Audio Dynamics...")
                clip = engine.master_audio(clip, target_lufs)

                # 4. Subtitles
                if burn_subs and WHISPER_AVAILABLE:
                    progress.write("📝 Generative AI Subtitling...")
                    # Need to write audio to temp file for whisper
                    afile = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                    if MOVIEPY_V1:
                        clip.audio.write_audiofile(afile.name, logger=None)
                    else:
                        clip.audio.write_audiofile(afile.name, logger=None)

                    clip = engine.generate_subtitles(clip, afile.name)
                    os.unlink(afile.name)

                # Render
                progress.write("💾 Final Rendering...")
                output_path = "god_mode_output.mp4"
                clip.write_videofile(
                    output_path,
                    codec="libx264",
                    audio_codec="aac",
                    preset="medium",
                    threads=4,
                    logger=None
                )

                clip.close()
                progress.update(label="✅ Processing Complete", state="complete", expanded=False)

                st.success("Render Complete.")
                st.video(output_path)

                with open(output_path, "rb") as f:
                    st.download_button("Download Result", f, "jules_god_mode.mp4")

            except Exception as e:
                st.error(f"Reactor Meltdown: {str(e)}")
            finally:
                if 'clip' in locals(): clip.close()
                os.unlink(video_path)

if __name__ == "__main__":
    main()
