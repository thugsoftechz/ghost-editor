
"""
jules_insane.py

# requirements.txt
# moviepy
# streamlit
# faster-whisper
# mediapipe
# pyloudnorm
# numpy
# librosa
# soundfile
# openai
# opencv-python
# scipy
"""

import os
import shutil
import tempfile
import time
import json
import random
import re
import numpy as np
import streamlit as st
import cv2

# --- Robust Imports ---

try:
    from moviepy.editor import (
        VideoFileClip, AudioFileClip, TextClip, CompositeVideoClip,
        CompositeAudioClip, concatenate_videoclips, vfx, afx, ImageClip
    )
    import moviepy.audio.fx.all as audio_fx
    MOVIEPY_V1 = True
except ImportError:
    import moviepy
    from moviepy import (
        VideoFileClip, AudioFileClip, TextClip, CompositeVideoClip,
        CompositeAudioClip, concatenate_videoclips, ImageClip
    )
    import moviepy.video.fx as vfx
    import moviepy.audio.fx as afx
    MOVIEPY_V1 = False

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

try:
    import pyloudnorm as pyln
    import librosa
    import soundfile as sf
    AUDIO_DSP_AVAILABLE = True
except ImportError:
    AUDIO_DSP_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# --- Modules ---

class ViralBrain:
    """Module 1: LLM-driven Content Intelligence"""
    def __init__(self, api_key=None):
        self.api_key = api_key
        if api_key and OPENAI_AVAILABLE:
            openai.api_key = api_key

    def transcribe_full(self, audio_path):
        if not WHISPER_AVAILABLE:
            return None
        st.write("🧠 ViralBrain: Running Faster-Whisper (Full Scan)...")
        # 'base' model for speed/quality balance
        model = WhisperModel("base", device="cpu", compute_type="int8")
        segments, info = model.transcribe(audio_path, word_timestamps=True)
        return list(segments)

    def analyze_viral_moments(self, segments, aggressive_factor=0.5):
        """
        Send transcript to LLM to find viral hooks.
        Fallback to heuristic cutting if LLM fails.
        """
        full_text = " ".join([s.text for s in segments])

        prompt = (
            f"Analyze the transcript. Identify the 3 most viral, funny, or insightful 60-second segments. "
            f"Return ONLY a JSON list of objects with keys 'start_time' and 'end_time' (in seconds). "
            f"Transcript: {full_text[:4000]}..."
        )

        viral_ranges = []
        llm_success = False

        if self.api_key and OPENAI_AVAILABLE:
            try:
                st.write("🧠 ViralBrain: Consulting LLM Oracle...")
                response = openai.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a viral video editor. Respond only in JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=300
                )
                content = response.choices[0].message.content

                # Simple parsing logic
                # Look for patterns like {"start_time": 10, "end_time": 70}
                matches = re.findall(r'"start_time":\s*([\d.]+),\s*"end_time":\s*([\d.]+)', content)
                if matches:
                    for start, end in matches:
                        viral_ranges.append((float(start), float(end)))
                    llm_success = True

            except Exception as e:
                st.warning(f"ViralBrain LLM disconnected ({str(e)}). Engaging Heuristic Mode.")

        if not llm_success:
            # Heuristic Fallback
            duration = segments[-1].end
            count = 3
            clip_len = 60 * (1.1 - aggressive_factor) # Adjust length based on factor

            for _ in range(count):
                start = random.uniform(0, max(0, duration - clip_len))
                end = min(duration, start + clip_len)
                viral_ranges.append((start, end))

        return viral_ranges

class RetentionEngine:
    """Module 2: Dynamic Zooming & Attention"""
    def __init__(self):
        self.mp_face = None
        if MEDIAPIPE_AVAILABLE:
            self.mp_face = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

    def apply_punch_in(self, clip, mode="Gen-Z"):
        """Randomly scales clip 100% or 115%."""
        if mode != "Gen-Z":
            return clip

        scale = random.choice([1.0, 1.15])
        if scale == 1.0:
            return clip

        if MOVIEPY_V1:
            return clip.resize(scale)
        else:
            return clip.with_effects([vfx.Resize(scale)])

    def apply_ken_burns(self, clip):
        """Slow zoom towards face (1.0 -> 1.1)."""
        if not MEDIAPIPE_AVAILABLE:
            # Fallback linear center zoom
            def zoom_func(t):
                return 1.0 + 0.1 * (t / clip.duration)
            if MOVIEPY_V1:
                return clip.resize(zoom_func)
            else:
                return clip.with_effects([vfx.Resize(zoom_func)])

        # Analyze first frame to find face center
        w, h = clip.size
        center_x, center_y = w/2, h/2

        try:
            # Get first frame
            if MOVIEPY_V1:
                frame = clip.get_frame(0)
            else:
                frame = clip.get_frame(0) # v2 might change, usually same

            results = self.mp_face.process(frame)
            if results.detections:
                bbox = results.detections[0].location_data.relative_bounding_box
                center_x = (bbox.xmin + bbox.width/2) * w
                center_y = (bbox.ymin + bbox.height/2) * h
        except:
            pass

        # Ken Burns: Zoom towards (center_x, center_y)
        # We need a scroll/crop function.
        # Simple implementation: Crop getting smaller around target point, then resize to original.

        def crop_zoom(get_frame, t):
            # Scale goes 1.0 -> 1.1
            scale = 1.0 + 0.1 * (t / clip.duration)
            # New Width/Height
            new_w = w / scale
            new_h = h / scale

            # Center of crop shifts towards target
            # t=0: center at w/2, h/2. t=end: center approaches target?
            # Standard Ken Burns: Center stays fixed on target?
            # Or we interpolate center from image center to face center?
            # Let's keep center on face.

            x1 = center_x - new_w / 2
            y1 = center_y - new_h / 2

            # Clamp
            x1 = max(0, min(x1, w - new_w))
            y1 = max(0, min(y1, h - new_h))

            img = get_frame(t)
            # Crop
            # numpy slicing: [y:y+h, x:x+w]
            cropped = img[int(y1):int(y1+new_h), int(x1):int(x1+new_w)]

            # Resize back to w, h
            return cv2.resize(cropped, (w, h))

        if MOVIEPY_V1:
            return clip.fl(crop_zoom)
        else:
            # v2 fl is usually fl(func, apply_to=[])
            return clip.fl(crop_zoom)

class ContentAugmentor:
    """Module 3 & 4: B-Roll & Karaoke"""
    def __init__(self, assets_dir="assets"):
        self.assets_dir = assets_dir
        os.makedirs(assets_dir, exist_ok=True)

    def inject_b_roll(self, clip, segments, offset):
        """Overlay B-Roll based on keywords. offset is start time of subclip in original video."""
        # Flat list of words
        words = []
        for s in segments:
            if hasattr(s, 'words'):
                words.extend(s.words)

        final_clip = clip

        # Scan words
        for word in words:
            # Word absolute start
            abs_start = word.start
            # Relative start in subclip
            rel_start = abs_start - offset

            # Check if word is inside this clip
            if rel_start < 0 or rel_start > clip.duration:
                continue

            term = word.word.strip().lower().strip(".,!?")
            asset_path = os.path.join(self.assets_dir, f"{term}.mp4")

            if os.path.exists(asset_path):
                try:
                    b_roll = VideoFileClip(asset_path).resize(height=clip.h)
                    # Duration: until end of word + 1s buffer
                    dur = (word.end - word.start) + 1.5
                    if rel_start + dur > clip.duration:
                        dur = clip.duration - rel_start

                    b_roll = b_roll.set_position("center").set_start(rel_start).set_duration(dur)
                    final_clip = CompositeVideoClip([final_clip, b_roll])
                except:
                    pass

        return final_clip

    def generate_karaoke_subs(self, clip, segments, offset):
        """Generates Highlighted TextClips."""
        if not segments: return clip

        text_clips = []
        w, h = clip.size
        fontsize = 50
        font = "Impact" if os.name == 'nt' else "DejaVu-Sans-Bold"

        all_words = []
        for s in segments:
            if hasattr(s, 'words'):
                all_words.extend(s.words)

        for i, word_obj in enumerate(all_words):
            rel_start = word_obj.start - offset
            rel_end = word_obj.end - offset

            if rel_end < 0 or rel_start > clip.duration:
                continue

            color = 'green' if i % 2 == 0 else 'red'
            word_text = word_obj.word.strip()

            try:
                # v1/v2 compatibility handled by try/except usually, or explicity check
                if MOVIEPY_V1:
                    txt = TextClip(
                        word_text, fontsize=fontsize, color=color, font=font,
                        stroke_color='black', stroke_width=2
                    ).set_position(('center', 0.8), relative=True).set_start(rel_start).set_duration(rel_end - rel_start)
                else:
                    txt = TextClip(
                        text=word_text, font_size=fontsize, color=color, font=font,
                        stroke_color='black', stroke_width=2
                    ).with_position(('center', 0.8), relative=True).with_start(rel_start).with_duration(rel_end - rel_start)

                text_clips.append(txt)
            except:
                continue

        if text_clips:
            # Composite
            return CompositeVideoClip([clip] + text_clips)
        return clip

class AudioEngine:
    """Module 5: Audio Glue"""
    def master(self, clip):
        if not AUDIO_DSP_AVAILABLE or clip.audio is None: return clip

        try:
            fs = 44100
            arr = clip.audio.to_soundarray(fps=fs)
            meter = pyln.Meter(fs)
            loudness = meter.integrated_loudness(arr)
            # Target -14
            gain = -14.0 - loudness
            arr = arr * (10 ** (gain/20.0))
            arr = np.clip(arr, -0.9, 0.9)

            if MOVIEPY_V1:
                from moviepy.audio.AudioClip import AudioArrayClip
                return clip.set_audio(AudioArrayClip(arr, fps=fs))
            else:
                from moviepy.audio.AudioClip import AudioArrayClip
                return clip.with_audio(AudioArrayClip(arr, fps=fs))
        except:
            return clip

class NeuralEditor:
    def __init__(self, openai_key=None):
        self.brain = ViralBrain(openai_key)
        self.retention = RetentionEngine()
        self.augmentor = ContentAugmentor()
        self.audio = AudioEngine()

    def process_video(self, video_path, viral_factor, attention_mode, assets_dir):
        if MOVIEPY_V1:
            clip = VideoFileClip(video_path)
        else:
            clip = VideoFileClip(video_path)

        # 1. Transcribe
        temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        clip.audio.write_audiofile(temp_audio, logger=None)
        segments = self.brain.transcribe_full(temp_audio)

        # 2. Viral Selection
        cuts = self.brain.analyze_viral_moments(segments, viral_factor)

        final_clips = []
        for start, end in cuts:
            # Cut
            if MOVIEPY_V1:
                sub = clip.subclip(start, end)
            else:
                sub = clip.subclipped(start, end)

            # Retention Zoom
            if attention_mode == "Gen-Z":
                sub = self.retention.apply_punch_in(sub, attention_mode)
            else:
                sub = self.retention.apply_ken_burns(sub)

            # Filter segments for this time range to pass to Augmentor
            # We pass the full list and the offset (start) to handle relative timing

            # B-Roll
            sub = self.augmentor.inject_b_roll(sub, segments, offset=start)

            # Karaoke
            sub = self.augmentor.generate_karaoke_subs(sub, segments, offset=start)

            final_clips.append(sub)

        if not final_clips:
            final_video = clip
        else:
            final_video = concatenate_videoclips(final_clips)

        # Master Audio
        final_video = self.audio.master(final_video)

        os.unlink(temp_audio)
        clip.close()

        return final_video

# --- UI ---

def main():
    st.set_page_config(page_title="Jules Insane Mode", page_icon="🧠", layout="wide")
    st.title("🧠 Jules Insane Mode: The Viral Engine")

    with st.sidebar:
        api_key = st.text_input("OpenAI API Key (Optional)", type="password")
        viral_factor = st.slider("Viral Factor", 0.1, 1.0, 0.5)
        attn_mode = st.radio("Attention Span", ["Gen-Z", "Millennial"])

        st.divider()
        st.write("Asset Manager (Upload 'money.mp4', etc)")
        uploaded_assets = st.file_uploader("Upload B-Roll", accept_multiple_files=True)
        if uploaded_assets:
            os.makedirs("assets", exist_ok=True)
            for f in uploaded_assets:
                with open(os.path.join("assets", f.name), "wb") as w:
                    w.write(f.read())
            st.success(f"Loaded {len(uploaded_assets)} assets.")

    video_file = st.file_uploader("Upload Raw Long-Form Video", type=["mp4"])

    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(video_file.read())

        if st.button("Activate Neural Engine"):
            editor = NeuralEditor(api_key)
            status = st.status("Spinning up Neural Networks...", expanded=True)

            try:
                status.write("🧠 Reading Video Memory...")
                output = editor.process_video(tfile.name, viral_factor, attn_mode, "assets")

                status.write("💾 Rendering Viral Output...")
                out_path = "insane_output.mp4"
                output.write_videofile(out_path, codec="libx264", audio_codec="aac", logger=None)

                status.update(label="✅ Viral Content Ready", state="complete", expanded=False)
                st.video(out_path)

            except Exception as e:
                st.error(f"Engine Failure: {e}")
            finally:
                os.unlink(tfile.name)

if __name__ == "__main__":
    main()
