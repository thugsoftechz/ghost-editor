
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
import math
from typing import List, Tuple, Optional, Dict

import numpy as np
import streamlit as st
import cv2
import librosa
import soundfile as sf
import pyloudnorm as pyln

# --- 1. Robust Imports & Compatibility ---

try:
    from moviepy.editor import (
        VideoFileClip, AudioFileClip, TextClip, CompositeVideoClip,
        CompositeAudioClip, concatenate_videoclips, vfx, afx, ImageClip
    )
    from moviepy.audio.AudioClip import AudioArrayClip
    import moviepy.audio.fx.all as audio_fx
    MOVIEPY_V1 = True
except ImportError:
    import moviepy
    from moviepy import (
        VideoFileClip, AudioFileClip, TextClip, CompositeVideoClip,
        CompositeAudioClip, concatenate_videoclips, ImageClip
    )
    from moviepy.audio.AudioClip import AudioArrayClip
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
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


# --- 2. Viral Brain (Content Intelligence) ---

class ViralBrain:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        if self.api_key and OPENAI_AVAILABLE:
            openai.api_key = self.api_key

        self.whisper_model = None

    def _load_whisper(self):
        if not self.whisper_model and WHISPER_AVAILABLE:
            self.whisper_model = WhisperModel("base", device="cpu", compute_type="int8")

    def transcribe(self, audio_path: str) -> List[dict]:
        if not WHISPER_AVAILABLE:
            return []

        self._load_whisper()
        st.write("🧠 ViralBrain: Transcribing Audio...")
        segments, _ = self.whisper_model.transcribe(audio_path, word_timestamps=True)

        results = []
        for s in segments:
            seg_data = {
                "start": s.start,
                "end": s.end,
                "text": s.text,
                "words": []
            }
            if s.words:
                for w in s.words:
                    seg_data["words"].append({
                        "word": w.word,
                        "start": w.start,
                        "end": w.end,
                        "probability": w.probability
                    })
            results.append(seg_data)
        return results

    def analyze_virality(self, segments: List[dict], audio_path: str, duration: float) -> List[Tuple[float, float]]:
        """
        Identifies segments.
        Primary: LLM Selection.
        Fallback: Librosa Silence Removal (DSP).
        """
        # LLM Logic
        if self.api_key and OPENAI_AVAILABLE and segments:
            try:
                full_text = " ".join([s["text"] for s in segments])
                prompt = (
                    "Analyze this transcript. Identify the top 3 most viral, funny, or intense "
                    "segments (approx 30-60s each). Return JSON: list of objects with 'start' and 'end' seconds. "
                    f"Transcript: {full_text[:8000]}"
                )

                response = openai.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a viral video editor. Return strictly JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=200
                )
                content = response.choices[0].message.content
                matches = re.findall(r'"start":\s*([\d.]+),\s*"end":\s*([\d.]+)', content)
                if matches:
                    return [(float(s), float(e)) for s, e in matches]

            except Exception as e:
                st.warning(f"ViralBrain LLM Error: {e}. Switching to DSP Fallback.")

        # Fallback: Librosa Silence Removal
        st.write("🧠 ViralBrain: Running DSP Silence Removal (Fallback)...")
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=None)
            # Split silence
            # top_db=40 (threshold -40dB)
            non_silent = librosa.effects.split(y, top_db=40)

            cuts = []
            for start_sample, end_sample in non_silent:
                start_t = start_sample / sr
                end_t = end_sample / sr

                # Buffer 0.2s
                start_t = max(0, start_t - 0.2)
                end_t = min(duration, end_t + 0.2)

                # Minimum duration filter (e.g. 1s) to avoid jitter
                if end_t - start_t > 1.0:
                    cuts.append((start_t, end_t))

            if not cuts:
                return [(0, duration)]

            return cuts

        except Exception as e:
            st.warning(f"DSP Error: {e}. Returning original.")
            return [(0, duration)]


# --- 3. Visual Retention Engine (Vision) ---

class RetentionEngine:
    def __init__(self):
        self.mp_face = None
        if MEDIAPIPE_AVAILABLE:
            self.mp_face = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

    def _get_face_trajectory(self, clip, sample_rate=5):
        if not MEDIAPIPE_AVAILABLE:
            return None

        st.write("👀 RetentionEngine: Tracking Faces...")
        w, h = clip.size
        x_centers = []

        # Use cv2 for explicit resizing (optimization)
        new_h = 240
        new_w = int(w * (new_h / h))

        last_x = 0.5
        for i, frame in enumerate(clip.iter_frames()):
            if i % sample_rate != 0:
                x_centers.append(last_x)
                continue

            # Use CV2 to resize for speed, then pass to mediapipe
            small_frame = cv2.resize(frame, (new_w, new_h))

            results = self.mp_face.process(small_frame)
            if results.detections:
                bbox = results.detections[0].location_data.relative_bounding_box
                center_x = bbox.xmin + bbox.width / 2
                last_x = center_x

            x_centers.append(last_x)

        fps = clip.fps if clip.fps else 24
        window = int(fps * 2)
        if window < 1: window = 1
        smoothed = np.convolve(x_centers, np.ones(window)/window, mode='same')

        return smoothed

    def smart_crop_9_16(self, clip):
        w, h = clip.size
        target_aspect = 9/16
        new_w = h * target_aspect

        trajectory = self._get_face_trajectory(clip)

        if trajectory is None:
            x1 = w/2 - new_w/2
            if MOVIEPY_V1:
                return clip.crop(x1=x1, width=new_w, height=h)
            else:
                return clip.with_effects([vfx.Crop(x1=x1, width=new_w, height=h)])

        def crop_func(t):
            fps = clip.fps if clip.fps else 24
            idx = int(t * fps)
            if idx >= len(trajectory):
                idx = len(trajectory) - 1

            center_norm = trajectory[idx]
            center_x = center_norm * w
            x1 = center_x - new_w / 2
            x1 = max(0, min(x1, w - new_w))
            return x1, 0, x1 + new_w, h

        if MOVIEPY_V1:
            return clip.crop(x1=lambda t: crop_func(t)[0], width=new_w, height=h)
        else:
            try:
                return clip.with_effects([vfx.Crop(x1=lambda t: crop_func(t)[0], width=new_w, height=h)])
            except:
                 x1 = w/2 - new_w/2
                 return clip.with_effects([vfx.Crop(x1=x1, width=new_w, height=h)])

    def apply_punch_ins(self, clip):
        scale = random.choice([1.0, 1.1])
        if scale == 1.0: return clip

        if MOVIEPY_V1:
            return clip.resize(scale)
        else:
            return clip.with_effects([vfx.Resize(scale)])

    def apply_dynamic_zoom(self, clip):
        """Ken Burns Effect: 1.0x -> 1.15x over clip duration."""
        def zoom(t):
            return 1.0 + 0.15 * (t / clip.duration)

        if MOVIEPY_V1:
            return clip.resize(zoom)
        else:
            return clip.with_effects([vfx.Resize(zoom)])


# --- 4. Professional Audio Engine ---

class AudioEngine:
    def master(self, clip, target_lufs=-14.0):
        if clip.audio is None: return clip
        try:
            st.write("🎧 AudioEngine: Mastering LUFS...")
            fs = 44100
            audio_arr = clip.audio.to_soundarray(fps=fs)
            meter = pyln.Meter(fs)
            loudness = meter.integrated_loudness(audio_arr)
            gain_db = target_lufs - loudness
            gain_lin = 10 ** (gain_db / 20.0)
            new_arr = audio_arr * gain_lin
            limit = 10 ** (-1.0 / 20.0)
            new_arr = np.clip(new_arr, -limit, limit)
            new_audio = AudioArrayClip(new_arr, fps=fs)

            if MOVIEPY_V1:
                return clip.set_audio(new_audio)
            else:
                return clip.with_audio(new_audio)
        except:
            return clip

    def add_ducking(self, clip, music_path):
        if not music_path: return clip
        try:
            if MOVIEPY_V1:
                music = AudioFileClip(music_path)
                music = afx.audio_loop(music, duration=clip.duration)
                music = music.fx(afx.volumex, 0.12)
                comp = CompositeAudioClip([clip.audio, music])
                return clip.set_audio(comp)
            else:
                music = AudioFileClip(music_path)
                music = music.with_effects([
                    afx.AudioLoop(duration=clip.duration),
                    afx.MultiplyVolume(0.12)
                ])
                comp = CompositeAudioClip([clip.audio, music])
                return clip.with_audio(comp)
        except:
            return clip

    def apply_crossfade(self, clip, duration=0.05):
        """Applies audio fade in/out to smoothing cuts."""
        if MOVIEPY_V1:
            new_audio = clip.audio.fx(audio_fx.audio_fadein, duration).fx(audio_fx.audio_fadeout, duration)
            return clip.set_audio(new_audio)
        else:
            new_audio = clip.audio.with_effects([
                afx.AudioFadeIn(duration),
                afx.AudioFadeOut(duration)
            ])
            return clip.with_audio(new_audio)

    def insert_transitions(self, clips, sfx_path=None):
        """
        Concatenates clips with transitions.
        Inserts SFX at start of clips (except first).
        """
        sfx = None
        if sfx_path and os.path.exists(sfx_path):
            try:
                sfx = AudioFileClip(sfx_path)
            except:
                pass

        final_clips = []
        for i, clip in enumerate(clips):
            # Apply Crossfade first
            clip = self.apply_crossfade(clip)

            # Add SFX if not first
            if i > 0 and sfx:
                clip_audio = clip.audio
                if MOVIEPY_V1:
                    comp = CompositeAudioClip([clip_audio, sfx])
                    comp = comp.set_duration(clip.duration)
                    clip = clip.set_audio(comp)
                else:
                    comp = CompositeAudioClip([clip_audio, sfx])
                    comp = comp.with_duration(clip.duration)
                    clip = clip.with_audio(comp)

            final_clips.append(clip)

        return concatenate_videoclips(final_clips)


# --- 5. Graphics & Color Engine ---

class GraphicsEngine:
    def apply_grade(self, clip, mode):
        def vlog_filter(im):
            im = im.astype(float)
            gray = np.dot(im[...,:3], [0.299, 0.587, 0.114])
            gray = gray[:,:,np.newaxis]
            im = gray + (im - gray) * 1.3
            return np.clip(im, 0, 255).astype(np.uint8)

        def cinematic_filter(im):
            im = im.astype(float)
            im = (im - 128) * 1.2 + 128
            luma = np.dot(im[...,:3], [0.299, 0.587, 0.114])
            highlights = luma > 150
            shadows = luma < 100
            im[highlights, 0] += 15
            im[highlights, 1] += 5
            im[highlights, 2] -= 10
            im[shadows, 0] -= 10
            im[shadows, 1] += 5
            im[shadows, 2] += 15
            return np.clip(im, 0, 255).astype(np.uint8)

        if mode == "Vlog Mode":
            return clip.fl_image(vlog_filter) if MOVIEPY_V1 else clip.image_transform(vlog_filter)
        elif mode == "Cinematic Mode":
            return clip.fl_image(cinematic_filter) if MOVIEPY_V1 else clip.image_transform(cinematic_filter)
        return clip

    def generate_karaoke(self, clip, segments, offset=0):
        if not segments: return clip
        st.write("🎨 GraphicsEngine: Animating Captions...")
        w, h = clip.size
        fontsize = int(h / 20)
        font = "Impact" if os.name == 'nt' else "DejaVu-Sans-Bold"
        text_clips = []
        words = []
        for s in segments:
            if "words" in s:
                words.extend(s["words"])

        for w_obj in words:
            word = w_obj["word"].strip()
            start = w_obj["start"] - offset
            end = w_obj["end"] - offset
            if start < 0 or start > clip.duration: continue

            try:
                # Optimized for retention: Single word pop-up
                if MOVIEPY_V1:
                    txt = TextClip(word, fontsize=fontsize, color='yellow', font=font,
                        stroke_color='black', stroke_width=2, method='caption'
                    ).set_position(('center', 0.8), relative=True).set_start(start).set_duration(end-start)
                else:
                    txt = TextClip(text=word, font_size=fontsize, color='yellow', font=font,
                        stroke_color='black', stroke_width=2, method='caption'
                    ).with_position(('center', 0.8), relative=True).with_start(start).with_duration(end-start)
                text_clips.append(txt)
            except: pass
        if text_clips:
            return CompositeVideoClip([clip] + text_clips)
        return clip

    def inject_b_roll(self, clip, segments, assets_dir, offset=0):
        if not os.path.exists(assets_dir): return clip
        words = []
        for s in segments:
            if "words" in s:
                words.extend(s["words"])

        b_roll_clips = []
        for w_obj in words:
            word = w_obj["word"].lower().strip(".,!?")
            asset_path = os.path.join(assets_dir, f"{word}.mp4")
            start = w_obj["start"] - offset
            if os.path.exists(asset_path) and 0 <= start < clip.duration:
                try:
                    b_roll = VideoFileClip(asset_path).resize(height=clip.h)
                    b_roll = b_roll.set_position("center").set_start(start).set_duration(1.5)
                    b_roll_clips.append(b_roll)
                except: pass

        if b_roll_clips:
            return CompositeVideoClip([clip] + b_roll_clips)
        return clip


# --- 6. Master Controller: JulesEngine ---

class JulesEngine:
    def __init__(self, output_dir="Jules_Output"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.brain = ViralBrain(st.session_state.get("openai_key"))
        self.vision = RetentionEngine()
        self.audio = AudioEngine()
        self.graphics = GraphicsEngine()

    def process_video(self, input_path: str, mode: str, viral_factor: float,
                      music_path: str = None, sfx_path: str = None,
                      preview_mode: bool = False):

        status = st.status(f"🚀 Processing: {os.path.basename(input_path)}", expanded=True)

        try:
            status.write("📂 Loading Media...")
            if MOVIEPY_V1:
                clip = VideoFileClip(input_path)
            else:
                clip = VideoFileClip(input_path)

            if preview_mode:
                clip = clip.subclip(0, min(10, clip.duration))

            # Extract Audio & Transcribe
            temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
            clip.audio.write_audiofile(temp_audio, logger=None)

            segments = self.brain.transcribe(temp_audio)

            # Analyze / Cut (Viral or Fallback DSP)
            if preview_mode:
                cuts = [(0, clip.duration)]
            else:
                status.write("🧠 Analyzing Content...")
                cuts = self.brain.analyze_virality(segments, temp_audio, clip.duration)

            processed_clips = []

            for i, (start, end) in enumerate(cuts):
                status.write(f"🎞️ Editing Segment {i+1}...")

                if MOVIEPY_V1:
                    sub = clip.subclip(start, end)
                else:
                    sub = clip.subclipped(start, end)

                # Visual Processing
                if mode == "Viral Short (9:16)":
                    sub = self.vision.smart_crop_9_16(sub)

                # Random Logic: Punch-in OR Ken Burns
                if random.choice([True, False]):
                    sub = self.vision.apply_punch_ins(sub)
                else:
                    sub = self.vision.apply_dynamic_zoom(sub)

                sub = self.graphics.inject_b_roll(sub, segments, "assets", offset=start)

                grade_mode = st.session_state.get("grade_mode", "None")
                if grade_mode != "None":
                    sub = self.graphics.apply_grade(sub, grade_mode)

                sub = self.graphics.generate_karaoke(sub, segments, offset=start)

                processed_clips.append(sub)

            # Concatenate with Transitions (Crossfade & SFX)
            if not processed_clips:
                final_video = clip
            else:
                final_video = self.audio.insert_transitions(processed_clips, sfx_path)

            # Audio Glue
            if music_path:
                status.write("🎵 Applying Smart Ducking...")
                final_video = self.audio.add_ducking(final_video, music_path)

            status.write("🎚️ Mastering Audio...")
            final_video = self.audio.master(final_video)

            filename = f"jules_{int(time.time())}_{os.path.basename(input_path)}"
            out_path = os.path.join(self.output_dir, filename)

            status.write("💾 Rendering to Disk...")
            final_video.write_videofile(out_path, codec="libx264", audio_codec="aac", logger=None, preset="medium", threads=4)

            status.update(label="✅ Finished", state="complete", expanded=False)

            os.unlink(temp_audio)
            clip.close()
            final_video.close()
            return out_path

        except Exception as e:
            status.update(label="❌ Failed", state="error")
            st.error(f"Error: {e}")
            return None


# --- 7. UI: Studio Mode ---

def main():
    st.set_page_config(page_title="Jules Studio", page_icon="🔥", layout="wide")
    st.markdown("""<style>.stApp { background-color: #0E1117; color: #FFF; } div.stButton > button { background-color: #FF4B4B; color: white; border-radius: 6px; font-weight: bold; width: 100%; }</style>""", unsafe_allow_html=True)

    st.sidebar.title("🔥 Jules Studio")
    st.sidebar.caption("Autonomous AI Video Engine")

    mode = st.sidebar.selectbox("Edit Mode", ["Viral Short (9:16)", "Cinematic (16:9)"])
    viral_factor = st.sidebar.slider("Viral Aggressiveness", 0.1, 1.0, 0.5)
    st.session_state["grade_mode"] = st.sidebar.selectbox("Color Grade", ["None", "Vlog Mode", "Cinematic Mode"])
    st.session_state["openai_key"] = st.sidebar.text_input("OpenAI Key (Optional)", type="password")

    st.sidebar.divider()
    bg_music = st.sidebar.file_uploader("Background Music", type=["mp3"])
    sfx_upload = st.sidebar.file_uploader("SFX (Whoosh)", type=["mp3"])

    music_path = None
    if bg_music:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            f.write(bg_music.read())
            music_path = f.name

    sfx_path = None
    if sfx_upload:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            f.write(sfx_upload.read())
            sfx_path = f.name

    st.title("🎬 Production Dashboard")
    root_input = st.text_input("Root Folder Path", ".")

    if root_input and os.path.exists(root_input):
        videos = []
        for r, d, f in os.walk(root_input):
            if "Jules_Output" in d: d.remove("Jules_Output")
            for file in f:
                if file.lower().endswith(('.mp4', '.mov', '.mkv')):
                    videos.append(os.path.join(r, file))

        st.write(f"Found {len(videos)} source files.")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶️ Generate 10s Preview"):
                engine = JulesEngine()
                if videos:
                    out = engine.process_video(videos[0], mode, viral_factor, music_path, sfx_path, preview_mode=True)
                    if out: st.video(out)

        with col2:
            if st.button("🚀 Render All (Production)"):
                engine = JulesEngine()
                progress = st.progress(0)
                for i, v in enumerate(videos):
                    engine.process_video(v, mode, viral_factor, music_path, sfx_path, preview_mode=False)
                    progress.progress((i+1)/len(videos))
                st.success("Batch Production Complete.")

if __name__ == "__main__":
    main()
