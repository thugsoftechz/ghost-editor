
import os
import shutil
import time
import numpy as np
import streamlit as st

# --- Robust MoviePy Imports (v1 vs v2) ---
try:
    # Attempt MoviePy v1 (moviepy.editor)
    from moviepy.editor import (
        VideoFileClip, AudioFileClip, ImageClip, CompositeVideoClip,
        CompositeAudioClip, concatenate_videoclips, vfx, afx
    )
    import moviepy.audio.fx.all as audio_fx
    MOVIEPY_V1 = True
except ImportError:
    # Fallback to MoviePy v2
    import moviepy
    from moviepy import (
        VideoFileClip, AudioFileClip, ImageClip, CompositeVideoClip,
        CompositeAudioClip, concatenate_videoclips
    )
    import moviepy.video.fx as vfx
    import moviepy.audio.fx as afx
    MOVIEPY_V1 = False

class AIEditorEngine:
    """
    Core logic for the Jules Pro Video Editor.
    Encapsulates all video processing to separate it from UI.
    """
    def __init__(self, silence_threshold=0.03, min_clip_duration=1.0,
                 crossfade_duration=0.05, music_vol=0.12):
        self.silence_threshold = silence_threshold
        self.min_clip_duration = min_clip_duration
        self.crossfade_duration = crossfade_duration
        self.music_vol = music_vol

    def _get_max_volume(self, clip):
        """Helper to check max volume of a clip/subclip efficiently."""
        if clip.audio is None:
            return 0

        try:
            # v2 safety: check if fps is set on audio, if not set it
            if not MOVIEPY_V1 and (not hasattr(clip.audio, 'fps') or clip.audio.fps is None):
                clip.audio.fps = 44100

            return clip.audio.max_volume()
        except Exception:
            # Fallback: manually get a chunk (robustness)
            try:
                # Analyze a small chunk (0.1s) if max_volume fails
                chunk = clip.audio.to_soundarray(nbytes=2, fps=44100)
                if chunk is None or chunk.size == 0:
                    return 0
                return np.max(np.abs(chunk))
            except:
                return 0

    def cut_silence(self, video):
        """
        Analyzes audio and removes silent segments.
        Returns a list of 'loud' subclips.
        """
        if video.audio is None:
            return None

        # Analysis Window
        window = 0.2
        duration = video.duration
        t = 0
        loud_segments = []
        current_start = None

        # Iterate through video audio
        while t < duration:
            t_end = min(t + window, duration)
            if MOVIEPY_V1:
                sub = video.subclip(t, t_end)
            else:
                sub = video.subclipped(t, t_end)

            if self._get_max_volume(sub) >= self.silence_threshold:
                if current_start is None:
                    current_start = t
            else:
                if current_start is not None:
                    loud_segments.append((current_start, t))
                    current_start = None
            t = t_end

        if current_start is not None:
            loud_segments.append((current_start, duration))

        # Filter short clips
        valid_segments = [
            (s, e) for s, e in loud_segments
            if (e - s) >= self.min_clip_duration
        ]

        if not valid_segments:
            return None

        # Create buffered clips
        clips = []
        for start, end in valid_segments:
            # Buffer: 0.1s padding if possible
            s_buf = max(0, start - 0.1)
            e_buf = min(duration, end + 0.1)

            if MOVIEPY_V1:
                clip = video.subclip(s_buf, e_buf)
            else:
                clip = video.subclipped(s_buf, e_buf)

            # Apply Audio Crossfades (Fade In/Out) to prevent clicks
            if MOVIEPY_V1:
                # v1: clip.audio.fx... returns modified audio
                # audio_fx.audio_fadein is the function
                new_audio = clip.audio.fx(audio_fx.audio_fadein, self.crossfade_duration) \
                                      .fx(audio_fx.audio_fadeout, self.crossfade_duration)
                clip = clip.set_audio(new_audio)
            else:
                # v2: with_effects([afx.AudioFadeIn(...)])
                # Apply to audio directly
                new_audio = clip.audio.with_effects([
                    afx.AudioFadeIn(self.crossfade_duration),
                    afx.AudioFadeOut(self.crossfade_duration)
                ])
                clip = clip.with_audio(new_audio)

            clips.append(clip)

        return clips

    def apply_color_grade(self, clip, mode):
        """
        Applies color grading based on mode using NumPy filters.
        """
        def filter_vlog(image):
            # Saturation 1.3x, Contrast 1.1x (Mild Pop)
            img = image.astype(float)
            # Contrast
            img = (img - 128.0) * 1.1 + 128.0
            # Saturation
            gray = np.dot(img[..., :3], [0.299, 0.587, 0.114])[..., np.newaxis]
            img = gray + (img - gray) * 1.3
            return np.clip(img, 0, 255).astype(np.uint8)

        def filter_cinematic(image):
            # High Contrast 1.2x, Teal/Orange Tint
            img = image.astype(float)

            # Contrast
            img = (img - 128.0) * 1.2 + 128.0

            # Luma for Highlights/Shadows
            luma = np.dot(img[..., :3], [0.299, 0.587, 0.114])

            highlights = luma > 150
            shadows = luma < 100

            # Highlights: Add Orange/Gold (R+, G+)
            img[highlights, 0] += 15 # R
            img[highlights, 1] += 5  # G
            img[highlights, 2] -= 10 # B

            # Shadows: Add Teal (R-, G+, B+)
            img[shadows, 0] -= 10 # R
            img[shadows, 1] += 5  # G
            img[shadows, 2] += 15 # B

            # Desaturate slightly (0.8x) for that "grim/clean" look
            gray = luma[..., np.newaxis]
            img = gray + (img - gray) * 0.8

            return np.clip(img, 0, 255).astype(np.uint8)

        if mode == "Vlog Mode":
            return clip.fl_image(filter_vlog) if MOVIEPY_V1 else clip.image_transform(filter_vlog)
        elif mode == "Cinematic Mode":
            return clip.fl_image(filter_cinematic) if MOVIEPY_V1 else clip.image_transform(filter_cinematic)
        else:
            return clip

    def add_transitions(self, clips, sfx_path):
        """
        Concatenates clips and inserts SFX at cut points.
        """
        # Load SFX
        sfx = None
        if sfx_path and os.path.exists(sfx_path):
            try:
                sfx = AudioFileClip(sfx_path)
            except:
                pass

        final_clips = []
        for i, clip in enumerate(clips):
            # Add SFX to start of clip (except the very first one)
            if i > 0 and sfx:
                # Composite Audio: Original Audio + SFX
                clip_audio = clip.audio

                # Check durations. If SFX is longer than clip, we might need to trim or just let it mix?
                # CompositeAudioClip duration is usually max of inputs.
                # We want to preserve clip duration.

                if MOVIEPY_V1:
                    comp = CompositeAudioClip([clip_audio, sfx])
                    # Force duration to match video clip
                    comp = comp.set_duration(clip.duration)
                    clip = clip.set_audio(comp)
                else:
                    comp = CompositeAudioClip([clip_audio, sfx])
                    comp = comp.with_duration(clip.duration)
                    clip = clip.with_audio(comp)

            final_clips.append(clip)

        return concatenate_videoclips(final_clips)

    def add_audio_ducking(self, video, music_path):
        """
        Adds background music looped at 12% volume.
        """
        if not music_path or not os.path.exists(music_path):
            return video

        try:
            bg_music = AudioFileClip(music_path)

            if MOVIEPY_V1:
                # Loop and Volume
                bg_music = afx.audio_loop(bg_music, duration=video.duration)
                bg_music = bg_music.fx(afx.volumex, self.music_vol)

                new_audio = CompositeAudioClip([video.audio, bg_music])
                return video.set_audio(new_audio)
            else:
                # v2
                # Loop: afx.AudioLoop(duration=...)
                bg_music = bg_music.with_effects([afx.AudioLoop(duration=video.duration)])
                # Volume: afx.MultiplyVolume(factor)
                bg_music = bg_music.with_effects([afx.MultiplyVolume(self.music_vol)])

                new_audio = CompositeAudioClip([video.audio, bg_music])
                return video.with_audio(new_audio)
        except:
            # If music fails, return original
            return video

    def add_watermark(self, video, watermark_path):
        """
        Overlays resized, padded watermark at bottom-right.
        """
        if not watermark_path or not os.path.exists(watermark_path):
            return video

        try:
            if MOVIEPY_V1:
                wm = ImageClip(watermark_path)
                # Resize to 10% of video height
                wm = wm.resize(height=int(video.h * 0.1))
                wm = wm.set_duration(video.duration)
                # Position: Bottom-Right with padding (margin)
                # margin creates a transparent wrapper
                wm = wm.margin(right=20, bottom=20, opacity=0)
                wm = wm.set_position(("right", "bottom"))

                return CompositeVideoClip([video, wm])
            else:
                wm = ImageClip(watermark_path)
                # v2: use resized() or with_effects([vfx.Resize(...)])
                # Check dir(vfx), 'Resize' is a class.
                wm = wm.with_effects([vfx.Resize(height=int(video.h * 0.1))])
                wm = wm.with_duration(video.duration)

                # Position & Margin
                # v2 Margin effect
                wm = wm.with_effects([vfx.Margin(right=20, bottom=20, opacity=0)])
                wm = wm.with_position(("right", "bottom"))

                return CompositeVideoClip([video, wm])
        except:
            return video

# --- Streamlit UI ---

def main():
    st.set_page_config(
        page_title="Jules Pro Editor",
        page_icon="🎬",
        layout="wide"
    )

    # Custom "Studio" CSS
    st.markdown("""
    <style>
        .stApp {
            background-color: #0E1117;
            color: #FAFAFA;
        }
        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: #262730;
            border-radius: 4px;
            color: #FAFAFA;
            padding: 10px 20px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #FF4B4B;
            color: white;
        }
        /* Button styling */
        div.stButton > button {
            background-color: #FF4B4B;
            color: white;
            border: none;
            padding: 10px 24px;
            font-size: 16px;
            border-radius: 8px;
            font-weight: 600;
        }
        div.stButton > button:hover {
            background-color: #FF2B2B;
        }
    </style>
    """, unsafe_allow_html=True)

    st.title("🎬 Jules Pro Video Editor")
    st.caption("Autonomous High-End Post-Production Engine")

    # Layout: Tabs
    tab_dash, tab_settings = st.tabs(["🎛️ Dashboard", "⚙️ Settings"])

    with tab_settings:
        st.header("Engine Configuration")
        col1, col2 = st.columns(2)
        with col1:
            threshold = st.slider("Silence Threshold", 0.01, 0.1, 0.03, 0.005,
                                  help="Volume level below which audio is considered silence.")
            min_dur = st.slider("Min Clip Duration (s)", 0.5, 3.0, 1.0,
                                help="Shortest allowed clip length.")
        with col2:
            music_vol = st.slider("Background Music Volume", 0.05, 0.5, 0.12,
                                  help="Ducking level for background track (12% default).")
            crossfade = st.slider("Crossfade Duration (s)", 0.0, 0.5, 0.05,
                                  help="Audio fade in/out at cuts.")

        st.subheader("Asset Paths")
        watermark_path = st.text_input("Watermark Image Path (.png)", "")
        bg_music_path = st.text_input("Background Music Path (.mp3)", "")
        sfx_path = st.text_input("Transition SFX Path (.mp3)", "whoosh.mp3")

    with tab_dash:
        st.header("Batch Processing")

        root_path = st.text_input("Root Video Folder", ".")
        color_mode = st.selectbox("Color Grading Mode", ["None", "Vlog Mode", "Cinematic Mode"])

        if st.button("🚀 Start Production"):
            if not os.path.exists(root_path):
                st.error(f"Root path not found: {root_path}")
            else:
                # Initialize Engine
                engine = AIEditorEngine(
                    silence_threshold=threshold,
                    min_clip_duration=min_dur,
                    crossfade_duration=crossfade,
                    music_vol=music_vol
                )

                # Scan for videos
                videos = []
                out_dir_name = "Jules_Pro_Output"

                st.write(f"Scanning `{root_path}`...")
                for r, d, f in os.walk(root_path):
                    # Robust ignore of output folder
                    if out_dir_name in d:
                        d.remove(out_dir_name)

                    for file in f:
                        if file.lower().endswith(('.mp4', '.mov', '.mkv')):
                            videos.append(os.path.join(r, file))

                if not videos:
                    st.warning("No video files found.")
                else:
                    output_root = os.path.join(root_path, out_dir_name)
                    os.makedirs(output_root, exist_ok=True)

                    progress_bar = st.progress(0)

                    for i, v_path in enumerate(videos):
                        fname = os.path.basename(v_path)

                        # Real-time Status Log
                        with st.status(f"Processing: **{fname}**", expanded=True) as status:
                            try:
                                status.write("🔍 Analyzing Audio & Smart-Cutting Silence...")
                                video = VideoFileClip(v_path)

                                # 1. Smart Cut
                                clips = engine.cut_silence(video)
                                if not clips:
                                    status.write("⚠️ Skipped: No loud segments found (or file error).")
                                    status.update(label=f"⚠️ Skipped: {fname}", state="error")
                                    video.close()
                                    continue

                                # 2. Transitions
                                status.write("✂️ Assembling Clips & Adding Transitions...")
                                processed_clip = engine.add_transitions(clips, sfx_path)

                                # 3. Color Grading
                                if color_mode != "None":
                                    status.write(f"🎨 Applying Color Grade: {color_mode}...")
                                    processed_clip = engine.apply_color_grade(processed_clip, color_mode)

                                # 4. Audio Ducking
                                status.write("🎵 Engineering Audio Layers (Ducking & Fades)...")
                                processed_clip = engine.add_audio_ducking(processed_clip, bg_music_path)

                                # 5. Watermark
                                if watermark_path:
                                    status.write("©️ Overlaying Watermark...")
                                    processed_clip = engine.add_watermark(processed_clip, watermark_path)

                                # Output
                                rel_path = os.path.relpath(v_path, root_path)
                                out_path = os.path.join(output_root, rel_path)
                                os.makedirs(os.path.dirname(out_path), exist_ok=True)

                                status.write("💾 Rendering Final Master...")
                                processed_clip.write_videofile(
                                    out_path,
                                    codec="libx264",
                                    audio_codec="aac",
                                    fps=24,
                                    preset="medium",
                                    threads=4,
                                    logger=None
                                )

                                # Cleanup
                                processed_clip.close()
                                video.close()

                                status.update(label=f"✅ Completed: {fname}", state="complete", expanded=False)

                            except Exception as e:
                                st.error(f"Error processing {fname}: {str(e)}")
                                status.update(label=f"❌ Failed: {fname}", state="error")
                                if 'video' in locals(): video.close()

                        progress_bar.progress((i + 1) / len(videos))

                    st.balloons()
                    st.success("✨ All videos processed successfully!")

if __name__ == "__main__":
    main()
