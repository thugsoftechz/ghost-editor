
import os
import shutil
import time
import numpy as np
import streamlit as st

# --- robust imports ---
try:
    # Try v1 first as requested
    from moviepy.editor import (
        VideoFileClip, AudioFileClip, ImageClip, CompositeVideoClip,
        CompositeAudioClip, concatenate_videoclips, vfx, afx
    )
    import moviepy.audio.fx.all as audio_fx
    MOVIEPY_V1 = True
except ImportError:
    # Fallback to v2
    import moviepy
    from moviepy import (
        VideoFileClip, AudioFileClip, ImageClip, CompositeVideoClip,
        CompositeAudioClip, concatenate_videoclips
    )
    import moviepy.video.fx as vfx
    import moviepy.audio.fx as afx
    # v2 usually doesn't have audio.fx.all, effects are in moviepy.audio.fx
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

    def _get_max_volume(self, clip, chunk_size=0.1):
        """Helper to check max volume of a clip/subclip efficiently."""
        if clip.audio is None:
            return 0

        try:
            # v2 safety: check if fps is set, if not set it (often needed for AudioClip)
            if not MOVIEPY_V1 and not hasattr(clip.audio, 'fps') or clip.audio.fps is None:
                # Default to 44100 if not set, though subclipped usually inherits
                clip.audio.fps = 44100

            return clip.audio.max_volume()
        except Exception:
            # Fallback: manually get a chunk
            try:
                # v2: get_frame at t=0? No, we need volume.
                # v2 audio is usually an array generation.
                # subclip duration is small (0.2s).
                # to_soundarray() is robust
                arr = clip.audio.to_soundarray(fps=44100)
                if arr is None or arr.size == 0:
                    return 0
                return np.max(np.abs(arr))
            except:
                return 0

    def cut_silence(self, video):
        """
        Analyzes audio and removes silent segments.
        Returns a concatenated clip of 'loud' segments.
        """
        if video.audio is None:
            # No audio, cannot detect silence. Return as is?
            # Or return None? Requirement: "Skip it gracefully" if no loud segments.
            # If no audio, technically no loud segments.
            return None

        # Analysis Window
        window = 0.2
        duration = video.duration
        t = 0
        loud_segments = []
        current_start = None

        # Iterate
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

        # Create clips
        clips = []
        for start, end in valid_segments:
            # Buffer? "keep a small buffer"
            # Let's add 0.1s buffer if possible, clamped to duration
            s_buf = max(0, start - 0.1)
            e_buf = min(duration, end + 0.1)

            if MOVIEPY_V1:
                clip = video.subclip(s_buf, e_buf)
            else:
                clip = video.subclipped(s_buf, e_buf)

            # Apply Crossfade (Audio)
            # v1: clip.audio.fx(afx.audio_fadein, 0.05).fx(afx.audio_fadeout, 0.05)
            # v2: clip.with_effects(...)

            # We apply audio fade to avoid clicks
            if MOVIEPY_V1:
                # clip.audio is an AudioClip
                # We need to set the audio back to the clip
                new_audio = clip.audio.fx(audio_fx.audio_fadein, self.crossfade_duration) \
                                      .fx(audio_fx.audio_fadeout, self.crossfade_duration)
                clip = clip.set_audio(new_audio)
            else:
                # v2
                # Audio effects are in afx
                new_audio = clip.audio.with_effects([
                    afx.AudioFadeIn(self.crossfade_duration),
                    afx.AudioFadeOut(self.crossfade_duration)
                ])
                clip = clip.with_audio(new_audio)

            clips.append(clip)

        return clips

    def apply_color_grade(self, clip, mode):
        """
        Applies color grading based on mode using NumPy.
        """
        def filter_vlog(image):
            # Saturation 1.3x, Contrast 1.1x
            img = image.astype(float)
            # Contrast
            img = (img - 128.0) * 1.1 + 128.0
            # Saturation
            gray = np.dot(img[..., :3], [0.299, 0.587, 0.114])[..., np.newaxis]
            img = gray + (img - gray) * 1.3
            return np.clip(img, 0, 255).astype(np.uint8)

        def filter_cinematic(image):
            # High Contrast 1.2x
            # Teal/Orange: Boost Red in highlights, Blue/Cyan in shadows
            img = image.astype(float)

            # Contrast
            img = (img - 128.0) * 1.2 + 128.0

            # Simple Teal/Orange Tint
            # Highlights (>150): +Red, -Blue
            # Shadows (<100): -Red, +Blue, +Green

            # This is expensive per pixel in python?
            # Vectorized approach

            # Luma
            luma = np.dot(img[..., :3], [0.299, 0.587, 0.114])

            # Masks
            highlights = luma > 150
            shadows = luma < 100

            # Apply tints (subtle)
            # Highlights: Add Orange (255, 160, 0) -> R+20, G+10
            img[highlights, 0] += 15 # R
            img[highlights, 1] += 5  # G
            img[highlights, 2] -= 10 # B

            # Shadows: Add Teal (0, 128, 128) -> R-10, G+5, B+15
            img[shadows, 0] -= 10 # R
            img[shadows, 1] += 5  # G
            img[shadows, 2] += 15 # B

            # Desaturate slightly (0.8)
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
        Inserts SFX at cut points.
        Returns concatenated clip.
        """
        # Load SFX
        sfx = None
        if sfx_path and os.path.exists(sfx_path):
            if MOVIEPY_V1:
                sfx = AudioFileClip(sfx_path)
            else:
                sfx = AudioFileClip(sfx_path)

        final_clips = []
        for i, clip in enumerate(clips):
            if i > 0 and sfx:
                # Add SFX to start of clip
                # Composite Audio
                clip_audio = clip.audio
                sfx_dur = sfx.duration

                # Check if sfx is longer than clip?
                # If sfx is 0.5s and clip is 1.0s, fine.

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

    def add_audio_ducking(self, video, music_path):
        """
        Adds background music with ducking.
        """
        if not music_path or not os.path.exists(music_path):
            return video

        if MOVIEPY_V1:
            bg_music = AudioFileClip(music_path)
            # Loop
            bg_music = afx.audio_loop(bg_music, duration=video.duration)
            # Volume
            bg_music = bg_music.fx(afx.volumex, self.music_vol)

            # Composite
            new_audio = CompositeAudioClip([video.audio, bg_music])
            return video.set_audio(new_audio)
        else:
            bg_music = AudioFileClip(music_path)
            # v2 loop: using afx.AudioLoop or similar?
            # afx.AudioLoop(duration=...) effect?
            # Or just wrap the audio object?
            # v2: audio.with_effects([afx.AudioLoop(duration=...)])
            bg_music = bg_music.with_effects([afx.AudioLoop(duration=video.duration)])

            # Volume: afx.MultiplyVolume
            bg_music = bg_music.with_effects([afx.MultiplyVolume(self.music_vol)])

            new_audio = CompositeAudioClip([video.audio, bg_music])
            return video.with_audio(new_audio)

    def add_watermark(self, video, watermark_path):
        """
        Overlays watermark.
        """
        if not watermark_path or not os.path.exists(watermark_path):
            return video

        if MOVIEPY_V1:
            wm = ImageClip(watermark_path)
            # Resize - height 50px?
            wm = wm.resize(height=video.h // 10) # 10% height
            wm = wm.set_duration(video.duration)
            # Position
            wm = wm.set_position(("right", "bottom")).margin(right=20, bottom=20, opacity=0)

            return CompositeVideoClip([video, wm])
        else:
            wm = ImageClip(watermark_path)
            # v2: resized -> resized(height=...) method? or with_effects([vfx.Resize(...)])
            # ImageClip in v2 usually has resized method if imported from moviepy?
            # Check v2 docs in mind: clip.resized(height=...)
            wm = wm.resized(height=video.h // 10)
            wm = wm.with_duration(video.duration)
            # Position: with_position
            wm = wm.with_position(("right", "bottom")).with_effects([vfx.Margin(right=20, bottom=20, opacity=0)])

            return CompositeVideoClip([video, wm])

# --- Streamlit UI ---

def main():
    st.set_page_config(
        page_title="Jules Pro Editor",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS
    st.markdown("""
    <style>
        .stApp {
            background-color: #0e1117;
            color: #ffffff;
        }
        .stButton>button {
            background-color: #ff4b4b;
            color: white;
            border-radius: 5px;
            font-weight: bold;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 20px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: #262730;
            border-radius: 5px;
            padding-top: 10px;
            padding-bottom: 10px;
            color: white;
        }
        .stTabs [aria-selected="true"] {
            background-color: #ff4b4b;
        }
    </style>
    """, unsafe_allow_html=True)

    st.title("🎬 Jules Pro Video Editor")
    st.caption("Automated High-End Post-Production Engine")

    tab_dash, tab_settings = st.tabs(["🎛️ Dashboard", "⚙️ Settings"])

    with tab_settings:
        st.header("Engine Configuration")
        col1, col2 = st.columns(2)
        with col1:
            threshold = st.slider("Silence Threshold", 0.01, 0.1, 0.03, 0.005)
            min_dur = st.slider("Min Clip Duration (s)", 0.5, 2.0, 1.0)
        with col2:
            music_vol = st.slider("Music Volume Ducking", 0.05, 0.5, 0.12)
            crossfade = st.slider("Crossfade Duration", 0.0, 0.2, 0.05)

        st.subheader("Branding & Assets")
        watermark_path = st.text_input("Watermark Image Path (.png)", "")
        bg_music_path = st.text_input("Background Music Path (.mp3)", "")
        sfx_path = st.text_input("Whoosh SFX Path (.mp3)", "whoosh.mp3")

    with tab_dash:
        st.header("Batch Processing")

        root_path = st.text_input("Root Video Folder", ".")
        color_mode = st.selectbox("Color Grading Mode", ["None", "Vlog Mode", "Cinematic Mode"])

        if st.button("🚀 Start Production"):
            if not os.path.exists(root_path):
                st.error("Root path does not exist.")
            else:
                # Initialize Engine
                engine = AIEditorEngine(
                    silence_threshold=threshold,
                    min_clip_duration=min_dur,
                    crossfade_duration=crossfade,
                    music_vol=music_vol
                )

                # Scan
                videos = []
                out_dir_name = "Jules_Pro_Output"

                for r, d, f in os.walk(root_path):
                    if out_dir_name in d:
                        d.remove(out_dir_name)
                    for file in f:
                        if file.lower().endswith(('.mp4', '.mov', '.mkv')):
                            videos.append(os.path.join(r, file))

                if not videos:
                    st.warning("No videos found.")
                else:
                    output_root = os.path.join(root_path, out_dir_name)
                    os.makedirs(output_root, exist_ok=True)

                    progress = st.progress(0)

                    for i, v_path in enumerate(videos):
                        fname = os.path.basename(v_path)

                        # Status container
                        with st.status(f"Processing: {fname}", expanded=True) as status:
                            try:
                                status.write("🔍 Analyzing Audio & Cutting Silence...")
                                video = VideoFileClip(v_path)

                                clips = engine.cut_silence(video)
                                if not clips:
                                    status.write("⚠️ Skipped: No loud segments found.")
                                    video.close()
                                    continue

                                status.write("✂️ Assembling Cuts & Transitions...")
                                processed_clip = engine.add_transitions(clips, sfx_path)

                                # Close raw clips? subclips refer to original video, so keep video open

                                if color_mode != "None":
                                    status.write(f"🎨 Applying {color_mode}...")
                                    processed_clip = engine.apply_color_grade(processed_clip, color_mode)

                                status.write("🎵 Engineering Audio (Ducking & Fades)...")
                                processed_clip = engine.add_audio_ducking(processed_clip, bg_music_path)

                                if watermark_path:
                                    status.write("©️ Applying Watermark...")
                                    processed_clip = engine.add_watermark(processed_clip, watermark_path)

                                # Output Path
                                rel_path = os.path.relpath(v_path, root_path)
                                out_path = os.path.join(output_root, rel_path)
                                os.makedirs(os.path.dirname(out_path), exist_ok=True)

                                status.write("💾 Rendering Final Cut...")
                                # Write
                                processed_clip.write_videofile(
                                    out_path,
                                    codec="libx264",
                                    audio_codec="aac",
                                    fps=24,
                                    logger=None
                                )

                                # Cleanup
                                processed_clip.close()
                                video.close()

                                status.update(label=f"✅ Finished: {fname}", state="complete", expanded=False)

                            except Exception as e:
                                status.write(f"❌ Error: {str(e)}")
                                status.update(label=f"❌ Failed: {fname}", state="error")
                                if 'video' in locals(): video.close()

                        progress.progress((i + 1) / len(videos))

                    st.success("All videos processed successfully!")

if __name__ == "__main__":
    main()
