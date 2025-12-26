
import os
import shutil
import time
import numpy as np

# --- MoviePy Compatibility Layer ---
import moviepy
try:
    # Attempt to import for MoviePy v2
    from moviepy import VideoFileClip, AudioFileClip, CompositeAudioClip, concatenate_videoclips
    import moviepy.video.fx as vfx
    import moviepy.audio.fx as afx
    from moviepy import ColorClip
    MOVIEPY_V2 = True
except ImportError:
    # Fallback to MoviePy v1
    from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip, concatenate_videoclips, ColorClip
    import moviepy.video.fx.all as vfx
    import moviepy.audio.fx.all as afx
    MOVIEPY_V2 = False

def apply_effect(clip, effect_func, *args, **kwargs):
    """
    Applies an effect to a clip handling v1/v2 differences.
    effect_func: For v1, this is the function (e.g., vfx.fadein).
                 For v2, this might be the class or function, but v2 uses clip.with_effects([Effect(...)])
                 OR clip.with_effects([vfx.FadeIn(...)])
    """
    if MOVIEPY_V2:
        # In v2, effects are classes or functions in vfx/afx.
        # Most vfx in v2 are classes like FadeIn, Resize, etc.
        # Usage: clip.with_effects([vfx.FadeIn(duration=1.0)])

        # We need to map the "concept" of the effect to the v2 implementation.
        # This wrapper might be too generic. Let's make specific wrappers.
        pass
    else:
        # v1: clip.fx(effect_func, *args, **kwargs)
        return clip.fx(effect_func, *args, **kwargs)

# Specific Wrappers

def add_video_fadein(clip, duration):
    if MOVIEPY_V2:
        return clip.with_effects([vfx.FadeIn(duration=duration)])
    else:
        return clip.fx(vfx.fadein, duration)

def add_video_fadeout(clip, duration):
    if MOVIEPY_V2:
        return clip.with_effects([vfx.FadeOut(duration=duration)])
    else:
        return clip.fx(vfx.fadeout, duration)

def add_audio_fadein(clip, duration):
    if MOVIEPY_V2:
        # For audio, v2 usually has clip.audio.with_effects(...)
        # But VideoClip.with_effects works on audio too?
        # Let's use audio specific wrapper if possible, or just apply to clip
        # Actually in v2, audio effects are in moviepy.audio.fx
        # And we apply them to the audio clip?
        # clip.audio = clip.audio.with_effects([afx.AudioFadeIn(duration)])
        # But let's see if we can apply to video clip directly
        # VideoClip usually delegates.

        # Safe way:
        if clip.audio is not None:
            new_audio = clip.audio.with_effects([afx.AudioFadeIn(duration=duration)])
            return clip.with_audio(new_audio)
        return clip
    else:
        return clip.audio_fadein(duration)

def add_audio_fadeout(clip, duration):
    if MOVIEPY_V2:
        if clip.audio is not None:
            new_audio = clip.audio.with_effects([afx.AudioFadeOut(duration=duration)])
            return clip.with_audio(new_audio)
        return clip
    else:
        return clip.audio_fadeout(duration)

def add_audio_normalize(clip):
    if clip.audio is None:
        return clip

    if MOVIEPY_V2:
        new_audio = clip.audio.with_effects([afx.AudioNormalize()])
        return clip.with_audio(new_audio)
    else:
        # v1: apply to audio track then set back
        new_audio = clip.audio.fx(afx.audio_normalize)
        return clip.set_audio(new_audio)

def add_color_grading(clip):
    """
    Pop effect: Saturation +20%, Contrast +10%
    """
    # Contrast 1.1 (+10%)
    # Saturation 1.2 (+20%)

    if MOVIEPY_V2:
        # v2 vfx.LumContrast(lum=0, contrast=0.1, contrast_thr=127)
        # v2 vfx.MultiplyColor(factor) ? No that's brightness/tint
        # v2 has vfx.Saturation? No, checked dir(vfx) and it wasn't there explicitly?
        # Checked: ['AccelDecel', ..., 'LumContrast', ..., 'MultiplyColor', ..., 'Painting', ...]
        # Wait, I didn't see Saturation in the list I printed.
        # I saw 'MultiplyColor'.

        # If no built-in Saturation, we use `clip.image_transform` (v2 name for fl_image?)
        # v2 uses `clip.transform(func)`? No.
        # v2 uses `clip.image_transform(func)`?
        # Let's rely on manual numpy.

        def filter_pop(image):
            # image is numpy array (H, W, 3)
            # 1. Contrast
            # (pixel - 128) * 1.1 + 128
            # Careful with types
            img = image.astype(float)
            img = (img - 128.0) * 1.1 + 128.0

            # 2. Saturation
            # Simple approach: Convert to YUV or HSV?
            # Or simpler:
            # gray = 0.299R + 0.587G + 0.114B
            # new_pixel = gray + (pixel - gray) * 1.2

            gray = np.dot(img[..., :3], [0.299, 0.587, 0.114])
            # gray shape (H, W). make it (H, W, 1) or broadcast
            gray = gray[..., np.newaxis]

            img = gray + (img - gray) * 1.2

            # Clip
            img = np.clip(img, 0, 255)
            return img.astype(np.uint8)

        # In v2: clip.transform_video(filter_pop) ?
        # Or clip.image_transform(filter_pop)
        # Check docs/dir later. v2 `VideoClip` has `transform` method which takes a function?
        # Let's try `clip.image_transform` if it exists, else `clip.fl_image`.
        if hasattr(clip, 'image_transform'):
             return clip.image_transform(filter_pop)
        elif hasattr(clip, 'fl_image'):
             return clip.fl_image(filter_pop)
        else:
             # v2 might use effects for this.
             return clip

    else:
        # v1
        # clip.fx(vfx.lum_contrast, contrast=0.1)
        # clip.fx(vfx.colorx, 1.2)? No colorx is brightness.
        # Manual:
        def filter_pop(image):
            img = image.astype(float)
            img = (img - 128.0) * 1.1 + 128.0
            gray = np.dot(img[..., :3], [0.299, 0.587, 0.114])[..., np.newaxis]
            img = gray + (img - gray) * 1.2
            return np.clip(img, 0, 255).astype(np.uint8)

        return clip.fl_image(filter_pop)


# --- Core Logic ---

def process_video(video_path, output_path, sensitivity, enable_grading, enable_norm, sfx_path="whoosh.mp3"):
    """
    Process a single video with the "Pro" pipeline.
    """
    video = None
    final_video = None
    sfx = None

    try:
        video = VideoFileClip(video_path)

        if video.audio is None:
             # Cannot detect silence without audio. Just copy?
             # Or maybe just apply visual effects?
             # Let's assume we skip silence removal if no audio, but apply other "Pro" effects?
             # But "Smart-Cut Engine" is core. If no audio, let's treat as "all silent" -> remove?
             # Or "all loud" -> keep?
             # "Remove segments where volume is below..." -> If no volume, all is below.
             # So technically empty video.
             # Let's be safe and just keep it but apply effects.
             clips = [video]
        else:
            # Silence Detection
            # We need to iterate and find loud chunks.
            # Using v2 or v1 way.

            audio = video.audio
            duration = video.duration
            window = 0.2
            t = 0

            loud_segments = []
            current_start = None

            while t < duration:
                t_end = min(t + window, duration)
                # subclip handling
                if MOVIEPY_V2:
                     sub = audio.subclipped(t, t_end)
                else:
                     sub = audio.subclip(t, t_end)

                # Check volume
                if sub.max_volume() >= sensitivity:
                    if current_start is None:
                        current_start = t
                else:
                    if current_start is not None:
                        loud_segments.append((current_start, t))
                        current_start = None
                t = t_end

            if current_start is not None:
                loud_segments.append((current_start, duration))

            # Filter duration < 1.0s
            # Note: The requirement says "Do not keep clips shorter than 1.0 second"
            # This refers to the RESULTING clips (the loud parts).

            valid_segments = []
            for start, end in loud_segments:
                if (end - start) >= 1.0:
                    valid_segments.append((start, end))

            if not valid_segments:
                # Video is too silent.
                video.close()
                return False, "Video skipped: No useful footage (silence or clips < 1s)."

            clips = []
            for start, end in valid_segments:
                if MOVIEPY_V2:
                    sub = video.subclipped(start, end)
                else:
                    sub = video.subclip(start, end)
                clips.append(sub)

        # Pro Processing per clip
        processed_clips = []
        for clip in clips:
            # 1. Color Grading
            if enable_grading:
                clip = add_color_grading(clip)

            # 2. Audio Norm & Fades
            # Norm
            if enable_norm:
                clip = add_audio_normalize(clip)

            # Fades (0.05s) on audio to prevent clicks
            clip = add_audio_fadein(clip, 0.05)
            clip = add_audio_fadeout(clip, 0.05)

            processed_clips.append(clip)

        # Transitions
        # Insert whoosh between clips
        try:
            if os.path.exists(sfx_path):
                sfx_clip = AudioFileClip(sfx_path)
            else:
                sfx_clip = None
        except:
            sfx_clip = None

        final_clips = []
        for i, clip in enumerate(processed_clips):
            # If not first, add transition logic?
            # "Insert a specific sound effect ... between clips"
            # This usually means mixing the sfx at the junction.
            # Simple approach: Overlay sfx at the start of clip (except first one).

            if i > 0 and sfx_clip:
                # Overlay SFX at t=0 of this clip
                if clip.audio:
                    # v2: CompositeAudioClip([clip.audio, sfx_clip])
                    # Ensure sfx doesn't extend clip
                    s_dur = sfx_clip.duration
                    c_dur = clip.duration

                    # We might need to trim sfx if clip is shorter (unlikely given >1s rule)

                    comp = CompositeAudioClip([clip.audio, sfx_clip])
                    if MOVIEPY_V2:
                         comp = comp.with_duration(c_dur)
                         clip = clip.with_audio(comp)
                    else:
                         comp = comp.set_duration(c_dur)
                         clip = clip.set_audio(comp)

            final_clips.append(clip)

        # Concatenate
        final_video = concatenate_videoclips(final_clips)

        # Cinematography: Fade In/Out (1s) on final
        final_video = add_video_fadein(final_video, 1.0)
        final_video = add_video_fadeout(final_video, 1.0)

        # Write
        # v2 uses logger=None/str, v1 uses logger=None/'bar'
        final_video.write_videofile(output_path, codec="libx264", audio_codec="aac", fps=24, logger=None)

        # Cleanup
        video.close()
        final_video.close()
        if sfx_clip: sfx_clip.close()
        for c in processed_clips:
            c.close()

        return True, "Success"

    except Exception as e:
        # Attempt cleanup
        if video: video.close()
        if final_video: final_video.close()
        return False, str(e)


# --- GUI ---

if __name__ == "__main__":
    import streamlit as st

    st.set_page_config(page_title="Pro Video Editor", layout="wide")

    st.title("🎬 Professional AI Video Editor")

    # Sidebar
    st.sidebar.header("Settings")
    sensitivity = st.sidebar.slider("Silence Sensitivity", 0.01, 0.10, 0.03, 0.01)
    enable_grading = st.sidebar.checkbox("Enable Color Grading", value=True)
    enable_norm = st.sidebar.checkbox("Enable Audio Normalization", value=True)

    # Main
    st.write("Batch process your videos with AI-powered editing and professional post-production effects.")

    root_folder = st.text_input("Root Folder Path", ".")

    if st.button("Start Batch Processing"):
        if not os.path.isdir(root_folder):
            st.error("Invalid folder path.")
        else:
            video_files = []
            output_dir_name = "Pro_Edited_Results"

            # Scan
            st.write("Scanning files...")
            for root, dirs, files in os.walk(root_folder):
                # Exclude output dir
                if output_dir_name in dirs:
                    dirs.remove(output_dir_name)

                for file in files:
                    if file.lower().endswith(('.mp4', '.mov', '.mkv')):
                        video_files.append(os.path.join(root, file))

            if not video_files:
                st.warning("No video files found.")
            else:
                progress_bar = st.progress(0)
                status_area = st.empty()
                log_expander = st.expander("Detailed Log", expanded=True)

                # Create Output Root
                output_root = os.path.join(root_folder, output_dir_name)
                os.makedirs(output_root, exist_ok=True)

                total = len(video_files)
                success_count = 0

                for i, video_path in enumerate(video_files):
                    fname = os.path.basename(video_path)
                    status_area.text(f"Processing {i+1}/{total}: {fname}")

                    # Output path mirroring structure
                    rel_path = os.path.relpath(video_path, root_folder)
                    out_path = os.path.join(output_root, rel_path)
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)

                    # Process
                    success, msg = process_video(
                        video_path,
                        out_path,
                        sensitivity,
                        enable_grading,
                        enable_norm,
                        sfx_path="whoosh.mp3"
                    )

                    with log_expander:
                        if success:
                            st.success(f"✅ {fname}: {msg}")
                            success_count += 1
                        else:
                            st.error(f"❌ {fname}: {msg}")

                    progress_bar.progress((i + 1) / total)

                status_area.text(f"Completed! {success_count}/{total} videos processed successfully.")
                st.success("Batch Processing Finished.")
