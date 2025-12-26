
import os
import shutil
import tempfile
import numpy as np

# Handle MoviePy imports for v1 and v2
import moviepy
try:
    # MoviePy v2
    from moviepy import VideoFileClip, AudioFileClip, concatenate_videoclips, CompositeAudioClip
    MOVIEPY_V2 = True
except ImportError:
    # MoviePy v1
    from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips, CompositeAudioClip
    MOVIEPY_V2 = False

def is_silent(chunk, threshold):
    """
    Check if an audio chunk is silent.
    chunk: Audio segment (VideoFileClip or AudioFileClip subclip)
    threshold: Volume threshold (0.0 to 1.0)
    """
    # MoviePy v2 might handle to_soundarray differently or max_volume
    # But generally max_volume is a property or method
    if MOVIEPY_V2:
         # For v2, we might need to access audio array
         # chunk.audio.to_soundarray(fps=44100)
         pass

    # We will use max volume check
    if chunk.audio is None:
        return True

    try:
        # Get max volume. This can be slow if we process frame by frame.
        # Faster approach: iterate over audio in chunks.
        # But here we are passing a small clip (maybe?) No, we need to iterate.
        pass
    except:
        pass
    return False

def process_video(video_path, output_path, threshold=0.03, sfx_path="whoosh.mp3"):
    """
    Process a single video: remove silence and add SFX.
    """
    try:
        video = VideoFileClip(video_path)

        # Audio analysis to find silent spots
        # We will iterate through the audio in small chunks (e.g., 0.1s or 0.5s)
        # However, to be "smart", we want to find continuous silent regions.

        # Method:
        # 1. Convert audio to array
        # 2. Find indices where volume < threshold
        # 3. Identify segments to keep

        # Extract audio
        audio = video.audio
        if audio is None:
            # No audio, just copy? Or skip?
            # Requirement: "Analyze the audio track... Automatically remove silent segments"
            # If no audio, technically all is silence.
            # But usually we keep it if we can't analyze. Or remove all?
            # Let's assume we skip processing if no audio.
            print(f"No audio in {video_path}, copying.")
            video.close()
            shutil.copy2(video_path, output_path)
            return True, "No audio found, copied original."

        # Analysis
        # Get all audio data as array. Warning: Large videos might consume memory.
        # Use a window size.
        fps = 44100
        chunk_size = 0.1 # seconds

        # Using a simpler iteration approach to avoid loading full audio into RAM if possible
        # iterate in 0.1s chunks

        cut_intervals = [] # List of (start, end) to keep

        # We'll just build a list of non-silent clips
        clips = []

        # To avoid complex array logic, let's iterate.
        # A better way for longer videos:
        # Analyze volume in chunks.

        # Let's use `iter_chunks` if available or manual slicing
        # audio.iter_chunks(fps=...)

        # Simple algorithm:
        # 1. Split video into small segments (e.g. 0.5s)? No, that's too choppy.
        # 2. Walk through audio, marking "silent" or "loud".
        # 3. Group continuous "loud" segments.
        # 4. Add "padding" if needed (optional).

        # To implement:
        # We need the max volume over small windows.

        window = 0.2 # Analysis window
        t = 0
        end = video.duration

        loud_segments = []
        current_segment_start = None

        # We need a function to get max volume in a range
        # audio.subclip(t, t+window).max_volume()

        while t < end:
            t_next = min(t + window, end)
            sub = audio.subclipped(t, t_next) if MOVIEPY_V2 else audio.subclip(t, t_next)

            # Check max volume
            # v2 uses max_volume() method on audio clip?
            # v1: audio.max_volume()

            vol = sub.max_volume()

            if vol >= threshold:
                if current_segment_start is None:
                    current_segment_start = t
            else:
                if current_segment_start is not None:
                    loud_segments.append((current_segment_start, t))
                    current_segment_start = None

            t = t_next

        if current_segment_start is not None:
            loud_segments.append((current_segment_start, end))

        if not loud_segments:
            print(f"All silence in {video_path}")
            video.close()
            # Create a short empty video or skip?
            # Maybe keep 1 sec?
            return False, "Video is entirely silent."

        # Build new video
        final_clips = []

        # Load SFX
        try:
            sfx = AudioFileClip(sfx_path)
        except Exception as e:
            print(f"SFX missing or error: {e}. Proceeding without SFX.")
            sfx = None

        for i, (start, end_time) in enumerate(loud_segments):
            # Create subclip
            clip = video.subclipped(start, end_time) if MOVIEPY_V2 else video.subclip(start, end_time)

            # Add SFX at the start (transition) if it's not the very first clip of the original video
            # Requirement: "insert ... at every cut point".
            # This usually means between clips.
            # If i > 0, we are at a cut point.

            if i > 0 and sfx:
                # Composite audio to add sfx at start of this clip
                # We need to ensure clip has audio
                clip_audio = clip.audio
                sfx_dur = sfx.duration

                # We want the sfx to play at t=0 of this clip
                # CompositeAudioClip([clip_audio, sfx])
                # Note: CompositeAudioClip duration is max of inputs.
                # We want to preserve clip duration, sfx just overlays.

                new_audio = CompositeAudioClip([clip_audio, sfx])
                # Ensure duration matches clip
                new_audio = new_audio.with_duration(clip.duration) if MOVIEPY_V2 else new_audio.set_duration(clip.duration)

                clip = clip.with_audio(new_audio) if MOVIEPY_V2 else clip.set_audio(new_audio)

            final_clips.append(clip)

        final_video = concatenate_videoclips(final_clips)

        # Write output
        final_video.write_videofile(output_path, codec="libx264", audio_codec="aac", logger=None)

        # Cleanup
        video.close()
        final_video.close()
        if sfx: sfx.close()

        return True, "Processed successfully."

    except Exception as e:
        return False, str(e)

if __name__ == "__main__":
    # This block is for Streamlit
    import streamlit as st

    st.title("AI Video Editor")

    root_folder = st.text_input("Root Folder Path", ".")
    sensitivity = st.slider("AI Silence Sensitivity", 0.0, 0.1, 0.03, 0.001)

    if st.button("Start Processing"):
        # Logic to scan and process
        st.write("Scanning...")

        video_files = []
        for root, dirs, files in os.walk(root_folder):
            # Exclude output folder
            if "AI_Edited_Results" in dirs:
                dirs.remove("AI_Edited_Results")

            for file in files:
                if file.lower().endswith(('.mp4', '.mov', '.mkv')):
                    video_files.append(os.path.join(root, file))

        if not video_files:
            st.warning("No video files found.")
        else:
            progress_bar = st.progress(0)
            log_expander = st.expander("Processing Log", expanded=True)

            # Create output directory
            output_root = os.path.join(root_folder, "AI_Edited_Results")
            os.makedirs(output_root, exist_ok=True)

            for i, video_path in enumerate(video_files):
                # Calculate relative path to maintain structure?
                # Or just flat? "Exclude an output folder ... to prevent infinite loops"
                # usually implies we write somewhere else or same place.
                # Let's replicate structure inside AI_Edited_Results

                rel_path = os.path.relpath(video_path, root_folder)
                out_path = os.path.join(output_root, rel_path)
                os.makedirs(os.path.dirname(out_path), exist_ok=True)

                status_text = f"Processing {i+1}/{len(video_files)}: {os.path.basename(video_path)}"
                # st.write(status_text)

                success, msg = process_video(video_path, out_path, sensitivity, "whoosh.mp3")

                with log_expander:
                    if success:
                        st.success(f"{os.path.basename(video_path)}: {msg}")
                    else:
                        st.error(f"{os.path.basename(video_path)}: {msg}")

                progress_bar.progress((i + 1) / len(video_files))

            st.success("All done!")
