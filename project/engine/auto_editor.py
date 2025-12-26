
import os
import random
import numpy as np
try:
    from moviepy.editor import VideoFileClip, concatenate_videoclips, vfx
except ImportError:
    from moviepy import VideoFileClip, concatenate_videoclips
    import moviepy.video.fx as vfx

from project.engine.audio_engine import AudioEngine
import mediapipe as mp
import cv2

class AutoEditor:
    def __init__(self):
        self.audio_engine = AudioEngine()
        try:
            import mediapipe.python.solutions.face_detection as face_detection
            self.mp_face_detection = face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.5
            )
        except:
             try:
                 self.mp_face_detection = mp.solutions.face_detection.FaceDetection(
                    model_selection=1, min_detection_confidence=0.5
                 )
             except:
                 self.mp_face_detection = None

    def _get_face_center(self, frame):
        if not self.mp_face_detection:
             return 0.5, 0.5
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_face_detection.process(rgb)
        if results.detections:
            # First face
            bbox = results.detections[0].location_data.relative_bounding_box
            cx = bbox.xmin + bbox.width/2
            cy = bbox.ymin + bbox.height/2
            return cx, cy
        return 0.5, 0.5

    def edit(self, main_meta, b_roll_metas, output_path):
        """
        Assembles the final video.
        1. Load Main.
        2. Remove Silence (Simple volume threshold cut).
        3. Insert B-Roll at cuts.
        4. Audio Polish.
        5. Write.
        """
        print("🎬 Starting Auto-Edit...")

        main_clip = VideoFileClip(main_meta["path"])

        # 1. Audio Polish First (to help silence detection?)
        # Actually better to polish at end, but we need normalized audio for consistent threshold.
        # Let's normalize first.
        main_clip = self.audio_engine.process_main_audio(main_clip)

        # 2. Silence Removal / Jump Cutting
        # We'll chop the video into 'loud' segments
        # Heuristic: Scan audio in 0.5s chunks.
        try:
            # Threshold in RMS. -14LUFS is approx 0.2 amplitude?
            # Let's use relative threshold.
            audio_arr = main_clip.audio.to_soundarray(fps=22050)
            # convert stereo to mono for analysis
            if audio_arr.ndim > 1: audio_arr = audio_arr.mean(axis=1)

            # Simple gate
            window_size = int(22050 * 0.5) # 0.5s
            threshold = np.max(np.abs(audio_arr)) * 0.05 # 5% of peak

            loud_segments = []
            is_loud = False
            start_t = 0

            # Iterate windows
            # Using numpy reshape for speed
            # Pad to multiple of window
            pad_len = (window_size - (len(audio_arr) % window_size)) % window_size
            padded = np.pad(audio_arr, (0, pad_len))
            reshaped = padded.reshape(-1, window_size)
            rms_vals = np.sqrt(np.mean(reshaped**2, axis=1))

            # Identify active windows
            active_mask = rms_vals > threshold
            # Simple smoothing (fill gaps of 1 window)
            active_mask = np.convolve(active_mask, [1,1,1], 'same') > 0

            # Convert back to time ranges
            fps_window = 22050 / window_size # approx 2 fps

            current_start = None
            for i, active in enumerate(active_mask):
                t = i / fps_window
                if active:
                    if current_start is None: current_start = t
                else:
                    if current_start is not None:
                        # End of segment
                        loud_segments.append((current_start, t))
                        current_start = None
            if current_start is not None:
                loud_segments.append((current_start, main_clip.duration))

        except Exception as e:
            print(f"⚠️ Silence removal failed ({e}), using full clip.")
            loud_segments = [(0, main_clip.duration)]

        # 3. Assemble & B-Roll Overlay
        final_clips = []
        b_roll_pool = [VideoFileClip(b["path"]) for b in b_roll_metas]
        # Shuffle b-roll
        import random
        random.shuffle(b_roll_pool)
        b_roll_idx = 0

        print(f"✂️ Created {len(loud_segments)} segments from main video.")

        for i, (start, end) in enumerate(loud_segments):
            # Enforce min duration
            if end - start < 1.0: continue

            if hasattr(main_clip, 'subclipped'):
                sub = main_clip.subclipped(start, end)
            else:
                sub = main_clip.subclip(start, end)

            # Auto Zoom/Center (Simple: check middle frame)
            # Only do it occasionally or if face is far off center?
            # For robustness, let's just do a mild zoom (1.05x) to breathe
            # "Auto zoom when face detected"
            try:
                mid_t = (end - start) / 2
                frame = sub.get_frame(mid_t)
                cx, cy = self._get_face_center(frame)

                # If face is detected
                if cx != 0.5:
                    # Crop logic: Center around cx, cy
                    w, h = sub.size
                    # Zoom factor 1.2
                    zoom = 1.2
                    new_w, new_h = w/zoom, h/zoom

                    x1 = cx*w - new_w/2
                    y1 = cy*h - new_h/2

                    # Clamp
                    x1 = max(0, min(x1, w - new_w))
                    y1 = max(0, min(y1, h - new_h))

                    sub = sub.crop(x1=x1, y1=y1, width=new_w, height=new_h).resize((w, h))
            except:
                pass

            # Insert B-Roll between segments?
            # Logic: "Insert b-roll during speech gaps"
            # Since we removed the gaps, we now have a jump cut.
            # We cover the jump cut with B-roll.
            # We overlay B-roll over the *end* of prev clip and *start* of this clip?
            # Or just insert it?
            # Prompt: "Insert b-roll during speech gaps".
            # Usually means visual fill.
            # Simplified: Overlay B-roll for 3 seconds centered on the cut.
            # BUT: We are concatenating.

            final_clips.append(sub)

            # Add B-Roll covering the transition?
            # CompositeVideoClip is complex for concatenation flow.
            # Alternative: Just insert B-roll clip if audio allows?
            # But B-roll usually has no speech.
            # Let's keep it simple: Jump cuts.
            # If we have B-roll, we can replace visual of Main with B-roll audio-muted?
            # Let's do: Every 3rd clip, overlay B-roll visual on top of Main audio.

            if i % 3 == 0 and b_roll_pool:
                # Get b-roll
                br = b_roll_pool[b_roll_idx % len(b_roll_pool)]
                b_roll_idx += 1

                # Loop/Cut b-roll to match subclip duration
                if br.duration < sub.duration:
                    br = vfx.loop(br, duration=sub.duration)
                else:
                    br = br.subclip(0, sub.duration)

                br = br.resize(sub.size).set_audio(None)

                # Composite: Main Audio, B-roll Video
                # Wait, we want main audio.
                combined = sub.set_audio(sub.audio) # Keep main audio
                # Replace video?
                # We want B-Roll video + Main Audio
                # sub is (MainVid + MainAudio)
                # We want (BRollVid + MainAudio)

                replaced = br.set_audio(sub.audio)
                final_clips[-1] = replaced # Replace the last added clip

        # Concat
        if not final_clips:
            final_video = main_clip
        else:
            final_video = concatenate_videoclips(final_clips)

        # Write
        final_video.write_videofile(output_path, codec="libx264", audio_codec="aac", fps=24, logger=None)

        # Cleanup
        main_clip.close()
        for b in b_roll_pool: b.close()
        final_video.close()

        print("✅ Auto-Edit Complete.")
