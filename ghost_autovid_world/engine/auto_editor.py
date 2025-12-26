
import os
import numpy as np
try:
    from moviepy.editor import VideoFileClip, concatenate_videoclips, vfx
except ImportError:
    from moviepy import VideoFileClip, concatenate_videoclips
    import moviepy.video.fx as vfx

from ghost_autovid_world.engine.audio_engine import AudioEngine

class AutoEditor:
    def __init__(self):
        self.audio = AudioEngine()

    def _get_face_center_for_segment(self, start, end, face_locations):
        """
        Calculates average face center (x, y) for a time segment.
        face_locations is list of (t, bbox) where bbox is [xmin, ymin, width, height] relative.
        """
        if not face_locations:
            return 0.5, 0.5

        xs = []
        ys = []

        for t, bbox in face_locations:
            if start <= t <= end:
                # Center x = xmin + w/2
                cx = bbox.xmin + bbox.width / 2
                cy = bbox.ymin + bbox.height / 2
                xs.append(cx)
                ys.append(cy)

        if not xs:
            return 0.5, 0.5

        return np.mean(xs), np.mean(ys)

    def process(self, main_meta, b_roll_metas, output_path, status_callback=None):
        """
        Core pipeline: Silence Removal -> Smart Zoom -> B-Roll -> Mastering.
        """
        if status_callback: status_callback("🎬 Loading Main Video...")

        main_clip = VideoFileClip(main_meta["path"])
        main_clip = self.audio.process(main_clip)

        # Metadata
        face_locs = main_meta.get("face_locations", [])

        if status_callback: status_callback("✂️ Cutting Silence...")

        try:
            fs = 22050
            if hasattr(main_clip.audio, 'to_soundarray'):
                arr = main_clip.audio.to_soundarray(fps=fs)
            else:
                arr = None

            cuts = []
            if arr is not None:
                if arr.ndim > 1: arr = arr.mean(axis=1)

                w = int(fs * 0.5)
                pad = (w - len(arr)%w) % w
                arr_pad = np.pad(arr, (0, pad))
                rms = np.sqrt(np.mean(arr_pad.reshape(-1, w)**2, axis=1))

                threshold = np.max(rms) * 0.05
                active = np.convolve(rms > threshold, [1,1,1], 'same') > 0

                fps_w = 22050/w
                start = None
                for i, is_act in enumerate(active):
                    t = i / fps_w
                    if is_act and start is None:
                        start = t
                    elif not is_act and start is not None:
                        if t - start > 1.0:
                            cuts.append((start, t))
                        start = None
                if start is not None:
                    cuts.append((start, main_clip.duration))

            if not cuts: cuts = [(0, main_clip.duration)]

            # Assemble
            final_clips = []
            b_roll_pool = [VideoFileClip(b["path"]) for b in b_roll_metas]
            import random
            random.shuffle(b_roll_pool)

            for i, (s, e) in enumerate(cuts):
                if hasattr(main_clip, 'subclipped'):
                    sub = main_clip.subclipped(s, e)
                else:
                    sub = main_clip.subclip(s, e)

                # Smart Face Zoom
                # Calculate center
                cx, cy = self._get_face_center_for_segment(s, e, face_locs)

                # Apply crop if face found (not center)
                if cx != 0.5 or cy != 0.5:
                    w, h = sub.size
                    zoom_factor = 1.2 # 120% zoom
                    new_w = w / zoom_factor
                    new_h = h / zoom_factor

                    x1 = cx * w - new_w / 2
                    y1 = cy * h - new_h / 2

                    # Clamp
                    x1 = max(0, min(x1, w - new_w))
                    y1 = max(0, min(y1, h - new_h))

                    # Crop & Resize
                    sub = sub.crop(x1=x1, y1=y1, width=new_w, height=new_h).resize((w, h))

                # B-Roll Logic
                if i > 0 and i % 3 == 0 and b_roll_pool:
                    br = b_roll_pool[i % len(b_roll_pool)]
                    if br.duration < sub.duration:
                        try:
                            br = br.loop(duration=sub.duration)
                        except:
                            # v2 compatibility
                            br = vfx.loop(br, duration=sub.duration)
                    else:
                        br = br.subclip(0, sub.duration)

                    br = br.resize(sub.size).without_audio()

                    if hasattr(br, 'set_audio'):
                        mixed = br.set_audio(sub.audio)
                    else:
                        mixed = br.with_audio(sub.audio)
                    final_clips.append(mixed)
                else:
                    final_clips.append(sub)

            final = concatenate_videoclips(final_clips)
            final = self.audio.process(final)

            if status_callback: status_callback("💾 Rendering...")

            from ghost_autovid_world.engine.hardware_manager import HardwareManager
            hw = HardwareManager()

            final.write_videofile(
                output_path,
                codec="libx264",
                audio_codec="aac",
                threads=hw.ffmpeg_threads,
                logger=None
            )

            # Generate dummy captions (Tool constraint)
            srt_path = output_path.replace(".mp4", ".srt")
            with open(srt_path, "w") as f:
                f.write("1\n00:00:00,000 --> 00:00:05,000\n[Captions unavailable due to offline tool constraints]\n")

            main_clip.close()
            for b in b_roll_pool: b.close()
            final.close()

            return True

        except Exception as e:
            print(f"Edit Error: {e}")
            return False
