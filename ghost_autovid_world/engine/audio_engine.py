
import os
import numpy as np
import pyloudnorm as pyln
from moviepy.audio.AudioClip import AudioArrayClip

class AudioEngine:
    def process_file(self, input_path, output_path, target_lufs=-14.0):
        try:
            # Safe import
            try:
                from moviepy.editor import VideoFileClip, AudioFileClip
            except ImportError:
                from moviepy import VideoFileClip, AudioFileClip

            # Check if video or audio
            ext = os.path.splitext(input_path)[1].lower()
            is_video = ext in ['.mp4', '.mov', '.avi', '.mkv']

            if is_video:
                clip = VideoFileClip(input_path)
            else:
                clip = AudioFileClip(input_path)

            processed_clip = self.process(clip, target_lufs=target_lufs)

            # Write output
            if is_video:
                # Use threads=1 for safety if not specified, or just default
                processed_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
            else:
                processed_clip.write_audiofile(output_path)

            clip.close()
            if processed_clip != clip:
                processed_clip.close()
            return True
        except Exception as e:
            print(f"Audio Process File Error: {e}")
            return False

    def process(self, clip, target_lufs=-14.0):
        if clip.audio is None: return clip

        try:
            fs = 44100
            # Get array
            try:
                sarray = clip.audio.to_soundarray(fps=fs)
            except:
                return clip

            if sarray.ndim == 1:
                sarray = np.column_stack((sarray, sarray))

            # Meter
            meter = pyln.Meter(fs)
            loudness = meter.integrated_loudness(sarray)

            # Normalize
            if loudness > -70: # Don't boost noise
                gain_db = target_lufs - loudness
                # Limit gain to avoid explosion
                if gain_db > 20: gain_db = 20

                gain_lin = 10 ** (gain_db / 20.0)
                new_arr = sarray * gain_lin

                # Limiter (-1 dBTP)
                peak = np.max(np.abs(new_arr))
                if peak > 0.89: # approx -1dB
                    new_arr = new_arr * (0.89 / peak)

                new_audio = AudioArrayClip(new_arr, fps=fs)

                if hasattr(clip, 'set_audio'):
                    return clip.set_audio(new_audio)
                else:
                    return clip.with_audio(new_audio)

        except Exception as e:
            print(f"Audio Engine Error: {e}")

        return clip
