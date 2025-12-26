
import os
import numpy as np
import pyloudnorm as pyln
try:
    from moviepy.editor import AudioFileClip
except ImportError:
    from moviepy import AudioFileClip
from moviepy.audio.AudioClip import AudioArrayClip

class AudioEngine:
    def process_main_audio(self, clip):
        """
        Normalizes audio to -14 LUFS.
        Returns a clip with processed audio.
        """
        if clip.audio is None:
            return clip

        try:
            # 1. To Array
            fs = 44100
            # Try/Except for MoviePy version differences
            try:
                sarray = clip.audio.to_soundarray(fps=fs)
            except:
                return clip

            if sarray.ndim == 1:
                # Monovto stereo
                sarray = np.column_stack((sarray, sarray))

            # 2. Measure
            meter = pyln.Meter(fs)
            loudness = meter.integrated_loudness(sarray)

            # 3. Normalize
            target_lufs = -14.0
            if loudness > -100: # Avoid normalizing pure silence
                normalized_audio = pyln.normalize.loudness(sarray, loudness, target_lufs)

                # Check peaks
                peak = np.max(np.abs(normalized_audio))
                if peak > 1.0:
                    normalized_audio = normalized_audio / peak # Hard limit

                # 4. Re-attach
                new_audio = AudioArrayClip(normalized_audio, fps=fs)
                if hasattr(clip, 'set_audio'):
                    return clip.set_audio(new_audio)
                else:
                    return clip.with_audio(new_audio)

        except Exception as e:
            print(f"⚠️ Audio processing failed: {e}")

        return clip
