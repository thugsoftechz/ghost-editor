
import numpy as np
import pyloudnorm as pyln
from moviepy.audio.AudioClip import AudioArrayClip

class AudioEngine:
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
