
import os
import math

class AIEngine:
    def __init__(self, model_size="tiny", device="cpu", compute_type="int8"):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.model = None

    def load_model(self):
        try:
            from faster_whisper import WhisperModel
            # Force CPU usage if needed or auto
            self.model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)
            return True
        except ImportError:
            print("faster_whisper not installed.")
            return False
        except Exception as e:
            print(f"AI Model Load Error: {e}")
            return False

    def transcribe(self, audio_path, language=None):
        if not self.model:
            if not self.load_model():
                return None, None

        try:
            # beam_size=5 is standard
            segments, info = self.model.transcribe(audio_path, language=language, beam_size=5)
            # segments is a generator, convert to list to consume it
            result_segments = list(segments)
            return result_segments, info
        except Exception as e:
            print(f"Transcription Error: {e}")
            return None, None

    def generate_srt(self, segments, output_path):
        def format_timestamp(seconds):
            hours = math.floor(seconds / 3600)
            seconds %= 3600
            minutes = math.floor(seconds / 60)
            seconds %= 60
            millis = round((seconds - math.floor(seconds)) * 1000)
            seconds = math.floor(seconds)
            return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                for i, segment in enumerate(segments):
                    start = format_timestamp(segment.start)
                    end = format_timestamp(segment.end)
                    text = segment.text.strip()
                    f.write(f"{i + 1}\n{start} --> {end}\n{text}\n\n")
            return True
        except Exception as e:
            print(f"SRT Generation Error: {e}")
            return False
